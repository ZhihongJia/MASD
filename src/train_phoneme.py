import numpy as np
import argparse
import os
import torch
import random
from EEGNet import *
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data=None, feature=None, label=None):
        super().__init__()
        self.data = data
        self.feature = feature
        self.label = label
    def __getitem__(self, idx):
        if self.feature is None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx], self.feature[idx], self.label[idx]
    def __len__(self):
        return self.data.shape[0]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def top_k(true_label, pred_topk, k):
    expanded_true_label = np.tile(true_label.reshape(-1, 1), (1, k))
    correct_predictions = np.sum(np.equal(pred_topk, expanded_true_label), axis=1)
    topk_acc = np.mean(correct_predictions)
    return topk_acc

def InfoNCE(outputs, features, temperature, device):
    sample_num = outputs.shape[0]
    outputs = F.normalize(outputs.reshape(sample_num, -1))
    features = F.normalize(features.reshape(sample_num, -1))
    inner_product = torch.mm(outputs, features.T)
    inner_product = inner_product / temperature
    loss = F.cross_entropy(inner_product, torch.tensor(range(sample_num)).to(device))
    return loss

def train(Model, dataloader_train, dataloader_val, model_path):
    criterion_ce = torch.nn.CrossEntropyLoss()
    model = Model.to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_loss = np.inf
    last_best_epoch = 0

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0
        pred_1, ground = [], []
        for batch_id, traindata in enumerate(dataloader_train):
            inputs, wav_features, labels = traindata
            inputs = inputs.to(torch.float32).to(device)
            wav_features = wav_features.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            outputs, label_pred = model(inputs)
            loss_clip = InfoNCE(outputs, wav_features, args.temperature, device)  # (48, 120)
            loss_ce = criterion_ce(label_pred, labels)
            loss = loss_clip + LAMB * loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            train_loss += loss.detach().cpu().item()
            
            del inputs, labels, label_pred, wav_features, outputs

        train_loss = train_loss/(batch_id+1)

        pred_1 = np.concatenate(pred_1)
        ground = np.concatenate(ground)
        acc_1 = accuracy_score(ground, pred_1)*100
        bca_1 = balanced_accuracy_score(ground, pred_1)*100
        val_loss, val_acc1, val_bca1 = eval(model, dataloader_val)

        if val_loss < model_loss:
            model_loss = val_loss
            last_best_epoch = epoch
            torch.save(model, os.path.join(model_path, 'net.pkl'))
        if epoch - last_best_epoch > args.patience:
            break
    del model
    torch.cuda.empty_cache()
        

def eval(model, dataloader_val):
    torch.cuda.empty_cache()
    criterion_ce = torch.nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    pred_1, ground = [], []
    with torch.no_grad():
        for batch_id, valdata in enumerate(dataloader_val):
            inputs, wav_features, labels = valdata
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            outputs, label_pred = model(inputs) 
            loss = criterion_ce(label_pred, labels)

            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            val_loss += loss.detach().cpu().item()

            del inputs, labels, label_pred, wav_features, outputs

        val_loss = val_loss/(batch_id+1)
        pred_1 = np.concatenate(pred_1)
        ground = np.concatenate(ground)
        acc_1 = accuracy_score(ground, pred_1)*100
        bca_1 = balanced_accuracy_score(ground, pred_1)*100
    return val_loss, acc_1, bca_1
    
def test(model_path, dataloader_test):
    torch.cuda.empty_cache()
    criterion_ce = torch.nn.CrossEntropyLoss()
    net_path = os.path.join(model_path, 'net.pkl')
    model = torch.load(net_path).to(device)
    model.eval()
    test_loss = 0
    pred_1, ground = [], []
    with torch.no_grad():
        for batch_id, testdata in enumerate(dataloader_test):
            inputs, wav_features, labels = testdata
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            outputs, label_pred = model(inputs)
            loss = criterion_ce(label_pred, labels)

            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            test_loss += loss.detach().cpu().item()

            del inputs, labels, label_pred, wav_features, outputs
        
        test_loss = test_loss/(batch_id+1)
        pred_1 = np.concatenate(pred_1)
        ground = np.concatenate(ground)
        acc_1 = accuracy_score(ground, pred_1)*100
        bca_1 = balanced_accuracy_score(ground, pred_1)*100
    del model  
    return test_loss, acc_1, bca_1

def load_sound_features(label, fea_path):
    with open(fea_path, 'rb') as f:
        data_dict = pickle.load(f)
    features = []
    for word in label:
        features.append(data_dict[word])
    features = np.concatenate(features)
    return features

def meg_dataset(subject_id='S01', channel='read', label_type='word', root='./DATA_MEG'):
    meg_file_path = root + f'/{subject_id}/processed/{channel}/meg/trial.npy'
    label_file_path = root + f'/{subject_id}/processed/{channel}/trans_label/trans_label.npz'
    print('loading:----{}----'.format(subject_id))
    data = np.load(meg_file_path)
    label = np.load(label_file_path)[label_type]
    label_key = np.load(label_file_path)['key']
    sound_feature = load_sound_features(label_key, fea_path=os.path.join(os.path.dirname(__file__), f'./feature/wav_feature_{args.wav_type}.pkl'))
    return data, label, sound_feature

def run(seed):
    global LAMB
    data, label, feature = meg_dataset(subject_id=args.subject, channel=args.channel, label_type=args.label_type, root=args.root_path)
    INCHANNELS = data.shape[1]
    feature = np.mean(feature, axis=-1)
    if args.wav_type == 'mel':
        F1_wav = 4
        D_wav = 3
        F2_wav = 12
        LAMB=6
    elif args.wav_type == 'wav2vec':
        F1_wav = 17
        D_wav = 6
        F2_wav = 102
        LAMB=2
        feature = feature[:, :1020]
    elif args.wav_type == 'hubert':
        F1_wav = 19
        D_wav = 4
        F2_wav = 76
        LAMB=2
        feature = feature[:, :760]
    
    skf = StratifiedKFold(n_splits=args.kflod, shuffle=True, random_state=seed)
    test_acc1_flod, test_bca1_flod = [], []
    for flod_num, (train_val_index, test_index) in enumerate(skf.split(data, label), 1):
        meg_train_val, meg_read_test = data[train_val_index], data[test_index]
        feature_train_val, feature_read_test = feature[train_val_index], feature[test_index]
        label_train_val, label_read_test = label[train_val_index], label[test_index]
        meg_read_train, meg_read_val, feature_read_train, feature_read_val, label_read_train, label_read_val = train_test_split(meg_train_val, feature_train_val, label_train_val, test_size=2/12, random_state=seed, shuffle=True, stratify=label_train_val)
    
        model_path = os.path.join(os.path.dirname(__file__), f'model_phoneme/{args.model}/{args.label_type}/{args.wav_type}/{args.subject}_{args.channel}_{seed}_{flod_num}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        classes_num = len(np.unique(label_read_train))
        Model = EEGNet(n_classes=classes_num, Chans=INCHANNELS, Samples=args.samples, kernLenght=100, F1=F1_wav, D=D_wav, F2=F2_wav, dropoutRate=args.dropout, norm_rate=0.5)
        dataloader_train = torch.utils.data.DataLoader(Dataset(meg_read_train, feature_read_train, label_read_train), batch_size = args.bs, shuffle=True, drop_last=True)
        dataloader_val = torch.utils.data.DataLoader(Dataset(meg_read_val, feature_read_val, label_read_val), batch_size = args.bs, shuffle=False, drop_last=True)
        dataloader_test = torch.utils.data.DataLoader(Dataset(meg_read_test, feature_read_test, label_read_test), batch_size = args.bs, shuffle=False, drop_last=True)
        del meg_read_train, meg_read_val, meg_read_test, feature_read_train, feature_read_val, feature_read_test, label_read_train, label_read_val, label_read_test
        train(Model, dataloader_train, dataloader_val, model_path)
        test_loss, test_acc1, test_bca1 = test(model_path, dataloader_test)
        test_acc1_flod.append(test_acc1)
        test_bca1_flod.append(test_bca1)
    acc1_flod_avg = np.mean(test_acc1_flod)
    bca1_flod_avg = np.mean(test_bca1_flod)
    for i in range(args.kflod):
        print(f'flod_{i+1}: acc1_flod: {test_acc1_flod[i]}     bca1_flod: {test_bca1_flod[i]}')
    print(f'--------------------------------------------------------------------------------------')
    print(f'acc1_flod_avg: {acc1_flod_avg}     bca1_flod_avg: {bca1_flod_avg}')
    print('\n')
    return acc1_flod_avg, bca1_flod_avg

def rep_experiment(seed_list=[2024]):
    acc1, bca1 = [], []
    for seed in seed_list:
        print(f'*********************************************{seed}************************************************')
        set_seed(seed)
        test_acc1, test_bca1 = run(seed)
        acc1.append(test_acc1)
        bca1.append(test_bca1)
    acc1_avg = np.mean(acc1)
    bca1_avg = np.mean(bca1)
    print('\n')
    print('\n')
    print('*************************************************************************************************')
    print('*************************************************************************************************')
    for i in range(len(seed_list)):
        print(f'seed: {seed_list[i]}    acc1: {acc1[i]}   bca1: {bca1[i]}')
    print('********************************************over*************************************************')
    print(f'acc1_avg: {acc1_avg}     bca1_avg: {bca1_avg}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu_id')
    parser.add_argument('--subject', type=str, default='S01', help='subject name')
    parser.add_argument('--root_path', type=str, default='../DATA', help='root_path')
    parser.add_argument('--channel', type=str, default='read', help='channel_path')
    parser.add_argument('--label_type', type=str, default='initial_class', help='label_type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--patience', type=int, default=50, help='early stop')
    parser.add_argument('--bs', type=int, default=48, help='batchsize')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--samples', type=int, default=320, help='sample num')
    parser.add_argument('--wav_type', type=str, default='mel', help='wav_type')
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature')
    parser.add_argument('--model', type=str, default='eegnet', help='model')
    parser.add_argument('--kflod', type=int, default=5, help='kflod')
    args = parser.parse_args()

    args_dict = vars(args)
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    torch.cuda.set_device(args.gpu_id)
    torch.set_num_threads(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}+{torch.cuda.current_device()}')

    rep_experiment(seed_list=list(range(2024, 2024 + 20)))

