import numpy as np
import argparse
import os
import torch
import random
from EEGNet_word_wav import *
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data=None, sound_feature=None, word_feature=None, label=None):
        super().__init__()
        self.data = data
        self.sound_feature = sound_feature
        self.word_feature = word_feature
        self.label = label
    def __getitem__(self, idx):
        if self.sound_feature is None and self.word_feature is None:
            return self.data[idx], self.label[idx]
        elif self.word_feature is None:
            return self.data[idx], self.sound_feature[idx], self.label[idx]
        else:
            return self.data[idx], self.sound_feature[idx], self.word_feature[idx], self.label[idx]
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
        pred_1, pred_k1, ground = [], [], []
        for batch_id, traindata in enumerate(dataloader_train):
            inputs, wav_features, word_features, labels = traindata
            inputs = inputs.to(torch.float32).to(device)
            wav_features = wav_features.to(torch.float32).to(device)
            word_features = word_features.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            outputs_wav, outputs_word, label_pred = model(inputs)
            loss_wav_clip = InfoNCE(outputs_wav, wav_features, args.temperature, device)  
            loss_word_clip = InfoNCE(outputs_word, word_features, args.temperature, device)  
            loss_ce = criterion_ce(label_pred, labels)
            loss = LAMB_WAV*loss_wav_clip + LAMB_WORD*loss_word_clip + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            pred_k1.append(torch.topk(label_pred, k=args.k1, dim=1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            train_loss += loss.detach().cpu().item()
            
            del inputs, labels, label_pred, wav_features, word_features, outputs_wav, outputs_word

        train_loss = train_loss/(batch_id+1)

        pred_1 = np.concatenate(pred_1)
        pred_k1 = np.concatenate(pred_k1)
        ground = np.concatenate(ground)
        acc_1 = accuracy_score(ground, pred_1)*100
        acc_k1 = top_k(ground, pred_k1, args.k1)*100
        bca_1 = balanced_accuracy_score(ground, pred_1)*100
        val_loss, val_acck1, val_acc1, val_bca1 = eval(model, dataloader_val)

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
    pred_1, pred_k1, ground = [], [], []
    with torch.no_grad():
        for batch_id, valdata in enumerate(dataloader_val):
            inputs, wav_features, word_features, labels = valdata
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            outputs_wav, outputs_word, label_pred = model(inputs) 
            loss = criterion_ce(label_pred, labels)

            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            pred_k1.append(torch.topk(label_pred, k=args.k1, dim=1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            val_loss += loss.detach().cpu().item()

            del inputs, labels, label_pred, wav_features, word_features, outputs_wav, outputs_word

        val_loss = val_loss/(batch_id+1)
        pred_1 = np.concatenate(pred_1)
        pred_k1 = np.concatenate(pred_k1)
        ground = np.concatenate(ground)
        acc_1 = accuracy_score(ground, pred_1)*100
        acc_k1 = top_k(ground, pred_k1, args.k1)*100
        bca_1 = balanced_accuracy_score(ground, pred_1)*100
    return val_loss, acc_k1, acc_1, bca_1
    
def test(model_path, dataloader_test):
    torch.cuda.empty_cache()
    criterion_ce = torch.nn.CrossEntropyLoss()
    net_path = os.path.join(model_path, 'net.pkl')
    model = torch.load(net_path).to(device)
    model.eval()
    test_loss = 0
    pred_1, pred_k1, ground = [], [], []
    with torch.no_grad():
        for batch_id, testdata in enumerate(dataloader_test):
            inputs, wav_features, word_features, labels = testdata
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            outputs_wav, outputs_word, label_pred = model(inputs)
            loss = criterion_ce(label_pred, labels)

            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            pred_k1.append(torch.topk(label_pred, k=args.k1, dim=1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            test_loss += loss.detach().cpu().item()

            del inputs, labels, label_pred, wav_features, word_features, outputs_wav, outputs_word
        
        test_loss = test_loss/(batch_id+1)
        pred_1 = np.concatenate(pred_1)
        pred_k1 = np.concatenate(pred_k1)
        ground = np.concatenate(ground)
        acc_1 = accuracy_score(ground, pred_1)*100
        acc_k1 = top_k(ground, pred_k1, args.k1)*100
        bca_1 = balanced_accuracy_score(ground, pred_1)*100
    del model  
    return test_loss, acc_k1, acc_1, bca_1

def load_sound_features(label, fea_path):
    with open(fea_path, 'rb') as f:
        data_dict = pickle.load(f)
    features = []
    for word in label:
        features.append(data_dict[word])
    features = np.concatenate(features)
    return features

def load_word_features(label, fea_path):
    with open(fea_path, 'rb') as f:
        data_dict = pickle.load(f)
    features = []
    for word in label:
        features.append(data_dict[word])
    features = np.array(features)
    return features

def meg_dataset_crosssub(test_subject_id='S01', channel='read', label_type='word', sub_list=['S01', 'S02', 'S03', 'S07', 'S08', 'S09'], seed=2024, root='./DATA_MEG'):
    data_train, data_val, sound_train, sound_val, word_train, word_val, label_train, label_val = [], [], [], [], [], [], [], []
    for sub in sub_list:
        meg_file_path = root + f'/{sub}/processed/{channel}/meg/trial.npy'
        label_file_path = root + f'/{sub}/processed/{channel}/trans_label/trans_label.npz'
        print('loading:----{}----'.format(sub))
        if sub != test_subject_id:
            data_sub = np.load(meg_file_path)
            label_sub = np.load(label_file_path)[label_type]
            label_key_sub = np.load(label_file_path)['key']
            sound_sub = load_sound_features(label_key_sub, fea_path=os.path.join(os.path.dirname(__file__), f'./feature/wav_feature_{args.wav_type}.pkl'))
            word_sub = load_word_features(label_key_sub, fea_path=os.path.join(os.path.dirname(__file__), f'./feature/word_feature_{args.word_type}.pkl'))
            data_sub_train, data_sub_val, sound_sub_train, sound_sub_val, word_sub_train, word_sub_val, label_sub_train, label_sub_val = train_test_split(data_sub, sound_sub, word_sub, label_sub, test_size=1/(len(sub_list)-1), random_state=seed, shuffle=True, stratify=label_sub)
            data_train.append(data_sub_train)
            sound_train.append(sound_sub_train)
            word_train.append(word_sub_train)
            label_train.append(label_sub_train)
            data_val.append(data_sub_val)
            sound_val.append(sound_sub_val)
            word_val.append(word_sub_val)
            label_val.append(label_sub_val)
        elif sub == test_subject_id:
            data_test = np.load(meg_file_path)
            label_test = np.load(label_file_path)[label_type]
            label_key_test = np.load(label_file_path)['key']
            sound_test = load_sound_features(label_key_test, fea_path=os.path.join(os.path.dirname(__file__), f'./feature/wav_feature_{args.wav_type}.pkl'))
            word_test = load_word_features(label_key_test, fea_path=os.path.join(os.path.dirname(__file__), f'./feature/word_feature_{args.word_type}.pkl'))
    data_train = np.concatenate(data_train, axis=0)
    sound_train = np.concatenate(sound_train, axis=0)
    word_train = np.concatenate(word_train, axis=0)
    label_train = np.concatenate(label_train, axis=0)
    data_val = np.concatenate(data_val, axis=0)
    sound_val = np.concatenate(sound_val, axis=0)
    word_val = np.concatenate(word_val, axis=0)
    label_val = np.concatenate(label_val, axis=0)
    return  data_train, sound_train, word_train, label_train, data_val, sound_val, word_val, label_val, data_test, sound_test, word_test, label_test

def run(seed):
    global LAMB_WAV, LAMB_WORD
    meg_read_train, sound_feature_read_train, word_feature_read_train, label_read_train, meg_read_val, sound_feature_read_val, word_feature_read_val, label_read_val, meg_read_test, sound_feature_read_test, word_feature_read_test, label_read_test = meg_dataset_crosssub(test_subject_id=args.subject, channel=args.channel, label_type=args.label_type, sub_list=args.sub_list, seed=seed, root=args.root_path)
    INCHANNELS = meg_read_train.shape[1]
    sound_feature_read_train, sound_feature_read_val, sound_feature_read_test = np.mean(sound_feature_read_train, axis=-1), np.mean(sound_feature_read_val, axis=-1), np.mean(sound_feature_read_test, axis=-1)
    if args.wav_type == 'mel':
        F1_wav = 4
        D_wav = 3
        F2_wav = 12
        LAMB_WAV=1/6
    elif args.wav_type == 'wav2vec':
        F1_wav = 17
        D_wav = 6
        F2_wav = 102
        LAMB_WAV=0.5
        sound_feature_read_train, sound_feature_read_val, sound_feature_read_test = sound_feature_read_train[:, :1020], sound_feature_read_val[:, :1020], sound_feature_read_test[:, :1020]
    elif args.wav_type == 'hubert':
        F1_wav = 19
        D_wav = 4
        F2_wav = 76
        LAMB_WAV=0.5
        sound_feature_read_train, sound_feature_read_val, sound_feature_read_test = sound_feature_read_train[:, :760], sound_feature_read_val[:, :760], sound_feature_read_test[:, :760]
    if args.word_type == 'fasttext':
        F1_word = 6
        D_word = 5
        F2_word = 30
        LAMB_WORD=0.25
    elif args.word_type == 'bert':
        F1_word = 19
        D_word = 4
        F2_word = 76
        LAMB_WORD=1
        word_feature_read_train, word_feature_read_val, word_feature_read_test = word_feature_read_train[:, :760], word_feature_read_val[:, :760], word_feature_read_test[:, :760]
    
    model_path = os.path.join(os.path.dirname(__file__), f'model_cross/{args.model}/{args.label_type}/{args.wav_type}_{args.word_type}/{args.subject}_{args.channel}_{seed}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    classes_num = len(np.unique(label_read_train))
    Model = EEGNet_word_wav(n_classes=classes_num, Chans=INCHANNELS, Samples=args.samples, kernLenght=100, F1_wav=F1_wav, D_wav=D_wav, F2_wav=F2_wav, F1_word=F1_word, D_word=D_word, F2_word=F2_word, dropoutRate=args.dropout, norm_rate=0.5)
    dataloader_train = torch.utils.data.DataLoader(Dataset(meg_read_train, sound_feature_read_train, word_feature_read_train, label_read_train), batch_size = args.bs, shuffle=True, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(Dataset(meg_read_val, sound_feature_read_val, word_feature_read_val, label_read_val), batch_size = args.bs, shuffle=False, drop_last=True)
    dataloader_test = torch.utils.data.DataLoader(Dataset(meg_read_test, sound_feature_read_test, word_feature_read_test, label_read_test), batch_size = args.bs, shuffle=False, drop_last=True)
    del meg_read_train, meg_read_val, meg_read_test, sound_feature_read_train, sound_feature_read_val, sound_feature_read_test, word_feature_read_train, word_feature_read_val, word_feature_read_test, label_read_train, label_read_val, label_read_test
    train(Model, dataloader_train, dataloader_val, model_path)
    test_loss, test_acck1, test_acc1, test_bca1 = test(model_path, dataloader_test)
    return test_acck1, test_acc1, test_bca1

def rep_experiment(seed_list=[2024]):
    acck1, acc1, bca1 = [], [], []
    for seed in seed_list:
        print(f'*********************************************{seed}************************************************')
        set_seed(seed)
        test_acck1, test_acc1, test_bca1 = run(seed)
        acck1.append(test_acck1)
        acc1.append(test_acc1)
        bca1.append(test_bca1)
    acck1_avg = np.mean(acck1)
    acc1_avg = np.mean(acc1)
    bca1_avg = np.mean(bca1)
    print('\n')
    print('\n')
    print('*************************************************************************************************')
    print('*************************************************************************************************')
    for i in range(len(seed_list)):
        print(f'seed: {seed_list[i]}    acc{args.k1}: {acck1[i]}    acc1: {acc1[i]}   bca1: {bca1[i]}')
    print('********************************************over*************************************************')
    print(f'acc{args.k1}_avg: {acck1_avg}    acc1_avg: {acc1_avg}     bca1_avg: {bca1_avg}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu_id')
    parser.add_argument('--subject', type=str, default='S01', help='subject_test')
    parser.add_argument('--sub_list', nargs='+', default=['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09'], help='sub_list')
    parser.add_argument('--root_path', type=str, default='../DATA', help='root_path')
    parser.add_argument('--channel', type=str, default='read_cross', help='channel_path')
    parser.add_argument('--label_type', type=str, default='word', help='label_type') 
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--patience', type=int, default=50, help='early stop')
    parser.add_argument('--bs', type=int, default=720, help='batchsize')
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    parser.add_argument('--samples', type=int, default=320, help='sample num')
    parser.add_argument('--model', type=str, default='eegnet_word_wav', help='model')
    parser.add_argument('--wav_type', type=str, default='mel', help='wav_type')
    parser.add_argument('--word_type', type=str, default='fasttext', help='wav_type')  
    parser.add_argument('--temperature', type=float, default=0.01, help='temperature')
    parser.add_argument('--kflod', type=int, default=5, help='kflod')
    parser.add_argument('--k1', type=int, default=5, help='topk1')
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


