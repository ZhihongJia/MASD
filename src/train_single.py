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
import distutils

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label, feature=None):
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
            inputs, labels = traindata
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            _, label_pred = model(inputs)
            loss = criterion_ce(label_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            pred_k1.append(torch.topk(label_pred, k=args.k1, dim=1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())
            
            train_loss += loss.detach().cpu().item()
            
            del inputs, labels, label_pred, _

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
            inputs, labels = valdata
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            _, label_pred = model(inputs)
            loss = criterion_ce(label_pred, labels)

            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            pred_k1.append(torch.topk(label_pred, k=args.k1, dim=1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            val_loss += loss.detach().cpu().item()

            del inputs, labels, label_pred, _

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
            inputs, labels = testdata
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.long).to(device)

            inputs = inputs.unsqueeze(1)
            _, label_pred = model(inputs)
            loss = criterion_ce(label_pred, labels)

            pred_1.append(torch.max(label_pred, 1)[1].detach().cpu().numpy())
            pred_k1.append(torch.topk(label_pred, k=args.k1, dim=1)[1].detach().cpu().numpy())
            ground.append(labels.detach().cpu().numpy())

            test_loss += loss.detach().cpu().item()

            del inputs, labels, label_pred, _
        
        test_loss = test_loss/(batch_id+1)
        pred_1 = np.concatenate(pred_1)
        pred_k1 = np.concatenate(pred_k1)
        ground = np.concatenate(ground)
        acc_1 = accuracy_score(ground, pred_1)*100
        acc_k1 = top_k(ground, pred_k1, args.k1)*100
        bca_1 = balanced_accuracy_score(ground, pred_1)*100
    del model  
    return test_loss, acc_k1, acc_1, bca_1

def time_noise(data, labels, multiplier=1, noise_type='salt_and_pepper'):
    def add_gaussian_noise(data, weight=0.1):
        mean = np.mean(data)
        std = np.std(data)
        noise = np.random.normal(mean, std, data.shape)
        return data + weight*noise
    def add_salt_and_pepper_noise(data, salt_prob=0.01, pepper_prob=0.01):
        noisy_data = data.copy()
        num_salt = np.ceil(salt_prob * data.size)
        num_pepper = np.ceil(pepper_prob * data.size)
        coords = [np.random.randint(0, i, int(num_salt)) for i in data.shape]
        noisy_data[tuple(coords)] = np.max(data)
        coords = [np.random.randint(0, i, int(num_pepper)) for i in data.shape]
        noisy_data[tuple(coords)] = np.min(data)
        return noisy_data
    def add_poisson_noise(data):
        noise = np.random.poisson(size=data.shape)
        return data + noise
    def add_pink_noise(data, alpha=1.0):
        num_samples = data.shape[-1]
        num_columns = int(np.ceil(np.log2(num_samples)))
        shape = (data.shape[0], data.shape[1], 2 ** num_columns)
        noise = np.zeros(shape)
        b = np.random.randn(*shape)
        for i in range(1, num_columns):
            noise[:, :, ::2 ** i] += b[:, :, ::2 ** i]
        noise = noise[:, :, :num_samples]
        noise *= (np.arange(num_samples) + 1) ** (-alpha / 2.0)
        return data + noise

    if noise_type == 'gaussian': 
        add_noise_func = add_gaussian_noise
    elif noise_type == 'salt_and_pepper':  
        add_noise_func = add_salt_and_pepper_noise
    elif noise_type == 'poisson':  
        add_noise_func = add_poisson_noise
    elif noise_type == 'pink': 
        add_noise_func = add_pink_noise
    else:
        raise ValueError(f"Unknown noise: {noise_type}")

    aug_data = []
    aug_label = []
    for _ in range(multiplier):
        noisy_data = add_noise_func(data)
        aug_data.append(noisy_data)
        aug_label.append(labels)

    aug_data = np.concatenate(aug_data,axis=0)
    aug_label = np.concatenate(aug_label)

    augmented_data = np.concatenate((data, aug_data), axis=0)
    augmented_labels = np.concatenate((labels, aug_label), axis=0)

    return augmented_data, augmented_labels

def freq_noise(data, labels, multiplier=1, noise_type='salt_and_pepper'): 
    freq_data = np.fft.rfft(data, axis=-1)
    augmented_freq_data, augmented_labels = time_noise(freq_data, labels, multiplier, noise_type)
    augmented_data = np.fft.irfft(augmented_freq_data, axis=-1)
    return augmented_data, augmented_labels

def meg_dataset(subject_id='S01', channel='read', label_type='word', root='./DATA_MEG'):
    meg_file_path = root + f'/{subject_id}/processed/{channel}/meg/trial.npy'
    label_file_path = root + f'/{subject_id}/processed/{channel}/trans_label/trans_label.npz'
    print('loading:----{}----'.format(subject_id))
    data = np.load(meg_file_path)
    label = np.load(label_file_path)[label_type]
    return data, label

def run(seed, data, label):
    INCHANNELS = data.shape[1]
    skf = StratifiedKFold(n_splits=args.kflod, shuffle=True, random_state=seed)
    test_acck1_flod, test_acc1_flod, test_bca1_flod = [], [], []
    for flod_num, (train_val_index, test_index) in enumerate(skf.split(data, label), 1):
        meg_train_val, meg_read_test = data[train_val_index], data[test_index]
        label_train_val, label_read_test = label[train_val_index], label[test_index]
        meg_read_train, meg_read_val, label_read_train, label_read_val = train_test_split(meg_train_val, label_train_val, test_size=2/12, random_state=seed, shuffle=True, stratify=label_train_val)
        if args.shuffle == True:
            np.random.shuffle(label_read_train)
            np.random.shuffle(label_read_val)
        elif args.aug == 'time_noise':
            meg_read_train, label_read_train = time_noise(meg_read_train, label_read_train, multiplier=1, noise_type=args.noise_type)
        elif args.aug == 'freq_noise':
            meg_read_train, label_read_train = freq_noise(meg_read_train, label_read_train, multiplier=1, noise_type=args.noise_type)
       
        model_path = os.path.join(os.path.dirname(__file__), f'model_single/{args.model}/{args.label_type}/{args.subject}_{args.channel}_{seed}_{flod_num}_{args.noise_type}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        classes_num = len(np.unique(label_read_train))
        Model = EEGNet(n_classes=classes_num, Chans=INCHANNELS, Samples=args.samples, kernLenght=100, F1=4, D=2, F2=8, dropoutRate=0.25, norm_rate=0.5)
        dataloader_train = torch.utils.data.DataLoader(Dataset(meg_read_train, label_read_train), batch_size = args.bs, shuffle=True, drop_last=True)
        dataloader_val = torch.utils.data.DataLoader(Dataset(meg_read_val, label_read_val), batch_size = args.bs, shuffle=False, drop_last=True)
        dataloader_test = torch.utils.data.DataLoader(Dataset(meg_read_test, label_read_test), batch_size = args.bs, shuffle=False, drop_last=True)
        del meg_read_train, meg_read_val, meg_read_test, label_read_train, label_read_val, label_read_test
        train(Model, dataloader_train, dataloader_val, model_path)
        test_loss, test_acck1, test_acc1, test_bca1 = test(model_path, dataloader_test)
        test_acck1_flod.append(test_acck1)
        test_acc1_flod.append(test_acc1)
        test_bca1_flod.append(test_bca1)
    acck1_flod_avg = np.mean(test_acck1_flod)
    acc1_flod_avg = np.mean(test_acc1_flod)
    bca1_flod_avg = np.mean(test_bca1_flod)
    for i in range(args.kflod):
        print(f'flod_{i+1}: acc{args.k1}_flod: {test_acck1_flod[i]}      acc1_flod: {test_acc1_flod[i]}     bca1_flod: {test_bca1_flod[i]}')
    print(f'--------------------------------------------------------------------------------------')
    print(f'acc{args.k1}_flod_avg: {acck1_flod_avg}    acc1_flod_avg: {acc1_flod_avg}     bca1_flod_avg: {bca1_flod_avg}')
    print('\n')
    return acck1_flod_avg, acc1_flod_avg, bca1_flod_avg

def rep_experiment(seed_list=[2024]):
    acck1, acc1, bca1 = [], [], []
    data, label = meg_dataset(subject_id=args.subject, channel=args.channel, label_type=args.label_type, root=args.root_path)
    for seed in seed_list:
        print(f'*********************************************{seed}************************************************')
        set_seed(seed)
        test_acck1, test_acc1, test_bca1 = run(seed, data, label)
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
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
    parser.add_argument('--subject', type=str, default='S01', help='subject name')
    parser.add_argument('--root_path', type=str, default='../DATA', help='root_path')
    parser.add_argument('--channel', type=str, default='read', help='channel_path')
    parser.add_argument('--label_type', type=str, default='word', help='label_type')   
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--patience', type=int, default=50, help='early stop')
    parser.add_argument('--bs', type=int, default=48, help='batchsize')
    parser.add_argument('--samples', type=int, default=320, help='sample num')
    parser.add_argument('--model', type=str, default='eegnet', help='model')
    parser.add_argument('--kflod', type=int, default=5, help='kflod')
    parser.add_argument('--k1', type=int, default=5, help='topk1')
    parser.add_argument('--shuffle', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='shuffle?')
    parser.add_argument('--aug', type=str, default=None, help='aug?')  
    parser.add_argument('--noise_type', type=str, default='salt_and_pepper', help='noise_type') 
    args = parser.parse_args()

    args_dict = vars(args)
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    torch.cuda.set_device(args.gpu_id)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}+{torch.cuda.current_device()}')

    rep_experiment(seed_list=list(range(2024, 2024 + 20)))


