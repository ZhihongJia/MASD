import os
import re
import mne
import numpy as np
import scipy.io

from scipy import signal as scipysignal
from sklearn.preprocessing import robust_scale
from label import Label_Ref
import argparse

class PreProcessor():
    def __init__(self, block_num=15, trial_num=48, trial_len=1.6, fs=1000) -> None:
        self.block_num = block_num
        self.trial_num = trial_num
        self.trial_len = trial_len
        self.fs = fs


    def readLabel(self, label_path, label_save_path, save):
        log_file = os.path.join(label_path, 'exp_info.log')
        pattern = r"\{'word': '(\S+)', 'label': (\d+), 'initial': (None|'\S+'), 'final': '(\S+)', 'tone': (\d+)\}"
        self.wordlist = []

        with open(log_file, "r") as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    word = match.group(1)
                    self.wordlist.append(word)

        self.wordlist = np.array(self.wordlist)

        label_ref = Label_Ref().label
        label_key = self.wordlist
        label_word_class = np.array([label_ref[word]['word'] for word in self.wordlist])
        label_initial_class = np.array([label_ref[word]['initial_class'] for word in self.wordlist])
        label_initial_class_8 = np.array([label_ref[word]['initial_class_8'] for word in self.wordlist])
        label_final_class = np.array([label_ref[word]['final_class'] for word in self.wordlist])
        label_tone_class = np.array([label_ref[word]['tone_class'] for word in self.wordlist])
        label_initial = np.array([label_ref[word]['initial'] for word in self.wordlist])
        label_final = np.array([label_ref[word]['final'] for word in self.wordlist])

        if save == True:
            if not os.path.exists(label_save_path):
                os.makedirs(label_save_path)
            trans_label_file = os.path.join(label_save_path, 'trans_label.npz')
            np.savez(trans_label_file, key=label_key,
                                       word=label_word_class, 
                                       initial_class=label_initial_class, 
                                       initial_class_8=label_initial_class_8, 
                                       final_class=label_final_class, 
                                       tone_class=label_tone_class, 
                                       initial=label_initial, 
                                       final=label_final)


    def readMEG(self, meg_path, trial_start_threshold, pick_channels, channel_save_path, save):
        pick_channels = np.array(pick_channels)

        channel_path = os.path.join(meg_path, 'channel.mat')
        channel_info = scipy.io.loadmat(channel_path)
        if 'channel' in channel_info.keys():
            channel = channel_info['channel'][0,0]
        elif 'Channel' in channel_info.keys():
            channel = channel_info['Channel'][0,0]
        else:
            raise 'Error'
        channel_data = channel['Channel'][0]

        self.channel_Name = np.hstack(channel_data['Name'])
        self.channel_Type = np.hstack(channel_data['Type'])

        if save == True:
            if not os.path.exists(channel_save_path):
                os.makedirs(channel_save_path)
            np.save(os.path.join(channel_save_path, 'channel_Name.npy'), self.channel_Name)


        self.MEG_index = np.where(np.isin(self.channel_Name, pick_channels))[0]
        STIM_index = np.where(self.channel_Type=='STIM')

        data_path = os.path.join(meg_path, 'raw.mat')
        data = scipy.io.loadmat(data_path)
        if 'raw' in data.keys():
            raw = data['raw'][0,0]
        elif 'Raw' in data.keys():
            raw = data['Raw'][0,0]
        elif 'rawData' in data.keys():
            raw = data['rawData'][0,0]
        else:
            raise ValueError('Error: not found in data')
        if 'F' in raw.dtype.names:
            self.MEG = np.array(raw['F'][self.MEG_index])
            self.STIM = np.array(raw['F'][STIM_index])
        elif 'trial' in raw.dtype.names:
            self.MEG = np.array(raw['trial'][0,0][self.MEG_index])
            self.STIM = np.array(raw['trial'][0,0][STIM_index])
        else:
            raise ValueError('Error: not found in raw')

        STIM_left = self.STIM[..., :-1]
        STIM_right = self.STIM[...,1:]
        updown_edge = STIM_right - STIM_left
        self.trial_start = np.where((updown_edge[0] > trial_start_threshold))[0]
        trial_start_left = self.trial_start[..., :-1]
        trial_start_right = self.trial_start[..., 1:]
        trial_start_interval = trial_start_right - trial_start_left
        pad_start_index = np.where(((trial_start_interval > 1.5*self.trial_len*self.fs) & 
                                    (trial_start_interval < 2.5*self.trial_len*self.fs)))[0]
        temp_pad_STIM = (self.trial_start[pad_start_index] + self.trial_start[pad_start_index+1]) / 2
        self.trial_start = np.insert(self.trial_start, pad_start_index+1, temp_pad_STIM)
        trial_start_left = self.trial_start[..., :-1]
        trial_start_right = self.trial_start[..., 1:]
        trial_start_interval = trial_start_right - trial_start_left
        del_start_index = np.where((trial_start_interval < 0.2*self.trial_len*self.fs))[0]
        self.trial_start = np.delete(self.trial_start, del_start_index+1)


    def dataDetrend(self):
        self.MEG = scipysignal.detrend(self.MEG, axis=1)


    def reReference(self):
        Ref_data = np.mean(self.MEG, axis=0)
        self.MEG = self.MEG - Ref_data


    def notch_filter(self, f_notch):
        b1, a1 = scipysignal.iirnotch(w0=f_notch, Q=30, fs=self.fs)
        self.MEG = scipysignal.filtfilt(b1, a1, self.MEG)
        b2, a2 = scipysignal.iirnotch(w0=f_notch*2, Q=30, fs=self.fs)
        self.MEG = scipysignal.filtfilt(b2, a2, self.MEG)
        b3, a3 = scipysignal.iirnotch(w0=f_notch*3, Q=30, fs=self.fs)
        self.MEG = scipysignal.filtfilt(b3, a3, self.MEG)


    def bandpass_filter(self, f_low=70, f_high=150):
        b_band, a_band = scipysignal.butter(N=4, Wn=(f_low, f_high), btype='bandpass', fs=self.fs)
        self.MEG = scipysignal.filtfilt(b_band, a_band, self.MEG)

    def clamping(self):
        self.MEG = robust_scale(self.MEG, axis=1)
        self.MEG = np.clip(self.MEG, a_min=-15, a_max=15)

    def hilbertTrans(self):
        self.envelope = np.abs(scipysignal.hilbert(self.MEG))
    
    def dataResample(self, new_sample_rate):
        down_factor = int(self.fs / new_sample_rate)
        self.envelope = scipysignal.resample(self.envelope, self.envelope.shape[1] // down_factor, axis=1)

    def dataSplit(self, meg_save_path, new_sample_rate, save=False):
        self.envelope_trial = []
        self.trial_start = (self.trial_start / self.fs) * new_sample_rate

        for i in range(len(self.trial_start)):
            temp_trial = self.envelope[..., int(self.trial_start[i]):int(self.trial_start[i])+int(new_sample_rate*self.trial_len)]
            self.envelope_trial.append(temp_trial)
        self.envelope_trial = np.array(self.envelope_trial)

        if save == True:
            if not os.path.exists(meg_save_path):
                os.makedirs(meg_save_path)
            np.save(os.path.join(meg_save_path, 'trial.npy'), self.envelope_trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel_path', type=str, default='read', help='channel name')    # read    read_cross
    parser.add_argument('--root_path', type=str, default='../DATA/', help='root_path')
    parser.add_argument('--root_save_path', type=str, default='../DATA/', help='root_save_path')
    args = parser.parse_args()

    args_dict = vars(args)
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")

    subject_list = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']
    
    if args.channel_path == 'read':
        meg_channels = ['SM1z', 'SM4z', 'SM5z', 'SM7z', 'SL2z', 'SL3z', 'AL1z',
                        'SL5z', 'SR2z', 'SR3z', 'AR1z', 'SR5z', 'MM2z', 'MM4z', 'MM5z', 'MM6z', 'MM7z',
                        'MM8z', 'MM9z', 'ML2z', 'AL2z', 'ML4z', 'ML5z', 'ML6z', 'ML7z', 'ML8z', 'MR2z',
                        'AR2z', 'MR4z', 'MR5z', 'MR6z', 'MR7z', 'MR8z', 'IM1z', 'AR6z', 'AL6z', 'IM5z',
                        'IM6z', 'IM7z', 'IM8z', 'IL1z', 'IL3z', 'IL4z', 'IL5z', 'IL6z', 'IL7z', 'IR1z',
                        'IR3z', 'IR4z', 'IR5z', 'IR6z', 'IR7z', 'AM1z', 'AM3z', 'AM5z', 'AL3z', 'AL5z',
                        'AM4z', 'AR3z', 'AR5z', 'AM6z']
    elif args.channel_path == 'read_cross':   
        meg_channels = ['SM1z', 'SM4z', 'SM5z', 'SM7z', 'SL2z', 'SL3z', 'AL1z',
                        'SL5z', 'SR2z', 'SR3z', 'AR1z', 'SR5z', 'MM2z', 'MM4z', 'MM5z', 'MM6z', 'MM7z',
                        'MM8z', 'MM9z', 'ML2z', 'AL2z', 'ML4z', 'ML5z', 'ML6z', 'ML7z', 'ML8z', 'MR2z',
                        'MR4z', 'MR5z', 'MR6z', 'MR7z', 'MR8z', 'IM1z', 'AR6z', 'IM5z',
                        'IM6z', 'IM7z', 'IM8z', 'IL3z', 'IL4z', 'IL5z', 'IL6z', 'IL7z', 'IR1z',
                        'IR3z', 'IR4z', 'IR5z', 'IR6z', 'IR7z', 'AM1z', 'AM5z', 'AL3z', 'AL5z',
                        'AM4z', 'AR3z', 'AR5z', 'AM6z']
    
    fs = 1000
    env_sample_rate = 200
    trial_start_threshold = 0.6

    label_save = True
    meg_save = True
    channel_save = True
    for subject in subject_list:
        print(f'***************************************{subject}*****************************************************')
        label_path = args.root_path + f'{subject}/raw/psydat'
        meg_path = args.root_path + f'{subject}/raw/data'

        label_save_path = args.root_save_path + f'{subject}/processed/{args.channel_path}/trans_label'
        meg_save_path = args.root_save_path + f'{subject}/processed/{args.channel_path}/meg'
        channel_save_path = args.root_save_path + f'{subject}/processed/{args.channel_path}/channel'
        
        preprocessor = PreProcessor(block_num=15, trial_num=48, trial_len=1.6)
        preprocessor.readLabel(label_path, label_save_path, save=label_save)

        preprocessor.readMEG(meg_path, trial_start_threshold, pick_channels=meg_channels, channel_save_path=channel_save_path, save=channel_save)
        preprocessor.dataDetrend()
        preprocessor.reReference()
        preprocessor.bandpass_filter(f_low=3, f_high=200)
        preprocessor.bandpass_filter(f_low=70, f_high=170)
        preprocessor.notch_filter(f_notch=50)
        preprocessor.clamping()
        preprocessor.hilbertTrans()
        preprocessor.dataResample(env_sample_rate)
        preprocessor.dataSplit(meg_save_path, env_sample_rate, save=meg_save)