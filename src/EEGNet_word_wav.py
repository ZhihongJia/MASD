import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

class EEGNet(nn.Module):

    def __init__(self,
                 Chans: int,
                 Samples: int,
                 kernLenght: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet, self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),  
            nn.BatchNorm2d(num_features=self.F1),
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False), 
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),   
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),  
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),  
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),   
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        output = self.block1(x) 
        features = self.block2(output)   
        features = features.reshape(output.size(0), -1)
        return features

class EEGNet_word_wav(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,  # 32
                 kernLenght: int,
                 F1_wav: int,
                 D_wav: int,
                 F2_wav: int,
                 F1_word: int,
                 D_word: int,
                 F2_word: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet_word_wav, self).__init__()

        self.eegnet_wav = EEGNet(Chans=Chans, Samples=Samples, kernLenght=kernLenght, F1=F1_wav, D=D_wav, F2=F2_wav, dropoutRate=dropoutRate, norm_rate=norm_rate)
        self.eegnet_word = EEGNet(Chans=Chans, Samples=Samples, kernLenght=kernLenght, F1=F1_word, D=D_word, F2=F2_word, dropoutRate=dropoutRate, norm_rate=norm_rate)
        self.clf = nn.Linear(in_features=(F2_wav+F2_word) * (Samples//(4*8)), out_features=n_classes, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        feature_wav = self.eegnet_wav(x)    
        feature_word = self.eegnet_word(x)  
        feature = torch.cat([feature_wav, feature_word], dim=-1)    
        label_pred = self.clf(feature)  
        return feature_wav, feature_word, label_pred