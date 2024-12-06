import torch
import torch.nn as nn
from torch.nn import init

class VGG16(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64):
        super(VGG16, self).__init__()
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker

        # Convolutional layers                            
        self.features = nn.Sequential(                                          # [1,  32, 32] input
            nn.Conv2d(self.nch_in, 64, kernel_size=3, padding=1), nn.ReLU(),    # [64, 32, 32] # bn 추가
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [64, 16, 16]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),            # [128, 16, 16]
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [128, 8, 8]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),           # [256, 8, 8]
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [256, 4, 4]
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),           # [512, 4, 4]
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),   
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # [512, 2, 2]
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),           # [512, 2, 2]
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                               # [512, 1, 1]
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(), nn.Dropout(),                      # [4096]
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),                     # [4096]
            nn.Linear(4096, self.nch_out)                                       # [10]
        )

    def forward(self, x):
        x = self.features(x)        # Convolutional layers
        x = x.view(x.size(0), -1)   # Flatten the tensor  # 남은 요소 수를 자동으로 계산해 그 자리에 채워넣는 것
        x = self.classifier(x)      # Fully connected layers

        return x



