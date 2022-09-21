import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self,in_channel, hidden_channels, kernal_size, T):
        super(block, self).__init__() 
        self.covn1d_1 = nn.Conv1d(in_channels = in_channel, out_channels = hidden_channels, kernel_size = kernal_size,padding = int((kernal_size-1)/2))
        self.covn1d_2 = nn.Conv1d(in_channels = hidden_channels, out_channels = hidden_channels, kernel_size = kernal_size,padding = int((kernal_size-1)/2))
        self.covn1d_3 = nn.Conv1d(in_channels = hidden_channels, out_channels = hidden_channels, kernel_size = kernal_size,padding = int((kernal_size-1)/2))
        self.covn1d_4 = nn.Conv1d(in_channels = hidden_channels, out_channels = hidden_channels, kernel_size = kernal_size,padding = int((kernal_size-1)/2))
        self.fc_1 = nn.Conv1d(in_channels = hidden_channels, out_channels = in_channel, kernel_size = kernal_size, padding = int((kernal_size-1)/2))
        self.fc_2 = nn.Conv1d(in_channels = hidden_channels, out_channels = in_channel, kernel_size = kernal_size, padding = int((kernal_size-1)/2))
        # self.fc_1 = nn.Linear( hidden_channels * T, in_channel * T )
        # self.fc_2 = nn.Linear( hidden_channels * T, in_channel * T )
        
    def forward(self, x):
        b,c,t = x.shape
        x = torch.relu(self.covn1d_1(x))
        x = torch.relu(self.covn1d_2(x) + x)    
        x = torch.relu(self.covn1d_3(x) + x)
        x = torch.relu(self.covn1d_4(x) + x)
        # print(x.shape) # b c t
        # signal =  self.fc_1( x.view(x.size(0),-1) ).reshape(b,c,t)
        # noise = self.fc_2( x.view(x.size(0),-1) ).reshape(b,c,t)
        signal = self.fc_1( x )
        noise = self.fc_1( x ) 
        return signal, noise

class MMNN_4(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernal_size, T,DEVICE= torch.device('cuda:0')):
        super(MMNN_4, self).__init__()
        self.block_1 = block(in_channels, hidden_channels, kernal_size, T)
        self.block_2 = block(in_channels, hidden_channels, kernal_size, T)
        self.block_3 = block(in_channels, hidden_channels, kernal_size, T)
        self.block_4 = block(in_channels, hidden_channels, kernal_size, T)
        self.to(DEVICE)
    def forward(self, x):
        signal_1, noise_1 = self.block_1( x )
        signal_2, noise_2 = self.block_2( x - noise_1 )
        signal_3, noise_3 = self.block_3( x - noise_2 )
        signal_4, noise_4 = self.block_4( x - noise_3 )
        
        output = signal_1 + signal_2 + signal_3 + signal_4
        
        return output.squeeze()

def make_model(args, in_channels,DEVICE):
    model = MMNN_4(in_channels=in_channels, 
        hidden_channels = 32,
        kernal_size = 25,
        T = 3000, 
        DEVICE = DEVICE
        )
    # param init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model 

# x = torch.ones(1,1,512)
# net = MMNN_4(32,25,512)
# print('MMNN-4 for OA removal',net(x).size())
# x = torch.ones(1,1,1024)
# net = MMNN_4(32,103,1024)
# print('MMNN-4 for MA removal',net(x).size())