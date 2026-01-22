import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import CBAM

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def activate_func(activate_method):
    if activate_method == 'relu':
        return nn.ReLU()
    elif activate_method == 'leaky':
        return nn.LeakyReLU(negative_slope=0.01)
    elif activate_method == 'swish':
        return Swish()
    else:
        raise ValueError(f"Unsupported activation method: {activate_method}")
    
class DoubleConv(nn.Module):
    """2 * (Norm -> ReLU -> Dropout -> conv(3,1,1))"""
    def __init__(self, in_channel, out_channel, dropout, activate_method, mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channel = out_channel

        # conv1 without dropout
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            activate_func(activate_method),
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(mid_channel),
            activate_func(activate_method),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False)
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))
    
class DownSample(nn.Module):
    """downsample using conv(3,2,1), 2 is scale ratio"""
    def __init__(self, channel, scale_ratio):
        super().__init__()
        self.down = nn.Conv2d(channel, channel, 3, scale_ratio, 1)

    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    """upsample using [upsample + conv(1,1,0)]"""
    def __init__(self, channel, scale_ratio):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_ratio, mode="nearest"),
            nn.Conv2d(channel, channel//2, 1, 1),
            #nn.Conv2d(channel, channel//2, 3, 1, 1)
        )

    def forward(self, x):
        return self.up(x)

class net(nn.Module):
    def __init__(self, 
                 in_channel=3, 
                 out_channel=1, 
                 channels=[16, 32, 64, 128],
                 dropout=0,
                 activate_method='relu',
                 use_checkpoint=False
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.channels = channels
        self.dropout = dropout
        self.activate_method = activate_method

        self.start = nn.Sequential(
            DoubleConv(in_channel, channels[0], self.dropout, self.activate_method),
            CBAM(gate_channels=channels[0], reduction_ratio=2, pool_types=['max', 'lse']), # CBAM attention module
        )    

        self.down1 = nn.Sequential(                         #------3x: 0.5km -> 1.5km                          
            DownSample(channels[0], 3), #32->32
            DoubleConv(channels[0], channels[1], self.dropout, self.activate_method) #32->64
        ) 
        self.down2 = nn.Sequential(                         #---2x: 1.5km -> 3km
            DownSample(channels[1], 3), #64->64
            DoubleConv(channels[1], channels[2], self.dropout, self.activate_method) #64->128
        )
        self.down3 = nn.Sequential(                         #2x: 3km -> 6km
            DownSample(channels[2], 3), #128->128
            DoubleConv(channels[2], channels[3], self.dropout, self.activate_method) #128->256
        )
        
        self.up3 = nn.Sequential(                           #2x: 6km -> 3km
            UpSample(channels[3], 3), #256->256//2=128
            DoubleConv(channels[3], channels[2], self.dropout, self.activate_method) #128*2=256->128
        )
        self.up2 = nn.Sequential(                           #---2x: 3km -> 1.5km
            UpSample(channels[2], 3), #128->128//2=64
            DoubleConv(channels[2], channels[1], self.dropout, self.activate_method) #64*2=128->64
        )
        self.up1 = nn.Sequential(                           #------3x: 1.5km -> 0.5km
            UpSample(channels[1], 3), #64->64//2=32
            DoubleConv(channels[1], channels[0], self.dropout, self.activate_method) #32*2=64->32
        )

        # out
        self.end = nn.Conv2d(channels[0],out_channel,1,1)

    def forward(self, x_i): 
        x0_o = self.start[0](x_i)
        x0 = self.start[1](x0_o)    
        x1 = self.down1(x0)     
        x2 = self.down2(x1)     
        x3 = self.down3(x2)     
        
        out = self.up3[0](x3)   
        out = torch.cat([out, x2], dim=1)
        out = self.up3[1](out)  
        
        out = self.up2[0](out)  
        out = torch.cat([out, x1], dim=1)
        out = self.up2[1](out)  
        
        out = self.up1[0](out)  
        out = torch.cat([out, x0_o], dim=1)
        out = self.up1[1](out)  

        return self.end(out)
    
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = net()
    model = model.to(device)
    
    # train demo
    def train_demo(model, batch_size=5):
        model.train()  # train
        
        # torch tensors for train demo
        x_i = torch.randn(batch_size, 3, 486, 486, requires_grad=True, device=device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        y = model(x_i)
        # loss function
        loss = y.mean()       
        # backward
        loss.backward()
        optimizer.step()
        
        print("=== Training Mode ===")
        print(f"Shape of x_i={x_i.shape}, y={y.shape}")
        print(f"Loss: {loss.item()}")
        
    train_demo(model)

if __name__ == "__main__":
    test()


