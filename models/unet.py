import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from torch.utils.checkpoint import checkpoint
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
        if self.use_checkpoint:
            x0 = checkpoint(self.start, x_i, use_reentrant=False)
            x1 = checkpoint(self.down1, x0, use_reentrant=False)
            x2 = checkpoint(self.down2, x1, use_reentrant=False)
            x3 = checkpoint(self.down3, x2, use_reentrant=False)
            
            out = checkpoint(self.up3[0], x3, use_reentrant=False)
            out = torch.cat([out, x2], dim=1)
            out = checkpoint(self.up3[1], out, use_reentrant=False)
            
            out = checkpoint(self.up2[0], out, use_reentrant=False)
            out = torch.cat([out, x1], dim=1)
            out = checkpoint(self.up2[1], out, use_reentrant=False)
            
            out = checkpoint(self.up1[0], out, use_reentrant=False)
            out = torch.cat([out, x0], dim=1)
            out = checkpoint(self.up1[1], out, use_reentrant=False)
            
            return checkpoint(self.end, out, use_reentrant=False)
        else:
            x0 = self.start(x_i)    
            x1 = self.down1(x0)     
            x2 = self.down2(x1)     
            x3 = self.down3(x2)     
            
            out = self.up3[0](x3)   
            del x3  # 释放x3
            out = torch.cat([out, x2], dim=1)
            del x2  # 释放x2
            out = self.up3[1](out)  
            
            out = self.up2[0](out)  
            out = torch.cat([out, x1], dim=1)
            del x1  # 释放x1
            out = self.up2[1](out)  
            
            out = self.up1[0](out)  
            out = torch.cat([out, x0], dim=1)
            del x0  # 释放x0
            out = self.up1[1](out)  

            return self.end(out)
    
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net(use_checkpoint=True).to(device)
    
    # 训练模式演示
    def train_demo(model, batch_size=64):
        model.train()  # 确保模型在训练模式
        
        # 创建需要梯度的输入张量
        x_i = torch.randn(batch_size, 3, 486, 486, requires_grad=True, device=device)
        
        # 使用自动混合精度
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with torch.amp.autocast(device_type=device.type):
            y = model(x_i)
            # 添加一个虚拟的损失函数来演示梯度计算
            loss = y.mean()
            
        # 测试反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("=== Training Mode ===")
        print(f"Shape of x_i={x_i.shape}, y={y.shape}")
        print(f"Loss: {loss.item()}")
        
    # 验证模式演示
    def eval_demo(model, batch_size=16):
        model.eval()  # 切换到评估模式
        
        with torch.no_grad():  # 在验证时不计算梯度
            # 创建输入张量（验证时不需要requires_grad）
            x_i = torch.randn(batch_size, 3, 486, 486, device=device)

            # 在验证时仍然使用自动混合精度以节省显存
            with torch.amp.autocast(device_type=device.type):
                y = model(x_i)
                # 计算验证指标（这里仅作示例）
                val_loss = F.mse_loss(y, torch.zeros_like(y))
                
            print("\n=== Evaluation Mode ===")
            print(f"Shape of x_i={x_i.shape}, y={y.shape}")
            print(f"Validation Loss: {val_loss.item()}")
            
            # 可以添加其他验证指标
            print(f"Output mean: {y.mean().item()}")
            print(f"Output std: {y.std().item()}")
    
    # 运行训练和验证演示
    train_demo(model)
    eval_demo(model)

if __name__ == "__main__":
    test()

