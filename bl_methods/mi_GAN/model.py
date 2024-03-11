import torch
import torch.nn as nn
import pdb

def dowsample_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2),
        nn.Conv3d(out_channels, out_channels, kernel_size=4, padding=1, stride=2), # downsample
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2)
    )

def equal_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1), # downsample
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2)
    )

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, info_dim):
        super(UNet3D, self).__init__()

        # 下采样部分
        self.encoder1 = dowsample_conv_block(in_channels, 64)
        self.encoder2 = dowsample_conv_block(64, 128)
        self.encoder3 = dowsample_conv_block(128, 256)
        self.encoder4 = dowsample_conv_block(256, 512)
        self.encoder5 = dowsample_conv_block(512, 512)
        self.avg_pool = nn.AvgPool3d(kernel_size=2)

        
        # merge
        self.upsample_mi = nn.Linear(info_dim, 512)
        self.conv_merged = equal_conv_block(1024, 512)
        
        # 上采样部分
        
        self.decoder5 = equal_conv_block(1024, 256)
        self.decoder4 = equal_conv_block(512, 128)
        self.decoder3 = equal_conv_block(256, 64)
        self.decoder2 = equal_conv_block(128, 32)
        self.decoder1 = equal_conv_block(32, out_channels)

        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x, mi):
        # 下采样
        #pdb.set_trace()
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc5 = self.avg_pool(enc5)
        #pdb.set_trace()
        info = self.upsample_mi(mi).view(x.shape[0], 512, 1, 1, 1)
        merged = torch.cat([enc5, info], dim=1)
        upsampled_merged = self.upsample(merged)
        
        conved_merged = self.conv_merged(upsampled_merged)
        # 上采样

        dec4 = self.upsample(conved_merged)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec4 = self.decoder5(dec4)
        dec3 = self.upsample(dec4)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec3 = self.decoder4(dec3)

        dec2 = self.upsample(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec2 = self.decoder3(dec2)

        dec1 = self.upsample(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        dec1 = self.decoder2(dec1)
        
        dec1 = self.upsample(dec1)
        out = self.decoder1(dec1)
        
        
        return out

class discriminator3D(nn.Module):
    def __init__(self, in_channels):
        super(discriminator3D, self).__init__()
        
        self.conv1 = dowsample_conv_block(in_channels=in_channels, out_channels=32)
        self.conv2 = dowsample_conv_block(in_channels=32, out_channels=64)
        self.conv3 = dowsample_conv_block(in_channels=64, out_channels=128)
        self.conv4 = dowsample_conv_block(in_channels=128, out_channels=256)
        
        self.linear2 = nn.Linear(256*4*4*4, 1024)
        self.linear1 = nn.Linear(1024, 1)

    def forward(self, x):
        
        conved = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        #pdb.set_trace()
        out = self.linear1(self.linear2(nn.Flatten()(conved)))
        return out
        
    
    
    

if __name__ == '__main__':
    
# 模型实例化
    in_channels = 1  # 输入通道数
    out_channels = 1  # 输出通道数
    model = UNet3D(1, 1, info_dim=4)

    # 打印模型结构
    print(model)
    img = torch.ones((2, 1, 64, 64, 64))
    info = torch.ones(2, 4)
    output = model(x=img, mi=info)
    #pdb.set_trace()
    
    d = discriminator3D(in_channels=1)
    output = d(output)
    print(output)

