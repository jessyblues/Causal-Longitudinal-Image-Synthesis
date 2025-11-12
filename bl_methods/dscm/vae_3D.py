import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    """3D残差块，包含两次卷积和归一化"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 如果输入输出通道数不同或步长不为1，需要调整残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

class Conv3DEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(Conv3DEncoder, self).__init__()
        # 输入尺寸: 160×160×160
        self.initial_conv = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm3d(16)
        
        # 残差块序列 - 每次下采样将尺寸减半
        self.layer1 = ResidualBlock3D(16, 32, stride=2)  # 80×80×80
        self.layer2 = ResidualBlock3D(32, 64, stride=2)  # 40×40×40
        self.layer3 = ResidualBlock3D(64, 128, stride=2) # 20×20×20
        self.layer4 = ResidualBlock3D(128, 128, stride=2)# 10×10×10
        
        # 全局平均池化减少参数数量
        self.avg_pool = nn.AdaptiveAvgPool3d((10, 10, 10))
        
        # 计算特征维度
        self.feature_dim = 128 * 10 * 10 * 10
        
        # 输出潜在变量的均值和方差
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        
    def forward(self, x):
        # 初始卷积
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # 通过残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 池化并展平
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # 输出均值和方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Conv3DDecoder(nn.Module):
    def __init__(self, latent_dim, parent_vars_dim=3):
        super(Conv3DDecoder, self).__init__()
        # 计算需要复制parent variables的次数
        self.copy_count = max(1, latent_dim // parent_vars_dim)  # 至少复制1次
        
        # 输入包含latent vector和复制后的parent variables
        combined_dim = latent_dim + parent_vars_dim * self.copy_count
        
        # 映射到特征空间
        self.fc = nn.Linear(combined_dim, 128 * 10 * 10 * 10)
        
        # 残差块序列 - 每次上采样将尺寸加倍
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )  # 20×20×20
        self.res1 = ResidualBlock3D(128, 64, stride=1)
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )  # 40×40×40
        self.res2 = ResidualBlock3D(64, 32, stride=1)
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )  # 80×80×80
        self.res3 = ResidualBlock3D(32, 16, stride=1)
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )  # 160×160×160
        
        # 最终卷积层恢复到输入通道数
        self.final_conv = nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z, parents):
        # 复制parent variables以增加其影响
        parents_repeated = parents.repeat(1, self.copy_count)  # (batch_size, parent_vars_dim * copy_count)
        
        # 拼接latent vector和parent variables
        x = torch.cat([z, parents_repeated], dim=1)
        
        # 映射到特征空间并重塑
        x = x.float()
        x = self.fc(x)
        x = x.view(x.size(0), 128, 10, 10, 10)  # 重塑为3D特征
        
        # 通过上采样和残差块
        x = self.layer1(x)
        x = self.res1(x)
        
        x = self.layer2(x)
        x = self.res2(x)
        
        x = self.layer3(x)
        x = self.res3(x)
        
        x = self.layer4(x)
        
        # 最终输出
        x = torch.sigmoid(self.final_conv(x))  # 使用sigmoid确保输出在0-1之间
        return x

class VAE3D(nn.Module):
    def __init__(self, latent_dim=128, parent_vars_dim=3):
        super(VAE3D, self).__init__()
        self.encoder = Conv3DEncoder(latent_dim)
        self.decoder = Conv3DDecoder(latent_dim, parent_vars_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, parents):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, parents)
        return recon_x, mu, logvar
    
    def gen_cf(self, x, parents):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        cf_x = self.decoder(z, parents)
        return cf_x

# 改进的损失函数，可选择不同的重构损失
def vae_loss(recon_x, x, mu, logvar, recon_loss_type='bce'):
    # 根据输入选择不同的重构损失
    if recon_loss_type == 'bce':
        # 二元交叉熵损失，适用于0-1范围的输入
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 160*160*160), 
                                           x.view(-1, 160*160*160), 
                                           reduction='sum')
    elif recon_loss_type == 'mse':
        # 均方误差损失，适用于更广泛范围的输入
        recon_loss = F.mse_loss(recon_x.view(-1, 160*160*160),
                               x.view(-1, 160*160*160),
                               reduction='mean') * x.shape[0]  # 乘以批量大小以匹配总损失
    else:
        raise ValueError(f"不支持的重构损失类型: {recon_loss_type}")
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + 0.00005*kl_loss, recon_loss, kl_loss
    