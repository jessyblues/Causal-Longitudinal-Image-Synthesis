import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
sys.path.append('/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dscm')
from vae_3D import VAE3D, vae_loss
import os
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import random
import pandas as pd
from typing import Dict
from torch import Tensor
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    RandSpatialCrop,
    CenterSpatialCrop,
    Resize,
    SpatialPad,
)

# 设置日志，只有主进程会输出日志
def setup_logging(rank, log_file="vae_ddp_training.log"):
    if rank == 0:
        # 为FileHandler指定mode='w'实现覆盖重写
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, stream_handler]
        )
    else:
        # 非主进程禁用日志
        logging.basicConfig(level=logging.CRITICAL + 1)
# 数据集类
class ADNI_3d_Dataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        subject_csv_file: str,
        volume_csv_file: str,
        transform,
        min_and_max:dict = None,
        concat_pa=True,
        parents = ['VIT', 'GreyMatter', 'Ventricles']
        
    ):
        super().__init__()
        self.root = image_dir
        self.transform = transform
        self.concat_pa = concat_pa  # return concatenated parents
        self.parents = parents

        print(f"\nLoading subject csv data: {subject_csv_file}")
        self.subject_df = pd.read_csv(subject_csv_file)
        self.subjects = set(self.subject_df['Subject'].tolist())
        
        self.volume_df = pd.read_csv(volume_csv_file)
        self.data_list = []
        
        for subject in os.listdir(image_dir):
            if subject not in self.subjects:
                continue
            else:
                dates = os.listdir(os.path.join(image_dir, subject))
                dates = sorted(dates)
                for date in dates:
                    image_file = os.listdir(os.path.join(image_dir, subject, date))[0]
                    image_path = os.path.join(image_dir, subject, date, image_file)
                    line = self.volume_df[(self.volume_df['Subject'] == subject) & (self.volume_df['Acq Date'] == date)]
                    if len(line) == 0:
                        continue

                    VIT = line['WholeBrain'].values[0]
                    GreyMatter = line['GreyMatter'].values[0]
                    Ventricles = line['SegVentricles'].values[0]
                    
                    VIT = (VIT - min_and_max['brain_volume'][0]) / (min_and_max['brain_volume'][1] - min_and_max['brain_volume'][0]) * 2 - 1
                    GreyMatter = (GreyMatter - min_and_max['grey_matter'][0]) / (min_and_max['grey_matter'][1] - min_and_max['grey_matter'][0]) * 2 - 1
                    Ventricles = (Ventricles - min_and_max['ventricle_volume'][0]) / (min_and_max['ventricle_volume'][1] - min_and_max['ventricle_volume'][0]) * 2 - 1
                    
                    self.data_list.append({
                        'image_path': image_path,
                        'Subject': subject,
                        'Acq Date': date,
                        'VIT': VIT,
                        'GreyMatter': GreyMatter,
                        'Ventricles': Ventricles})

        
        self.df = pd.DataFrame(self.data_list)
        print(f"#samples: {len(self.df)}")
        self.return_x = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.data_list[idx]
        x = sitk.ReadImage(sample['image_path'])
        x = sitk.GetArrayFromImage(x)  # z, y, x
        x = torch.tensor(x).unsqueeze(0).float()  # 1,
        
        x = x - x.min()
        x = x / x.max()  # normalize to [0,1]
        
        if self.transform is not None:
            sample["x"] = self.transform(x)


        if self.concat_pa:
            #ipdb.set_trace()
            sample["pa"] = torch.cat(
                [torch.tensor([sample[k]]) for k in self.parents], dim=0
            )
        #print(sample["pa"].shape)
        #quit()
        return sample
    

def adni_3d(args) -> Dict[str, ADNI_3d_Dataset]:
    # Load data
    if not args.data_dir:
        args.data_dir = "/home/yujiali/dataset/ADNI/T1/aligned_brain_MNI/"
    image_dir = args.data_dir
    
    train_subject_csv_file = '/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dataset_config/train_subjects.csv'
    test_subject_csv_file = '/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dataset_config/test_subjects.csv'
    valid_subject_csv_file = '/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dataset_config/test_subjects.csv'
    
    volume_csv_file = '/home/yujiali/dataset/ADNI/T1/excel/all_seg.csv'
    df = pd.read_csv(volume_csv_file)
    min_and_max = {
        'brain_volume': [df['WholeBrain'].min(), df['WholeBrain'].max()],
        'grey_matter': [df['GreyMatter'].min(), df['GreyMatter'].max()],
        'ventricle_volume': [df['SegVentricles'].min(), df['SegVentricles'].max()]
    }


    augmentation = {
        "train": Compose(
            [
                SpatialPad(spatial_size=(160, 192, 160), mode='symmetric'),
                RandSpatialCrop((160, 192, 160), random_size=False),
                # CenterSpatialCrop((args.input_res, args.input_res, args.input_res)),
                Resize((args.input_res, args.input_res, args.input_res)),
                
            ]
        ),
        "eval": Compose(
            [
                SpatialPad(spatial_size=(160, 192, 160), mode='symmetric'),
                CenterSpatialCrop((160, 192, 160)),
                # Resize((args.input_res, args.input_res, args.input_res)),
                Resize((args.input_res, args.input_res, args.input_res)),
            ]
        ),
    }

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = ADNI_3d_Dataset(
            image_dir=image_dir,
            subject_csv_file=(train_subject_csv_file if split == "train" else (valid_subject_csv_file if split == "valid" else test_subject_csv_file)),
            volume_csv_file=volume_csv_file,
            transform=augmentation[("eval" if split != "train" else split)],
            concat_pa=True,
            min_and_max=min_and_max
        )
    
    return datasets



def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()

    
def save_volume_to_nii(volume, filepath):
    """将3D体积数据保存为NIfTI文件"""
    volume = volume.squeeze().cpu().numpy()  # 移动到CPU并转换为numpy数组
    nii_image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(nii_image, filepath)

def train(rank, args):
    """单个进程的训练函数"""
    # 设置日志
    setup_logging(rank, args.log_file)
    
    # 初始化分布式进程组
    logging.info(f"进程 {rank} 初始化中...")
    dist.init_process_group(
        backend='nccl',  # 使用NCCL后端，适合GPU
        init_method=args.init_method,
        world_size=args.world_size,
        rank=rank
    )
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 创建TensorBoard写入器（只在主进程）
    writer = None
    if rank == 0:
        # 创建日志目录
        log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"TensorBoard日志将保存到: {log_dir}")
        
        os.makedirs(args.visualization_dir, exist_ok=True)
        logging.info(f"可视化结果将保存到: {args.visualization_dir}")
    
    # 创建模型并移动到设备
    model = VAE3D(latent_dim=args.latent_dim).to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 创建数据集
    datasets = adni_3d(args)
    train_dataset = datasets['train']
    val_dataset = datasets['valid']
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # 数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,  # 使用分布式采样器
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 创建保存模型的目录（只在主进程执行）
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载检查点
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and rank == 0:
        checkpoint_path = os.path.join(args.save_dir, 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            logging.info(f"从检查点 {checkpoint_path} 恢复")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            
            # 广播检查点信息到所有进程
            start_epoch_tensor = torch.tensor(start_epoch, device=device)
            best_loss_tensor = torch.tensor(best_loss, device=device)
        else:
            start_epoch_tensor = torch.tensor(0, device=device)
            best_loss_tensor = torch.tensor(float('inf'), device=device)
    else:
        start_epoch_tensor = torch.tensor(0, device=device)
        best_loss_tensor = torch.tensor(float('inf'), device=device)
    
        # 包装DDP
    model = DDP(model, device_ids=[rank])
    # 确保所有进程同步起始epoch和最佳损失
    dist.broadcast(start_epoch_tensor, src=0)
    dist.broadcast(best_loss_tensor, src=0)
    start_epoch = start_epoch_tensor.item()
    best_loss = best_loss_tensor.item()
    
    # 选择固定的样本用于可视化重构效果
    visualization_samples = None
    visualization_parents = None
    if rank == 0:
        # 随机选择几个样本
        vis_indices = random.sample(range(len(val_dataset)), args.vis_samples)
        visualization_samples = torch.stack([val_dataset[i]['x'] for i in vis_indices]).to(device)
        visualization_parents = torch.stack([val_dataset[i]['pa'] for i in vis_indices]).to(device)
        logging.info(f"将使用 {args.vis_samples} 个样本进行重构可视化")
    
    # 训练循环
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        # 设置采样器的epoch，确保不同epoch的shuffle不同
        train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        # 只在主进程显示进度条
        if rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            progress_bar = train_dataloader
        
        for batch_idx, data_batch in enumerate(progress_bar):
            
            
            
            data = data_batch['x'].to(device, non_blocking=True)
            parents = data_batch['pa'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            recon_batch, mu, logvar = model(data, parents)
            
            # 计算损失
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, args.recon_loss)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # 更新主进程的进度条
            if rank == 0:
                progress_bar.set_postfix(
                    loss=loss.item()/data.shape[0],
                    recon_loss=recon_loss.item()/data.shape[0],
                    kl_loss=kl_loss.item()/data.shape[0]
                )
                
                # 记录每一步的损失（可选，会产生大量日志）
                if args.log_steps and batch_idx % 10 == 0:
                    global_step = epoch * len(train_dataloader) + batch_idx
                    writer.add_scalar('step/loss', loss.item()/data.shape[0], global_step)
                    writer.add_scalar('step/recon_loss', recon_loss.item()/data.shape[0], global_step)
                    writer.add_scalar('step/kl_loss', kl_loss.item()/data.shape[0], global_step)
        
        # 计算平均损失
        avg_loss = total_loss / len(train_dataset)
        avg_recon_loss = total_recon_loss / len(train_dataset)
        avg_kl_loss = total_kl_loss / len(train_dataset)
        
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            # 在验证集上评估
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_data = val_batch['x'].to(device, non_blocking=True)
                    val_parents = val_batch['pa'].to(device, non_blocking=True)
                    
                    recon_val, mu_val, logvar_val = model(val_data, val_parents)
                    v_loss, v_recon_loss, v_kl_loss = vae_loss(recon_val, val_data, mu_val, logvar_val, args.recon_loss)
                    
                    val_loss += v_loss.item()
                    val_recon_loss += v_recon_loss.item()
                    val_kl_loss += v_kl_loss.item()
            
            avg_val_loss = val_loss / len(val_dataset)
            avg_val_recon_loss = val_recon_loss / len(val_dataset)
            avg_val_kl_loss = val_kl_loss / len(val_dataset)
            
            if rank == 0:
                logging.info(
                    f"验证集 - 平均总损失: {avg_val_loss:.6f}, "
                    f"平均重构损失: {avg_val_recon_loss:.6f}, "
                    f"平均KL损失: {avg_val_kl_loss:.6f}"
                )
            model.train()
            
            if rank == 0:
                writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
                writer.add_scalar('epoch/val_recon_loss', avg_val_recon_loss, epoch)
                writer.add_scalar('epoch/val_kl_loss', avg_val_kl_loss, epoch)
        
        # 主进程处理日志和可视化
        if rank == 0:
            # 记录 epoch 级别的损失
            writer.add_scalar('epoch/loss', avg_loss, epoch)
            writer.add_scalar('epoch/recon_loss', avg_recon_loss, epoch)
            writer.add_scalar('epoch/kl_loss', avg_kl_loss, epoch)
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('epoch/learning_rate', current_lr, epoch)
            
            logging.info(
                f'Epoch {epoch+1}, 平均总损失: {avg_loss:.6f}, '
                f'平均重构损失: {avg_recon_loss:.6f}, '
                f'平均KL损失: {avg_kl_loss:.6f}, '
                f'学习率: {current_lr:.6f}'
            )
            
            # 学习率调度
            scheduler.step(avg_loss)
            
            # 定期可视化重构效果
            if args.eval_every > 0 and (epoch + 1) % args.vis_interval == 0 and visualization_samples is not None:
                model.eval()
                with torch.no_grad():
                    recon_samples, _, _ = model(visualization_samples, visualization_parents)
                
                # 记录原始数据和重构数据的切片
                for i in range(min(args.vis_samples, 5)):  # 最多可视化5个样本

                    save_volume_to_nii(
                        visualization_samples[i], 
                        os.path.join(args.visualization_dir, f'original_sample_{i}_epoch_{epoch+1}.nii.gz')
                    )              
                    # 新增：保存重构图像
                    save_volume_to_nii(
                        recon_samples[i], 
                        os.path.join(args.visualization_dir, f'recon_sample_{i}_epoch_{epoch+1}.nii.gz')
                    )               
                          
                model.train()
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # 注意访问.module获取原始模型
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss
            }
            
            
            torch.save(checkpoint, os.path.join(args.save_dir, 'latest_checkpoint.pth'))
            
            # 保存最佳模型
            if (epoch + 1) % args.eval_every == 0 and avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'vae_3d_best.pth'))
                logging.info(f"保存最佳模型 (损失: {best_loss:.6f})")
            
            # 定期保存模型
            if (epoch + 1) % args.save_interval == 0:
                torch.save(model.module.state_dict(), 
                           os.path.join(args.save_dir, f'vae_3d_epoch_{epoch+1}.pth'))
                logging.info(f"模型已保存到 {args.save_dir}/vae_3d_epoch_{epoch+1}.pth")
    
    # 训练结束，关闭TensorBoard写入器
    if rank == 0:
        writer.close()
        torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'vae_3d_final.pth'))
        logging.info(f"最终模型已保存到 {args.save_dir}/vae_3d_final.pth")
    
    # 清理
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='使用DDP和TensorBoard训练带残差块的3D VAE模型')
    # 原有参数
    parser.add_argument('--latent_dim', type=int, default=1024, help='潜在向量维度')
    parser.add_argument('--batch_size', type=int, default=1, help='每个进程的批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    parser.add_argument('--num_workers', type=int, default=0, help='每个进程的数据加载线程数')

    parser.add_argument('--save_interval', type=int, default=2, help='模型保存间隔')
    parser.add_argument('--data_dir', type=str, default=None, help='真实数据目录')
    parser.add_argument('--recon_loss', type=str, default='mse', choices=['bce', 'mse'], help='重构损失类型')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值，0表示不裁剪')
    parser.add_argument('--resume', action='store_true', help='从最新检查点恢复训练')
    parser.add_argument('--log_file', type=str, default='/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dscm/exp/vae_ddp_training.log', help='日志文件路径')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='进程数量，通常等于GPU数量')
    parser.add_argument('--init_method', type=str, default='env://', help='分布式初始化方法')
    
    # TensorBoard相关参数
    parser.add_argument('--run_name', type=str, default='run_1', help='当前运行的名称，用于区分不同实验')
    parser.add_argument('--vis_samples', type=int, default=5, help='用于可视化的样本数量')
    parser.add_argument('--vis_interval', type=int, default=2, help='可视化的epoch间隔')
    parser.add_argument('--log_steps', action='store_true', help='是否记录每一步的损失，不仅是每个epoch')
    parser.add_argument('--input_res', type=int, default=160, help='输入体积的分辨率，假设为立方体')
    parser.add_argument('--eval_every', type=int, default=2, help='验证集评估的epoch间隔（0表示不评估）')
    parser.add_argument('--exp_dir', type=str, default='/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dscm/exp', help='模型保存目录')
    
    
    args = parser.parse_args()
    args.save_dir = os.path.join(args.exp_dir, 'ckpts')
    args.visualization_dir = os.path.join(args.exp_dir, 'visualization')
    args.log_dir = os.path.join(args.exp_dir, 'logs')
    
    # 检查是否有可用GPU
    if args.world_size > torch.cuda.device_count():
        args.world_size = torch.cuda.device_count()
        print(f"调整world_size为可用GPU数量: {args.world_size}")
    
    # 设置环境变量，用于进程间通信
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 选择一个空闲端口
    
    # 启动多个进程
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    # 确保CUDA可用
    assert torch.cuda.is_available(), "DDP训练需要至少一个GPU"
    main()
    