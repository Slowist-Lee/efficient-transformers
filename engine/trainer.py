import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

from models.transformer import CaptioningTransformer
from data.coco_utils import load_coco_data

class COCODataset(Dataset):
    def __init__(self, data_dict, split='train'):
        self.features = data_dict[f'{split}_features']
        self.captions = data_dict[f'{split}_captions']
        self.image_idxs = data_dict[f'{split}_image_idxs']
        
    def __len__(self):
        return self.captions.shape[0]
        
    def __getitem__(self, idx):
        img_idx = self.image_idxs[idx]
        feature = torch.tensor(self.features[img_idx], dtype=torch.float32)
        caption = torch.tensor(self.captions[idx], dtype=torch.long)
        
        return feature, caption

def main():
    # 固定随机种子，保证可复现性
    torch.manual_seed(231)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. 加载数据字典 (开启 PCA 降维，特征维度通常是 512)
    print("Loading COCO data with PCA features...")
    data = load_coco_data(pca_features=True, max_train=5000) # 可以先用 5000 条跑通
    
    # 2. 构建 DataLoader
    train_dataset = COCODataset(data, split='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=2,   # 开启多进程加速数据准备
        pin_memory=True  # 加速 CPU 内存到 GPU 显存的拷贝
    )

    # 3. 初始化模型
    # 注意：pca_features=True 时，input_dim 应该是 512
    feature_dim = data['train_features'].shape[1] 
    print(f"Feature dimension: {feature_dim}")
    
    model = CaptioningTransformer(
        word_to_idx=data['word_to_idx'],
        input_dim=feature_dim,
        wordvec_dim=256,
        num_heads=4,
        num_layers=2,
        max_length=30
    ).to(device)

    # 4. 设置优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 5. 开始标准训练循环
    num_epochs = 200
    loss_history = []
    model.train()

    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (features, captions) in enumerate(train_loader):
            features = features.to(device)
            captions = captions.to(device)

            # 自回归模型的关键：
            # 模型输入是不包含最后一个词的序列 (Captions_in)
            # 预测目标是不包含第一个词的序列 (Captions_target)
            captions_in = captions[:, :-1]
            captions_target = captions[:, 1:]

            # 前向传播
            logits = model(features, captions_in)

            # 计算 Loss: 需将 logits 展平为 (N*SeqLen, VocabSize)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1]), 
                captions_target.reshape(-1)
            )

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止 Transformer 早期训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            loss_history.append(loss.item())

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # 保存训练好的权重
    torch.save(model.state_dict(), 'transformer_captioning_pca.pth')
    print("Model saved to transformer_captioning_pca.pth")

    # 绘制 Loss 曲线
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    # 1. 确保 image 文件夹存在
    os.makedirs('image', exist_ok=True)
    
    # 2. 保存图片到指定路径，bbox_inches='tight' 可以防止边缘被裁剪
    save_path = os.path.join('image', 'training_loss.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")
    print('Final loss: ', loss_history[-1])

if __name__ == '__main__':
    main()