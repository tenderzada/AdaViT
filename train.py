# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np

from adaptive_vit import AdaptiveViT

def get_data_loaders(batch_size, image_size=224):
    """获取CIFAR-100数据加载器"""
    # ViT 需要的图像尺寸和归一化
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. 随机采样延迟预算 [cite: 229]
        # 预算范围是 (1/num_adaptive_layers) 到 1.0
        min_budget = 1.0 / model.num_adaptive_layers
        budget = torch.rand(inputs.size(0), device=device) * (1.0 - min_budget) + min_budget

        # 2. 前向传播和计算损失
        optimizer.zero_grad()
        outputs = model(inputs, budget)
        loss = criterion(outputs, labels)
        
        # 3. 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, fixed_budget):
    """在给定预算下评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 创建一个固定的预算
    budget_tensor = torch.full((loader.batch_size,), fixed_budget, device=device)
    
    for inputs, labels in tqdm(loader, desc=f"Evaluating with budget {fixed_budget:.2f}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 对于最后一个可能不满的batch，调整预算张量大小
        if inputs.size(0) != budget_tensor.size(0):
            budget_tensor = torch.full((inputs.size(0),), fixed_budget, device=device)

        outputs = model(inputs, budget_tensor)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据加载
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # 模型
    model = AdaptiveViT(
        timm_model_name='vit_base_patch16_224',
        num_classes=100,
        pretrained=True,
        num_adaptive_layers=args.num_adaptive_layers
    ).to(device)
    
    # 优化器和损失函数
    # 论文中提到为调度器设置不同的学习率，这里为了简化使用统一的学习率
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"\nEpoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
        
        # 在不同预算下进行评估
        budgets_to_test = np.linspace(1.0 / args.num_adaptive_layers, 1.0, 4)
        for budget in budgets_to_test:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device, fixed_budget=budget)
            print(f"  - Budget {budget:.2f}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

    print("Training finished.")
    # 保存模型
    torch.save(model.state_dict(), "adaptive_vit_cifar100.pth")
    print("Model saved to adaptive_vit_cifar100.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train AdaptiveViT on CIFAR-100")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num-adaptive-layers', type=int, default=6, help='Number of final layers to make adaptive')
    
    args = parser.parse_args()
    main(args)