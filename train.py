# correct_training.py
"""正确利用BioMedCLIP对比学习特点的训练代码"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F

class ContrastiveUltrasoundDataset(Dataset):
    """对比学习数据集：每个图像配对应的文本描述"""

    def __init__(self, folder_path, frames_per_video=5, transform=None):
        """
        为每个视频生成对应的文本描述
        """
        self.frames_per_video = frames_per_video
        self.transform = transform

        self.images = []
        self.texts = []  # 存储对应的文本描述
        self.labels = []  # 存储0/1标签（用于验证）

        # 处理正常视频
        normal_folder = os.path.join(folder_path, 'normal')
        if os.path.exists(normal_folder):
            for video_file in os.listdir(normal_folder):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    frames = self._extract_frames(os.path.join(normal_folder, video_file))

                    # 为正常视频生成多样化的文本描述
                    normal_texts = [
                        "this is a normal ultrasound image",
                        "healthy ultrasound scan with no abnormalities",
                        "unremarkable ultrasound findings",
                        "normal anatomy visualized on ultrasound",
                        "ultrasound showing typical healthy tissue",
                        "no pathological changes detected on ultrasound",
                        "normal echotexture and organ morphology",
                        "ultrasound examination within normal limits"
                    ]

                    for frame in frames:
                        # 随机选择一个文本描述，增加多样性
                        text = np.random.choice(normal_texts)
                        self.images.append(frame)
                        self.texts.append(text)
                        self.labels.append(0)

        # 处理异常视频
        abnormal_folder = os.path.join(folder_path, 'abnormal')
        if os.path.exists(abnormal_folder):
            for video_file in os.listdir(abnormal_folder):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    frames = self._extract_frames(os.path.join(abnormal_folder, video_file))

                    # 为异常视频生成多样化的文本描述
                    abnormal_texts = [
                        "this is an abnormal ultrasound image",
                        "pathological findings detected on ultrasound",
                        "abnormal ultrasound showing lesion",
                        "ultrasound revealing pathological changes",
                        "abnormal tissue characteristics on ultrasound",
                        "ultrasound showing signs of disease",
                        "pathological morphology on ultrasound examination",
                        "abnormal echotexture suggesting pathology"
                    ]

                    for frame in frames:
                        text = np.random.choice(abnormal_texts)
                        self.images.append(frame)
                        self.texts.append(text)
                        self.labels.append(1)

        print(f"数据集大小: {len(self.images)} 图像-文本对")
        print(f"正常样本: {self.labels.count(0)}, 异常样本: {self.labels.count(1)}")

    def _extract_frames(self, video_path):
        """从视频提取帧"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames-1, self.frames_per_video, dtype=int)

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = cap.read()
                if success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)

        cap.release()
        return frames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        text = self.texts[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, text, label


class ContrastiveTrainer:
    """对比学习训练器 - 真正利用BioMedCLIP的特点"""

    def __init__(self, model, tokenizer, device, learning_rate=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

        # 使用对比损失
        self.criterion = nn.CrossEntropyLoss()

        # 优化器 - 微调所有编码器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )

        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-7
        )

        # 温度参数（可学习）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        print(f"对比学习训练器初始化完成")
        print(f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def contrastive_loss(self, image_features, text_features):
        """
        计算对比损失
        """
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # 创建标签（对角线上的为正样本）
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size).to(self.device)

        # 计算双向对比损失
        loss_image = self.criterion(logits_per_image, labels)
        loss_text = self.criterion(logits_per_text, labels)

        loss = (loss_image + loss_text) / 2

        # 计算准确率
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=1)
            acc = (pred == labels).float().mean()

        return loss, acc

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = 0

        for images, texts, labels in train_loader:
            # 预处理
            images = images.to(self.device)

            # tokenize文本
            text_tokens = self.tokenizer(
                texts,
                context_length=256,
                #truncation=True,
                #padding='max_length',
                #return_tensors='pt'
            ).to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            # BioMedCLIP前向传播
            image_features, text_features, _ = self.model(
                images,
                text_tokens
            )

            # 计算对比损失
            loss, acc = self.contrastive_loss(image_features, text_features)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def validate(self, val_loader):
        """验证 - 使用分类准确率"""
        self.model.eval()
        correct = 0
        total = 0

        # 准备固定的文本特征（用于分类）
        normal_texts = [
            "normal ultrasound", "healthy scan", "unremarkable findings",
            "no abnormality", "normal tissue", "healthy organ"
        ]
        abnormal_texts = [
            "abnormal ultrasound", "pathological findings", "lesion detected",
            "abnormal tissue", "disease present", "pathology visible"
        ]
        all_texts = normal_texts + abnormal_texts

        with torch.no_grad():
            # 预计算文本特征
            text_tokens = self.tokenizer(
                all_texts,
                context_length=256,
                #truncation=True,
                #padding='max_length',
                #return_tensors='pt'
            ).to(self.device)

            text_features = self.model.encode_text(
                text_tokens['input_ids'],
                text_tokens['attention_mask']
            )
            text_features = F.normalize(text_features, dim=-1)

            # 计算normal和abnormal的平均特征
            normal_text_feat = text_features[:len(normal_texts)].mean(dim=0, keepdim=True)
            abnormal_text_feat = text_features[len(normal_texts):].mean(dim=0, keepdim=True)
            text_features = torch.cat([normal_text_feat, abnormal_text_feat])

            for images, _, labels in val_loader:
                images = images.to(self.device)

                # 计算图像特征
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

                # 计算与文本特征的相似度
                similarity = image_features @ text_features.t()
                predictions = torch.argmax(similarity, dim=1)

                correct += (predictions == labels.to(self.device)).sum().item()
                total += len(labels)

        return correct / total

    def train(self, train_loader, val_loader, epochs=20):
        """完整训练"""
        best_val_acc = 0

        print("\n开始对比学习训练...")
        print("="*60)

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_acc = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step()

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"训练损失: {train_loss:.4f}, 训练对比准确率: {train_acc:.4f}")
            print(f"验证分类准确率: {val_acc:.4f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                }, 'best_contrastive_model.pth')
                print(f"✓ 保存最佳模型，验证准确率: {val_acc:.4f}")

        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
        return best_val_acc


# 主训练函数
def main_contrastive_training():
    """主训练函数"""

    # 配置
    data_folder = 'F:\\swpu\\ML_DATA\\4ch视频二分类数据集\\4ch_binary_dataset_b_mode_train'
    batch_size = 8
    epochs = 20
    learning_rate = 1e-5
    frames_per_video = 5

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("\n加载BioMedCLIP模型...")
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )

    # 准备数据集
    print("\n准备数据集...")
    dataset = ContrastiveUltrasoundDataset(
        folder_path=data_folder,
        frames_per_video=frames_per_video,
        transform=preprocess
    )

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch],
            torch.tensor([b[2] for b in batch])
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch],
            torch.tensor([b[2] for b in batch])
        )
    )

    # 创建训练器
    trainer = ContrastiveTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate
    )

    # 开始训练
    best_acc = trainer.train(train_loader, val_loader, epochs=epochs)

    print(f"\n训练完成！最佳准确率: {best_acc:.4f}")

    return trainer


if __name__ == "__main__":
    trainer = main_contrastive_training()