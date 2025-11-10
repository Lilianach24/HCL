import torch
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse


class CLIPZeroShotClassifier:
    def __init__(self, clip_model_name="ViT-L/14", device="cuda"):
        # 加载预训练CLIP模型
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()  # 设置为评估模式

    def encode_text(self, text_descriptions):
        """编码类别文本描述"""
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def predict(self, image, text_features):
        """对单张图像进行零样本预测"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算图像与文本的相似度
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs, indices = similarity.cpu().topk(5)  # 获取前5个预测结果

        return probs, indices

    def evaluate(self, dataloader, text_features, class_names=None, num_samples_to_visualize=0):
        """评估零样本分类准确率"""
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_images = []  # 用于可视化的图像

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 编码图像
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 计算相似度并预测
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predictions = similarity.argmax(dim=-1)

                # 统计准确率
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 保存用于可视化的图像
                if i < num_samples_to_visualize:
                    all_images.extend(images.cpu())

        accuracy = 100.0 * correct / total

        # 打印一些预测示例
        if class_names is not None and total > 0:
            print(f"\n随机预测示例 ({min(5, total)}个):")
            sample_indices = np.random.choice(total, min(5, total), replace=False)
            for i in sample_indices:
                true_class = class_names[all_labels[i]]
                pred_class = class_names[all_predictions[i]]
                print(f"真实类别: {true_class}, 预测类别: {pred_class}")

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'images': all_images[:num_samples_to_visualize] if num_samples_to_visualize > 0 else None,
            'class_names': class_names
        }