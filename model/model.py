import torch
import torch.nn as nn
from clip import clip

torch.manual_seed(1)

device = "cuda" if torch.cuda.is_available() else "cpu"


# 网络架构 - 用CLIP提取特征后接线性层
class CLIPLinearModel(nn.Module):
    def __init__(self, clip_model_name="ViT-L/14", num_classes=10, pretrained=True):
        super(CLIPLinearModel, self).__init__()
        # 加载CLIP模型并转换为FP32
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip_model = self.clip_model.float().to(device)  # 转换为FP32并移动到设备
        self.clip_model.eval()

        self.feature_dim = self.clip_model.visual.output_dim
        self.linear = nn.Linear(self.feature_dim, num_classes).to(device)  # 线性层保持FP32

    def forward(self, images):
        with torch.no_grad():
            # 输入图像需为FP32 Tensor（由transform保证）
            images = images.to(device).float()  # 确保输入是FP32
            features = self.clip_model.encode_image(images)  # 输出FP32特征
        logits = self.linear(features)
        return features, logits

