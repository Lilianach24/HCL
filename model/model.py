import torch
import torch.nn as nn
from clip import clip

torch.manual_seed(1)

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPLinearModel(nn.Module):
    def __init__(self, clip_model_name="ViT-L/14", num_classes=10, pretrained=True):
        super(CLIPLinearModel, self).__init__()

        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip_model = self.clip_model.float().to(device)
        self.clip_model.eval()

        self.feature_dim = self.clip_model.visual.output_dim
        self.linear = nn.Linear(self.feature_dim, num_classes).to(device)

    def forward(self, images):
        with torch.no_grad():
            images = images.to(device).float()
            features = self.clip_model.encode_image(images) 
        logits = self.linear(features)
        return features, logits


