import torch
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse


class CLIPZeroShotClassifier:
    def __init__(self, clip_model_name="ViT-L/14", device="cuda"):
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.device = device
        self.model = self.model.to(device)
        self.model.eval() 

    def encode_text(self, text_descriptions):
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def predict(self, image, text_features):
        """Zero-shot prediction for a single image"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs, indices = similarity.cpu().topk(5) 

        return probs, indices

    def evaluate(self, dataloader, text_features, class_names=None, num_samples_to_visualize=0):
        """Evaluate zero-shot classification accuracy"""
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_images = [] 

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predictions = similarity.argmax(dim=-1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if i < num_samples_to_visualize:
                    all_images.extend(images.cpu())

        accuracy = 100.0 * correct / total

        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'images': all_images[:num_samples_to_visualize] if num_samples_to_visualize > 0 else None,
            'class_names': class_names

        }

