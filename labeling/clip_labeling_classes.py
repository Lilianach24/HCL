from PIL import Image
from torch.utils.data import Dataset
from torch.nn import Module
import numpy as np
import torch
from clip import clip
import torch.nn as nn
from utils.utils import get_transform, load_taglist
import os
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

single_template = ["a photo of a {}."]

def load_clip() -> Module:
    model, _ = clip.load("ViT-L/14")
    return model.to(device).eval()

def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res

def article(name):
    return "an" if name[0] in "aeiou" else "a"

def build_clip_label_embedding(model, categories): 
    # print("Creating pretrained CLIP image model")
    templates = ["a photo of a {}."]
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        openset_label_embedding = []
        for category in categories:
            # print("category =", category)
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates 
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ] 
            texts = clip.tokenize(texts) 
            #print("texts =", texts)
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding

class CIFAR100_generate_label_clip(Dataset):
    def __init__(self, root, dataset, X, Y, input_size, transform=None):
        self.root = root
        self.dataset = dataset
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        torch.manual_seed(1)
        np.random.seed(1)
        clip_model = load_clip()
        info = load_taglist(dataset="CIFAR100") 
        taglist_label = info["taglist"] 
        label_embed_label = build_clip_label_embedding(clip_model, taglist_label)
        label_embed = label_embed_label.repeat(1, 1, 1)
        label_embed = label_embed.to(device)
        all_probs = []
        temperature = 0.01
        file_path = os.path.join(self.root, self.dataset, "CLIP-L14")
        os.makedirs(file_path, exist_ok=True)
        for i in range(len(self.X)):
            x = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))
            imgs = self.transform(x).unsqueeze(0) 
            imgs = imgs.to(device)
            image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
            image_embeds = image_embeds.to(device)
            image_to_label = image_embeds.repeat(1, 100, 1)
            output = self.cos(image_to_label, label_embed)
            output_T = output / temperature
            prob = F.softmax(output_T, dim=1) 
            all_probs.append(prob.squeeze(0).detach().cpu().numpy())
            print(f"{i + 1}/{len(self.X)}")
            _, labels_g = torch.max(output, dim=1) 
            if self.Y[i] == labels_g :
                self.YT[i] = 1 
                file = open(f'{file_path}/train_label_tf.txt', 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0
                file = open(f'{file_path}/train_label_tf.txt', 'a')
                file.write("0\n")
                file.close()
            file = open(f'{file_path}/train_label_t.txt', 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(f'{file_path}/train_label_pre.txt', 'a')
            file.write(str(labels_g.item()) + '\n')
            file.close()
        np.save(f'{file_path}/label_prob.npy', np.array(all_probs))
        print("Complete the marking and save the softmax probabilities to a .npy file.")

    def __getitem__(self, index):
        # print(self.X[index].shape)
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        # x = Image.open(self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)

class Dataset_gengerate_label_clip(Dataset):
    def __init__(self, root, dataset, X, Y, dataset_name, num_classes, input_size):
        self.root = root
        self.dataset = dataset
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.dataset_name = dataset_name
        self.class_num = num_classes
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        torch.manual_seed(1)
        np.random.seed(1)
        clip_model = load_clip()
        info = load_taglist(dataset=dataset_name)
        taglist_label = info["taglist"]
        label_embed_label = build_clip_label_embedding(clip_model, taglist_label)
        label_embed = label_embed_label.repeat(1, 1, 1)
        label_embed = label_embed.to(device)
        all_probs = [] 
        temperature = 0.01
        file_path = os.path.join(self.root, self.dataset, "CLIP-L14")
        os.makedirs(file_path, exist_ok=True)
        for i in range(len(self.X)):
            print(f"{i + 1}/{len(self.X)}")
            x = Image.open(self.X[i])
            imgs = self.transform(x).unsqueeze(0)
            imgs = imgs.to(device)
            image_embeds = clip_model.encode_image(imgs).unsqueeze(1)
            image_embeds = image_embeds.to(device)
            image_to_label = image_embeds.repeat(1, num_classes, 1) 
            output = self.cos(image_to_label, label_embed)
            output_T = output / temperature
            prob = F.softmax(output_T, dim=1)
            all_probs.append(prob.squeeze(0).detach().cpu().numpy())

            _, labels_g = torch.max(output, dim=1)
            if labels_g == self.Y[i]:
                self.YT[i] = 1
                file = open(f'{file_path}/train_label_tf.txt', 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0
                file = open(f'{file_path}/train_label_tf.txt', 'a')
                file.write("0\n")
                file.close()
            file = open(f'{file_path}/train_label_t.txt', 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(f'{file_path}/train_label_pre.txt', 'a')
            file.write(str(labels_g.item()) + '\n')
            file.close()
            np.save(f'{file_path}/label_prob.npy', np.array(all_probs))

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


