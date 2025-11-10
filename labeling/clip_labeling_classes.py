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

def build_clip_label_embedding(model, categories):     #使用 CLIP 模型生成类别标签的嵌入，返回一个包含所有类别标签嵌入的张量。
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
                for template in templates   #对于每个category使用不同模板生成多个text
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]  # 改造句子
            texts = clip.tokenize(texts)  # tokenize，将文本列表转换为 CLIP 模型需要的 token 格式
            #print("texts =", texts)
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)   #对嵌入进行归一化，确保每个嵌入的范数为 1
            text_embedding = text_embeddings.mean(dim=0)    #对多个生成的文本嵌入取均值
            text_embedding /= text_embedding.norm()     #再次归一化
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)   #将所有类别的标签嵌入沿着新维度堆叠成一个张量。最终的形状是(num_categories,embedding_size)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()   #将张量转置
    return openset_label_embedding

class CIFAR100_generate_label_clip(Dataset):        #clip生成标签与真实标签做对比并写入文件保存
    def __init__(self, root, dataset, X, Y, input_size, transform=None):
        self.root = root
        self.dataset = dataset
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6) #定义了一个余弦相似度计算模块，它可以计算两个张量沿指定维度的余弦相似度。
        torch.manual_seed(1)
        np.random.seed(1)
        clip_model = load_clip()
        info = load_taglist(dataset="CIFAR100")     #加载 CIFAR-100 数据集的标签列表
        taglist_label = info["taglist"]     #提取 CIFAR-100 数据集的标签列表
        label_embed_label = build_clip_label_embedding(clip_model, taglist_label)       #使用 CLIP 模型为每个标签构建一个嵌入表示
        label_embed = label_embed_label.repeat(1, 1, 1)     # 对标签嵌入进行复制，使其能够与图像嵌入进行逐一比较。
        label_embed = label_embed.to(device)
        all_probs = [] # 保存标签概率
        temperature = 0.01
        file_path = os.path.join(self.root, self.dataset, "CLIP-L14")
        os.makedirs(file_path, exist_ok=True)
        for i in range(len(self.X)):
            x = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))       #将原始图像数据转换为 PIL 图像对象。
            imgs = self.transform(x).unsqueeze(0)       #对图像应用预定义的转换，并增加一个批次维度（因为 CLIP 模型一次处理一个批次的图像）
            imgs = imgs.to(device)
            image_embeds = clip_model.encode_image(imgs).unsqueeze(1)   #将图像通过 CLIP 模型进行编码，得到图像的嵌入表示。unsqueeze(1) 是为了添加一个额外的维度，使得嵌入的形状适合后续处理。
            image_embeds = image_embeds.to(device)
            image_to_label = image_embeds.repeat(1, 100, 1)     #将图像嵌入重复 100 次，以与所有标签进行对比（CIFAR-100 有 100 个类别）。
            output = self.cos(image_to_label, label_embed)      #计算余弦相似度
            output_T = output / temperature
            prob = F.softmax(output_T, dim=1)           # 计算标签概率
            all_probs.append(prob.squeeze(0).detach().cpu().numpy())
            print(f"{i + 1}/{len(self.X)}")
            _, labels_g = torch.max(output, dim=1)      #从 output 中找到最大值的索引，表示预测的标签。
            if self.Y[i] == labels_g :  # 检查 self.Y[i] 是否在前两个最相似的索引中
                self.YT[i] = 1  # 如果在，则标记为 1
                file = open(f'{file_path}/train_label_tf.txt', 'a')
                file.write("1\n")
                file.close()
            else:
                self.YT[i] = 0  # 否则标记为 0
                file = open(f'{file_path}/train_label_tf.txt', 'a')
                file.write("0\n")
                file.close()
            file = open(f'{file_path}/train_label_t.txt', 'a')
            file.write(str(self.Y[i]) + '\n')
            file.close()
            file = open(f'{file_path}/train_label_pre.txt', 'a')
            file.write(str(labels_g.item()) + '\n')
            file.close()
        np.save(f'{file_path}/label_prob.npy', np.array(all_probs))  # 修改保存文件名
        print("标记完成并保存 softmax 概率到 .npy 文件。")

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
        label_embed = label_embed_label.repeat(1, 1, 1)     #代表在一二三个维度都复制一次
        label_embed = label_embed.to(device)
        all_probs = []  # 保存标签概率
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
            image_to_label = image_embeds.repeat(1, num_classes, 1)     ##每个图像嵌入需要与每个标签类别的嵌入进行对比。因此，我们需要将图像嵌入扩展，使其能与所有类别的标签嵌入进行比较。
            output = self.cos(image_to_label, label_embed)
            output_T = output / temperature
            prob = F.softmax(output_T, dim=1)  # 计算标签概率
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
            np.save(f'{file_path}/label_prob.npy', np.array(all_probs))  # 修改保存文件名

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


class CIFAR100_handler_test(Dataset):
    def __init__(self, X, Y, input_size, transform=None):
        self.X = X
        self.Y = Y
        self.transform = get_transform(input_size)

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

class DatasetHandlerTest(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.X[index])
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)
