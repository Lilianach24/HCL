import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.utils import convert_to_rgb, get_transform, load_taglist

device = "cuda" if torch.cuda.is_available() else "cpu"


def resolution_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size))
    ])


def generate_label(model, processor, pil_image, taglist_label, num_classes):
    options = "\n".join([f"{idx}: {name}" for idx, name in enumerate(taglist_label)])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"""[步骤1]分析图片主体特征
                                            [步骤2]对比选项描述相似度
                                            [步骤3]直接输出最匹配一个的数字编号，无需解释,无需输出类别名称，数字编号必须在0-{num_classes - 1}之间
                                            可用标签：{options}
                                            最终答案：数字"""
                 },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text


class CIFAR100_handler_train_TF_saved(Dataset):
    """clip生成标签与真实标签做对比并写入文件保存"""

    def __init__(self, root, dataset, X, Y, input_size, model, processor, num_classes=100):
        self.root = root
        self.dataset = dataset
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.model = model
        self.processor = processor
        info = load_taglist(dataset="CIFAR100")  # 加载 CIFAR-100 数据集的标签列表
        taglist_label = info["taglist"]  # 提取 CIFAR-100 数据集的标签列表
        # all_probs = []  # 保存标签概率
        # epsilon = 0.01
        file_path = os.path.join(self.root, self.dataset, "Qwen_VL_7B_label")
        os.makedirs(file_path, exist_ok=True)
        for i in range(len(self.X)):
            pil_image = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))  # 将原始图像数据转换为 PIL 图像对象。
            output_text = generate_label(model, processor, pil_image, taglist_label, num_classes)
            output_label = int(output_text[0])
            print(f"{i + 1}/{len(self.X)}")
            # prob = np.full(len(taglist_label), epsilon / (len(taglist_label) - 1), dtype=np.float32)
            # prob[output_label] = 1.0 - epsilon
            # all_probs.append(prob)
            # print(f"真实标签{self.Y[i]}模型输出{output_text}预测标签{output_label}")
            if self.Y[i] == output_label:
                self.YT[i] = 1
                file = open(f"{file_path}/train_label_tf.txt", 'a')
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
            file.write(str(output_label) + '\n')  # 将 top2 索引保存到文件中
            file.close()
        # np.save(f'{file_path}/label_prob.npy', np.array(all_probs))
        print("标记完成")

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


class DatasetHandlerTrainClip(Dataset):
    def __init__(self, X, Y, input_size, root, dataset, model, processor, num_classes, transform=None):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.Y1 = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.resolution_transform = resolution_transform(input_size)
        self.root = root
        self.dataset = dataset
        self.model = model
        self.processor = processor
        torch.manual_seed(1)
        np.random.seed(1)
        info = load_taglist(dataset=dataset)
        taglist_label = info["taglist"]
        file_path = os.path.join(self.root, self.dataset, "Qwen_VL_7B_label")
        os.makedirs(file_path, exist_ok=True)
        for i in range(len(self.X)):
            x = Image.open(self.X[i]).convert("RGB")  # 从路径加载图片
            x = self.resolution_transform(x)
            output_text = generate_label(model, processor, x, taglist_label, num_classes)
            output_label = int(output_text[0])
            print(f"{i + 1}/{len(self.X)}")
            # print(f"真实标签{self.YT[i]}预测文本{output_text}预测标签{output_label}")
            if self.Y[i] == output_label:
                self.YT[i] = 1
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
            file.write(str(output_label) + '\n')  # 将 top2 索引保存到文件中
            file.close()
        print("标记完成")

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)

