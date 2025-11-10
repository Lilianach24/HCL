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

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": pil_image},
    #             {"type": "text", "text": f"""[步骤1]分析图片主体特征
    #                                         [步骤2]对比选项描述相似度
    #                                         [步骤3]直接输出最匹配一个的数字编号，无需解释,无需输出类别名称，数字编号必须在0-{num_classes - 1}之间
    #                                         可用标签：{options}
    #                                         最终答案：数字"""
    #              },
    #         ],
    #     }
    # ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": pil_image},
    #             {
    #                 "type": "text",
    #                 "text": (
    #                     f"[Step 1] Carefully analyze the main object and details in the image.\n"
    #                     f"[Step 2] Compare the image with the following candidate labels and select the best match.\n"
    #                     f"[Step 3] Directly output ONLY the numeric index of the most suitable label without any explanation, text, or punctuation.\n"
    #                     f"The numeric index must be an integer between 0 and {num_classes - 1}.\n\n"
    #                     f"Available labels:\n{options}\n\n"
    #                     f"Final answer: output the numeric index only."
    #                 )
    #             },
    #         ],
    #     }
    # ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text","text": f"""Classify this image.
                                            Choose the best matching label from these options:{options}
                                            Return only the index (0 to {num_classes - 1}) of your choice.
                                            Do not return any text, words, or explanations.
                                            """
                },
            ],
        }
    ]

    # 准备输入
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

    # 生成描述
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # # 如果不需要概率，返回预测标签和全零概率数组
    # if not return_probs:
    #     return output_text, np.zeros(num_classes)
    # else:
    #     prob_prompt = f"""给图片匹配以下每个类别的概率（0-100，总和尽量100），用逗号分隔：
    #     {options}
    #     示例输出：30,5,20,...（共{num_classes}个数字）"""
    #
    #     # 同样严格构造输入
    #     prob_messages = [
    #         {"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prob_prompt}]}]
    #     prob_text = processor.apply_chat_template(prob_messages, tokenize=False, add_generation_prompt=True)
    #     prob_image_inputs, _ = process_vision_info(prob_messages)
    #
    #     prob_inputs = processor(
    #         text=[prob_text],
    #         images=prob_image_inputs,
    #         return_tensors="pt"
    #     ).to(model.device)
    #
    #     # 生成概率分数
    #     prob_ids = model.generate(**prob_inputs, max_new_tokens=num_classes * 4,
    #                               pad_token_id=processor.tokenizer.pad_token_id)
    #     prob_text = processor.batch_decode(prob_ids, skip_special_tokens=True)[0].strip()
    #
    #     # 提取数字，忽略其他字符
    #     scores = [int(s) for s in prob_text.split(',') if s.strip().isdigit()]
    #     # 确保长度匹配，不足补0，超出截断
    #     if len(scores) < num_classes:
    #         scores += [0] * (num_classes - len(scores))
    #     else:
    #         scores = scores[:num_classes]
    #     # 转换为0-1的概率（归一化）
    #     total = sum(scores)
    #     probs = np.array(scores) / total if total != 0 else np.ones(num_classes) / num_classes

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
