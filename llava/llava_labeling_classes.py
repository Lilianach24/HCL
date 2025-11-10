import os
import numpy as np
import torch
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from utils.utils import convert_to_rgb, get_transform, load_taglist

device = "cuda" if torch.cuda.is_available() else "cpu"


def resolution_transform(image_size=384):
    return Compose([
        convert_to_rgb,
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                  std=[0.26862954, 0.26130258, 0.27577711]) 
    ])


def generate_label_llava(model, processor, pil_image, taglist_label, index=None):
    options = "\n".join([f"{idx}: {name}" for idx, name in enumerate(taglist_label)])

    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": f"""[步骤1]分析图片主体特征
                        [步骤2]对比选项描述相似度
                        [步骤3]直接输出最匹配一个的数字编号，无需解释,无需输出类别名称，数字编号必须在0-{len(taglist_label)-1}之间
                        可用标签：{options}
                        最终答案：数字"""},
                ]
            }
        ]
    
    # prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(0, torch.float16)
    # inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()
    
    try:
        match = re.search(r"\b(\d{1,3})\b", output_text)
        if not match:
            raise ValueError("No numbers were matched.")
    
        output_label = int(match.group(1))
        # if output_label < 0 or output_label >= len(taglist_label):
        #     raise ValueError(f"Number out of range: {output_label}")
    
        return [str(output_label)]
    
    except Exception as e:
        with open("./llava_labeling_errors.log", "a", encoding="utf-8") as logf:
            logf.write(f"[Index {index}] Output: {repr(output_text)} | Error: {e}\n")
        return ["0"]


    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    # )[0]

    # output_label = int(''.join(filter(str.isdigit, output_text)))
    
    # # if 0 <= output_label < len(taglist_label):
    # #     return [str(output_label)]
    # # else:
    # #     raise ValueError("index out of range")
    # return [str(output_label)]



class CIFAR100HandlerTrainLLaVA(Dataset):

    def __init__(self, root, dataset, X, Y, input_size, model, processor):
        self.root = root
        self.dataset = dataset
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.model = model
        self.processor = processor

        info = load_taglist(dataset="CIFAR100")
        self.taglist_label = info["taglist"]

        file_path = os.path.join(self.root, self.dataset, "LLaVA_v15_13B_label")
        os.makedirs(file_path, exist_ok=True)

        for i in range(len(self.X)):
            pil_image = Image.fromarray(np.uint8(self.X[i]).transpose((1, 2, 0)))
            output_text = generate_label_llava(model, processor, pil_image, self.taglist_label)
            output_label = int(output_text[0])
            print(f"{i + 1}/{len(self.X)}")

            if self.Y[i] == output_label:
                self.YT[i] = 1
                with open(f"{file_path}/train_label_tf.txt", 'a') as f:
                    f.write("1\n")
            else:
                self.YT[i] = 0
                with open(f"{file_path}/train_label_tf.txt", 'a') as f:
                    f.write("0\n")

            with open(f"{file_path}/train_label_t.txt", 'a') as f:
                f.write(str(self.Y[i]) + '\n')

            with open(f"{file_path}/train_label_pre.txt", 'a') as f:
                f.write(str(output_label) + '\n')

        print("Label complete!")

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


class DatasetHandlerTrainLLaVA(Dataset):

    def __init__(self, X, Y, input_size, root, dataset, model, processor, transform=None):
        self.X = X
        self.Y = Y
        self.YT = torch.empty(len(self.Y))
        self.transform = get_transform(input_size)
        self.resolution_transform = resolution_transform(input_size)
        self.root = root
        self.dataset = dataset
        self.model = model
        self.processor = processor

        info = load_taglist(dataset=dataset)
        self.taglist_label = info["taglist"]

        file_path = os.path.join(self.root, self.dataset, "LLaVA_v15_13B_label")
        os.makedirs(file_path, exist_ok=True)

        for i in range(len(self.X)):
            x = Image.open(self.X[i]).convert("RGB")
            x = self.resolution_transform(x)

            pil_image = Image.fromarray(np.uint8(x.permute(1, 2, 0).numpy() * 255))

            # output_text = generate_label_llava(model, processor, pil_image, self.taglist_label)
            output_text = generate_label_llava(model, processor, pil_image, self.taglist_label, index=i)
            output_label = int(output_text[0])

            print(f"{i + 1}/{len(self.X)}")

            if self.Y[i] == output_label:
                self.YT[i] = 1
                with open(f"{file_path}/train_label_tf.txt", 'a') as f:
                    f.write("1\n")
            else:
                self.YT[i] = 0
                with open(f"{file_path}/train_label_tf.txt", 'a') as f:
                    f.write("0\n")

            with open(f"{file_path}/train_label_t.txt", 'a') as f:
                f.write(str(self.Y[i]) + '\n')

            with open(f"{file_path}/train_label_pre.txt", 'a') as f:
                f.write(str(output_label) + '\n')

        print("Label complete!")

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.X[index]).transpose((1, 2, 0)))
        x = self.transform(x)
        y = self.Y[index]
        yt = self.YT[index]
        return x, y, yt

    def __len__(self):
        return len(self.X)


