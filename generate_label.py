import os

import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from clip import clip
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from data_loader.data_loader import DatasetLoader


def clip_labeling(dataset, model_name):
    # 加载数据集
    loader, info = DatasetLoader(
        root='./datasets',
        dataset=dataset,
        model=None,
        model_name=model_name,
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        processor=None,
    ).load_datasets()

def init_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "./Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        cache_dir=""
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast=True
    )

    return model, processor

def qwen_labeling(dataset, model_name):
    # 初始化模型
    model, processor = init_model()
    # 加载数据集
    loader, info = DatasetLoader(
        root='./datasets',
        dataset=dataset,
        model=model,
        model_name=model_name,
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        processor=processor,
    ).load_datasets()

def deepseek_labeling(dataset, model_name):
    # specify the path to the model
    # model_path = "deepseek-ai/deepseek-vl-7b-chat"  # 修改模型路径
    # processor = VLChatProcessor.from_pretrained(model_path)  # 修改处理器类
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)  # 修改模型类
    # model = model.to(torch.bfloat16).cuda().eval()
    model_path = "deepseek-ai/deepseek-vl2-tiny"  # 修改为tiny版本
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    # 模型加载和设备设置（保持不变）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 使用bfloat16提高精度
        low_cpu_mem_usage=True,
        device_map="auto"
    ).eval()

    loader, info = DatasetLoader(
        root='./datasets',
        dataset=dataset,
        model=model,
        model_name=model_name,
        pattern='train',
        input_size=224,
        batch_size=1,
        num_workers=0,
        processor=processor,
    ).load_datasets()

if __name__ == '__main__':
    # print('----------------------------clip_labeling-----------------------------')
    # clip_labeling('CIFAR100', 'clip')
    # clip_labeling('EuroSAT', 'clip')
    # print('________clip-label-end_________')
    #
    # print('-----------------------------qwen_labeling-----------------------------')
    # model, processor = init_model()
    #
    # print("==========CIFAR100==========")
    # qwen_labeling('CIFAR100', 'qwen')

    # print("==========caltech-101==========")
    #
    # print("==========stanford_cars==========")
    #
    # print("==========food-101==========")
    #
    # print("==========tiny-imagenet-200==========")
    # print('________qwen-label-end_________')
    print('----------------------------deepseek_labeling-----------------------------')
    deepseek_labeling('CIFAR100', 'deepseek')
