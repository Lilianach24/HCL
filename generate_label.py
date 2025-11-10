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
    # load dataset
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
        "Qwen/Qwen2.5-VL-7B-Instruct",
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
    model, processor = init_model()
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
    print('----------------------------clip_labeling-----------------------------')
    clip_labeling('CIFAR100', 'clip')
    print('________clip-label-end_________')
    #
    # print('-----------------------------qwen_labeling-----------------------------')
    # model, processor = init_model()
    #
    # print("==========CIFAR100==========")
    # qwen_labeling('CIFAR100', 'qwen')
    # print('________qwen-label-end_________')
