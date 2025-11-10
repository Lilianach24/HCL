import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from llava.llava_data_loader import LLaVADatasetLoader
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def llava_labeling(dataset, model_name):
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    LLaVADatasetLoader(
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
    print('---------------------llava_labeling-------------------------')
    llava_labeling('CIFAR100', 'llava')
    # llava_labeling('caltech-101', 'llava')
    # llava_labeling('food-101', 'llava')
    # llava_labeling('tiny-imagenet-200', 'llava')

