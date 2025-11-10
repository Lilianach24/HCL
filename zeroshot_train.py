import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from datetime import datetime

from model.zeroshot import CLIPZeroShotClassifier

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


def get_dataset(dataset_name, data_dir, split='test'):
    """获取指定数据集的测试集"""
    if dataset_name == 'CIFAR100':
        # CIFAR100数据集
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        dataset = datasets.CIFAR100(
            root=data_dir,
            train=(split == 'train'),
            download=True,
            transform=test_transform
        )

        # CIFAR100类别名称
        class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]

        return dataset, class_names

    elif dataset_name == 'stanford_cars':
        # Stanford Cars数据集（手动加载版本）
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 确保数据集已下载
        data_root = os.path.join(data_dir, 'stanford_cars')
        images_dir = os.path.join(data_root, 'car_ims')
        if not os.path.exists(images_dir):
            print("Stanford Cars数据集不存在，请确保已下载并放置在正确路径")
            print(f"期望路径: {images_dir}")
            raise ValueError(f"Stanford Cars数据集未找到: {images_dir}")

        # 加载标注文件
        import scipy.io
        annos_path = os.path.join(data_root, 'cars_annos.mat')
        if not os.path.exists(annos_path):
            raise ValueError(f"标注文件未找到: {annos_path}")

        # 解析标注文件
        annos = scipy.io.loadmat(annos_path)
        annotations = annos['annotations'][0]

        # 创建测试集索引
        test_indices = []
        for i, anno in enumerate(annotations):
            is_test = anno[6][0][0]
            if is_test:
                test_indices.append(i)

        # 创建自定义数据集
        from torch.utils.data import Dataset

        class StanfordCarsDataset(Dataset):
            def __init__(self, root_dir, annos, indices, transform=None):
                self.root_dir = root_dir
                self.annos = annos
                self.indices = indices
                self.transform = transform

                # 加载类别名称
                meta_path = os.path.join(data_root, 'cars_annos.mat')
                meta = scipy.io.loadmat(meta_path)
                self.class_names = [str(name[0]) for name in meta['class_names'][0]]

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                anno_idx = self.indices[idx]
                anno = self.annos[anno_idx]

                # 获取图像路径
                img_path = os.path.join(self.root_dir, anno[0][0])

                # 读取图像
                image = Image.open(img_path).convert('RGB')

                # 获取类别标签（注意：MATLAB索引从1开始，Python从0开始）
                label = anno[5][0][0] - 1

                # 应用变换
                if self.transform:
                    image = self.transform(image)

                return image, label

            def get_class_names(self):
                return self.class_names

        # 创建测试集
        test_dataset = StanfordCarsDataset(
            root_dir=data_root,
            annos=annotations,
            indices=test_indices,
            transform=test_transform
        )

        # 获取类别名称
        class_names = test_dataset.get_class_names()

        return test_dataset, class_names

    elif dataset_name == 'food-101':
        # Food-101数据集
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 确保数据集已下载
        if not os.path.exists(os.path.join(data_dir, 'food-101')):
            print("下载Food-101数据集...")
            # 注意：torchvision不直接支持Food-101，需要手动下载或使用其他方法
            raise ValueError("请先手动下载Food-101数据集到指定目录")

        # 创建数据集
        dataset_dir = os.path.join(data_dir, 'food-101', 'images')
        dataset = datasets.ImageFolder(root=dataset_dir, transform=test_transform)

        # 获取类别名称
        class_names = dataset.classes

        return dataset, class_names

    elif dataset_name == 'tiny-imagenet-200':
        # Tiny ImageNet-200数据集
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 确保数据集已下载
        if not os.path.exists(os.path.join(data_dir, 'tiny-imagenet-200')):
            print("下载Tiny ImageNet-200数据集...")
            # 注意：torchvision不直接支持Tiny ImageNet，需要手动下载
            raise ValueError("请先手动下载Tiny ImageNet-200数据集到指定目录")

        # 创建测试集
        test_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
        dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

        # 获取类别名称
        class_names = dataset.classes

        return dataset, class_names

    elif dataset_name == 'EuroSAT':
        # EuroSAT数据集处理（本地加载版本）
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 确保数据集已下载
        data_path = os.path.join(data_dir, 'EuroSAT')
        taglist_path = os.path.join(data_path, 'EuroSAT_ram_taglist.txt')
        raw_data_path = os.path.join(data_path, 'raw', '2750')

        if not os.path.exists(taglist_path) or not os.path.exists(raw_data_path):
            print("EuroSAT数据集不存在或格式不正确，请确保已下载并放置在正确路径")
            print(f"期望路径结构: {data_path}/EuroSAT_ram_taglist.txt 和 {raw_data_path}/<类别>/<图像>.jpg")
            raise ValueError(f"EuroSAT数据集未找到或格式错误: {data_path}")

        # 读取类别标签映射
        id_dict = {}
        with open(taglist_path, 'r') as f:
            for i, line in enumerate(f):
                id_dict[line.strip()] = i

        # 类别名称（按id排序）
        class_names = sorted(id_dict.keys(), key=lambda x: id_dict[x])

        # 获取所有图像路径
        import glob
        EuroSAT_imgs = glob.glob(f"{raw_data_path}/*/*.jpg")
        EuroSAT_imgs = [img_path.replace('\\', '/') for img_path in EuroSAT_imgs]

        # 获取对应标签
        EuroSAT_labels = [id_dict[img_path.split('/')[-2]] for img_path in EuroSAT_imgs]

        # 创建数据集分割
        from sklearn.model_selection import train_test_split
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            EuroSAT_imgs, EuroSAT_labels, test_size=0.3, random_state=42, stratify=EuroSAT_labels
        )

        # 创建自定义数据集
        from torch.utils.data import Dataset

        class EuroSATDataset(Dataset):
            def __init__(self, image_paths, labels, transform=None):
                self.image_paths = image_paths
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                label = self.labels[idx]

                # 读取图像
                image = Image.open(img_path).convert('RGB')

                # 应用变换
                if self.transform:
                    image = self.transform(image)

                return image, label

        # 根据split参数选择使用训练集或测试集
        if split == 'train':
            dataset = EuroSATDataset(train_imgs, train_labels, transform=test_transform)
        else:  # 'test'
            dataset = EuroSATDataset(test_imgs, test_labels, transform=test_transform)

        return dataset, class_names
    elif dataset_name == 'caltech-101':
        # Caltech-101数据集处理
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 确保数据集已下载
        data_root = os.path.join(data_dir, 'caltech-101')
        images_dir = os.path.join(data_root, '101_ObjectCategories')
        if not os.path.exists(images_dir):
            print("Caltech-101数据集不存在，请确保已下载并放置在正确路径")
            print(f"期望路径: {images_dir}")
            raise ValueError(f"Caltech-101数据集未找到: {images_dir}")

        # 创建自定义数据集
        from torch.utils.data import Dataset

        class Caltech101Dataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform

                # 获取所有类别文件夹
                self.class_folders = [f for f in os.listdir(root_dir)
                                      if os.path.isdir(os.path.join(root_dir, f)) and f != 'BACKGROUND_Google']

                # 排序类别文件夹（确保顺序一致）
                self.class_folders.sort()

                # 构建图像路径和标签列表
                self.images = []
                self.labels = []

                for label, class_folder in enumerate(self.class_folders):
                    class_dir = os.path.join(root_dir, class_folder)
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.images.append(os.path.join(class_dir, img_file))
                            self.labels.append(label)

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_path = self.images[idx]
                label = self.labels[idx]

                # 读取图像
                image = Image.open(img_path).convert('RGB')

                # 应用变换
                if self.transform:
                    image = self.transform(image)

                return image, label

            def get_class_names(self):
                # 移除'_'并首字母大写处理类别名称
                return [f.replace('_', ' ').title() for f in self.class_folders if f != 'BACKGROUND_Google']

        # 创建测试集（Caltech-101没有官方划分，使用全部数据）
        test_dataset = Caltech101Dataset(
            root_dir=images_dir,
            transform=test_transform
        )

        # 获取类别名称（移除背景类）
        class_names = test_dataset.get_class_names()

        return test_dataset, class_names
    elif dataset_name == 'dtd':
        # DTD数据集处理
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 确保数据集已下载
        data_path = os.path.join(data_dir, 'dtd')
        taglist_path = os.path.join(data_path, 'dtd_ram_taglist.txt')
        images_path = os.path.join(data_path, 'images')

        if not os.path.exists(taglist_path) or not os.path.exists(images_path):
            print("DTD数据集不存在或格式不正确，请确保已下载并放置在正确路径")
            print(f"期望路径结构: {data_path}/dtd_ram_taglist.txt 和 {images_path}/<类别>/<图像>.jpg")
            raise ValueError(f"DTD数据集未找到或格式错误: {data_path}")

        # 读取类别标签映射
        id_dict = {}
        with open(taglist_path, 'r') as f:
            for i, line in enumerate(f):
                id_dict[line.strip()] = i

        # 类别名称（按id排序）
        class_names = sorted(id_dict.keys(), key=lambda x: id_dict[x])

        # 获取所有图像路径
        import glob
        import random
        dtd_imgs = glob.glob(f"{images_path}/*/*.jpg")
        dtd_imgs = [img_path.replace('\\', '/') for img_path in dtd_imgs]

        # 提取标签（类别在路径的第4个索引位置）
        dtd_labels = [id_dict[img_path.split('/')[4]] for img_path in dtd_imgs]

        # 打包并打乱
        dtd_dataset = list(zip(dtd_imgs, dtd_labels))
        random.seed(42)  # 使用与项目一致的随机种子
        random.shuffle(dtd_dataset)

        # 按 70% train / 30% test 划分
        train_size = int(0.7 * len(dtd_dataset))
        train_set = dtd_dataset[:train_size]
        test_set = dtd_dataset[train_size:]

        train_imgs, train_labels = zip(*train_set)
        test_imgs, test_labels = zip(*test_set)

        # 创建自定义数据集
        from torch.utils.data import Dataset

        class DTDDataset(Dataset):
            def __init__(self, image_paths, labels, transform=None):
                self.image_paths = image_paths
                self.labels = labels
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                label = self.labels[idx]

                # 读取图像
                image = Image.open(img_path).convert('RGB')

                # 应用变换
                if self.transform:
                    image = self.transform(image)

                return image, label

        # 根据split参数选择使用训练集或测试集
        if split == 'train':
            dataset = DTDDataset(train_imgs, train_labels, transform=test_transform)
        else:  # 'test'
            dataset = DTDDataset(test_imgs, test_labels, transform=test_transform)

        return dataset, class_names
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def generate_text_prompts(class_names, prompt_template="a photo of a {}"):
    """为每个类别生成文本提示"""
    return [prompt_template.format(cls) for cls in class_names]


def visualize_predictions(results, output_file='clip_predictions.png', num_samples=5):
    """可视化预测结果"""
    if results['images'] is None or len(results['images']) < num_samples:
        print("没有足够的图像用于可视化")
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        image = results['images'][i]
        true_label = results['labels'][i]
        pred_label = results['predictions'][i]
        class_names = results['class_names']

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(image.permute(1, 2, 0))  # 调整通道顺序
        plt.title(f"真实: {class_names[true_label]}, 预测: {class_names[pred_label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"预测可视化已保存到: {output_file}")


def save_accuracy_to_file(accuracy, dataset, clip_model, output_dir):
    """将准确率保存到TXT文件"""
    # 创建结果目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 保存准确率到TXT文件
    accuracy_file = os.path.join(output_dir, f"accuracy_{dataset}_{clip_model.replace('/', '-')}.txt")
    with open(accuracy_file, 'w') as f:
        f.write(f"CLIP零样本分类准确率\n")
        f.write(f"数据集: {dataset}\n")
        f.write(f"CLIP模型: {clip_model}\n")
        f.write(f"准确率: {accuracy:.2f}%\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"准确率已保存到: {accuracy_file}")


def main():
    parser = argparse.ArgumentParser(description='CLIP零样本分类评估')
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        choices=['CIFAR100', 'stanford_cars', 'food-101', 'tiny-imagenet-200', 'EuroSAT',
                                 'caltech-101'],
                        help='选择数据集')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='数据集存储路径')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14', help='CLIP模型类型')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--prompt_template', type=str, default="a photo of a {}",
                        help='文本提示模板，{}将被类别名称替换')
    parser.add_argument('--output_dir', type=str, default='./zeroshot_results', help='结果保存目录')
    parser.add_argument('--visualize', type=int, default=5,
                        help='可视化预测结果的样本数，0表示不可视化')
    args = parser.parse_args()

    from datetime import datetime

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 获取数据集
    print(f"准备{args.dataset}数据集...")
    test_dataset, class_names = get_dataset(args.dataset, args.data_dir, split='test')

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化CLIP零样本分类器
    print(f"加载CLIP模型: {args.clip_model}")
    classifier = CLIPZeroShotClassifier(clip_model_name=args.clip_model, device=device)

    # 生成文本提示
    print(f"为{len(class_names)}个类别生成文本提示...")
    text_descriptions = generate_text_prompts(class_names, args.prompt_template)

    # 编码文本描述
    print("编码文本描述...")
    text_features = classifier.encode_text(text_descriptions)

    # 评估零样本分类准确率
    print("开始零样本评估...")
    results = classifier.evaluate(
        test_loader,
        text_features,
        class_names=class_names,
        num_samples_to_visualize=args.visualize if args.visualize > 0 else 0
    )

    # 保存准确率到文件
    save_accuracy_to_file(results['accuracy'], args.dataset, args.clip_model, args.output_dir)

    # 保存完整结果
    result_file = os.path.join(args.output_dir, f"clip_zero_shot_{args.dataset}_{args.clip_model.replace('/', '-')}.pt")
    torch.save(results, result_file)
    print(f"完整结果已保存到: {result_file}")

    # # 可视化预测结果
    # if args.visualize > 0:
    #     visualize_file = os.path.join(args.output_dir,
    #                                   f"clip_predictions_{args.dataset}_{args.clip_model.replace('/', '-')}.png")
    #     visualize_predictions(results, output_file=visualize_file,
    #                           num_samples=min(args.visualize, len(results['images'])))


if __name__ == "__main__":
    main()