import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from datetime import datetime

from model.zeroshot import CLIPZeroShotClassifier

torch.manual_seed(42)
np.random.seed(42)


def get_dataset(dataset_name, data_dir, split='test'):
    if dataset_name == 'CIFAR100':
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
    elif dataset_name == 'food-101':
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if not os.path.exists(os.path.join(data_dir, 'food-101')):
            print("The Food-101 dataset does not exist.")
            raise ValueError("Please manually download the Food-101 dataset first.")
        dataset_dir = os.path.join(data_dir, 'food-101', 'images')
        dataset = datasets.ImageFolder(root=dataset_dir, transform=test_transform)
        class_names = dataset.classes

        return dataset, class_names

    elif dataset_name == 'tiny-imagenet-200':
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        if not os.path.exists(os.path.join(data_dir, 'tiny-imagenet-200')):
            print("The Tiny ImageNet-200 dataset does not exist.")
            raise ValueError("Please manually download the Tiny ImageNet-200 dataset first.")
        test_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
        dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
        class_names = dataset.classes

        return dataset, class_names

    elif dataset_name == 'EuroSAT':
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        data_path = os.path.join(data_dir, 'EuroSAT')
        taglist_path = os.path.join(data_path, 'EuroSAT_ram_taglist.txt')
        raw_data_path = os.path.join(data_path, 'raw', '2750')
        if not os.path.exists(taglist_path) or not os.path.exists(raw_data_path):
            print("The EuroSAT dataset does not exist.")
            raise ValueError(f"Please manually download the EuroSAT dataset first.")
        id_dict = {}
        with open(taglist_path, 'r') as f:
            for i, line in enumerate(f):
                id_dict[line.strip()] = i
        class_names = sorted(id_dict.keys(), key=lambda x: id_dict[x])

        import glob
        EuroSAT_imgs = glob.glob(f"{raw_data_path}/*/*.jpg")
        EuroSAT_imgs = [img_path.replace('\\', '/') for img_path in EuroSAT_imgs]
        EuroSAT_labels = [id_dict[img_path.split('/')[-2]] for img_path in EuroSAT_imgs]
        from sklearn.model_selection import train_test_split
        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            EuroSAT_imgs, EuroSAT_labels, test_size=0.3, random_state=42, stratify=EuroSAT_labels
        )
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
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                return image, label

        if split == 'train':
            dataset = EuroSATDataset(train_imgs, train_labels, transform=test_transform)
        else:  # 'test'
            dataset = EuroSATDataset(test_imgs, test_labels, transform=test_transform)

        return dataset, class_names
    elif dataset_name == 'caltech-101':
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        data_root = os.path.join(data_dir, 'caltech-101')
        images_dir = os.path.join(data_root, '101_ObjectCategories')
        if not os.path.exists(images_dir):
            print("The Caltech-101 dataset does not exist.")
            raise ValueError(f"Please manually download the Caltech-101 dataset first.")

        from torch.utils.data import Dataset

        class Caltech101Dataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.class_folders = [f for f in os.listdir(root_dir)
                                      if os.path.isdir(os.path.join(root_dir, f)) and f != 'BACKGROUND_Google']
                self.class_folders.sort()
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
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)

                return image, label

            def get_class_names(self):
                return [f.replace('_', ' ').title() for f in self.class_folders if f != 'BACKGROUND_Google']

        test_dataset = Caltech101Dataset(
            root_dir=images_dir,
            transform=test_transform
        )
        class_names = test_dataset.get_class_names()

        return test_dataset, class_names
    elif dataset_name == 'dtd':
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        data_path = os.path.join(data_dir, 'dtd')
        taglist_path = os.path.join(data_path, 'dtd_ram_taglist.txt')
        images_path = os.path.join(data_path, 'images')

        if not os.path.exists(taglist_path) or not os.path.exists(images_path):
            print("The DTD dataset does not exist.")
            raise ValueError(f"Please manually download the DTD dataset first.")
        id_dict = {}
        with open(taglist_path, 'r') as f:
            for i, line in enumerate(f):
                id_dict[line.strip()] = i
        class_names = sorted(id_dict.keys(), key=lambda x: id_dict[x])
        
        import glob
        import random
        dtd_imgs = glob.glob(f"{images_path}/*/*.jpg")
        dtd_imgs = [img_path.replace('\\', '/') for img_path in dtd_imgs]
        dtd_labels = [id_dict[img_path.split('/')[4]] for img_path in dtd_imgs]
        dtd_dataset = list(zip(dtd_imgs, dtd_labels))
        random.seed(42)
        random.shuffle(dtd_dataset)
        #  70% train / 30% test 
        train_size = int(0.7 * len(dtd_dataset))
        train_set = dtd_dataset[:train_size]
        test_set = dtd_dataset[train_size:]
        train_imgs, train_labels = zip(*train_set)
        test_imgs, test_labels = zip(*test_set)

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
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label

        if split == 'train':
            dataset = DTDDataset(train_imgs, train_labels, transform=test_transform)
        else:  # 'test'
            dataset = DTDDataset(test_imgs, test_labels, transform=test_transform)

        return dataset, class_names
    else:
        raise ValueError(f"Unsupposed dataset: {dataset_name}")


def generate_text_prompts(class_names, prompt_template="a photo of a {}"):
    return [prompt_template.format(cls) for cls in class_names]


def visualize_predictions(results, output_file='clip_predictions.png', num_samples=5):
    """Visualize the prediction results"""
    if results['images'] is None or len(results['images']) < num_samples:
        print("There are not enough images for visualization.")
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        image = results['images'][i]
        true_label = results['labels'][i]
        pred_label = results['predictions'][i]
        class_names = results['class_names']

        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(image.permute(1, 2, 0)) 
        plt.title(f"true label: {class_names[true_label]}, pred label: {class_names[pred_label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"The prediction visualization has been saved to: {output_file}")


def save_accuracy_to_file(accuracy, dataset, clip_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    accuracy_file = os.path.join(output_dir, f"accuracy_{dataset}_{clip_model.replace('/', '-')}.txt")
    with open(accuracy_file, 'w') as f:
        f.write(f"CLIP zero-shot classification accuracy\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"CLIP model: {clip_model}\n")
        f.write(f"accuracy: {accuracy:.2f}%\n")
        f.write(f"datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Accuracy has been saved to: {accuracy_file}")


def main():
    parser = argparse.ArgumentParser(description='CLIP Zero-Shot Classification Assessment')
    parser.add_argument('--dataset', type=str, default='CIFAR100',
                        choices=['CIFAR100', 'stanford_cars', 'food-101', 'tiny-imagenet-200', 'EuroSAT',
                                 'caltech-101'],
                        help='select dataset')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Dataset storage path')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14', help='CLIP model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--prompt_template', type=str, default="a photo of a {}",
                        help='Text prompt template, {} will be replaced with category name')
    parser.add_argument('--output_dir', type=str, default='./zeroshot_results', help='Results save directory')
    parser.add_argument('--visualize', type=int, default=5,
                        help='The number of samples for visualizing the prediction results; 0 indicates no visualization.')
    args = parser.parse_args()

    from datetime import datetime

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"{args.dataset} dataset...")
    test_dataset, class_names = get_dataset(args.dataset, args.data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Load CLIP model: {args.clip_model}")
    classifier = CLIPZeroShotClassifier(clip_model_name=args.clip_model, device=device)
    print(f"Generate text hints for {len(class_names)} categories....")
    text_descriptions = generate_text_prompts(class_names, args.prompt_template)
    print("Encoded text description...")
    text_features = classifier.encode_text(text_descriptions)
    print("Start zero-sample evaluation...")
    results = classifier.evaluate(
        test_loader,
        text_features,
        class_names=class_names,
        num_samples_to_visualize=args.visualize if args.visualize > 0 else 0
    )
    save_accuracy_to_file(results['accuracy'], args.dataset, args.clip_model, args.output_dir)
    result_file = os.path.join(args.output_dir, f"clip_zero_shot_{args.dataset}_{args.clip_model.replace('/', '-')}.pt")
    torch.save(results, result_file)
    print(f"The complete results have been saved to: {result_file}")


if __name__ == "__main__":

    main()
