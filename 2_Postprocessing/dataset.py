
import os
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torchvision.transforms as transforms
import cv2

class TiffDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.tif'):
                file_path = os.path.join(root_dir, file_name)
                self.file_paths.append(file_path)
                
                try:
                    class_num = int(file_name.split('_')[-1].split('.')[0])
                    label = 0 if class_num in [1, 2,3] else 1
                except ValueError:
                    continue
                
                self.labels.append(label)
                
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = tiff.imread(file_path)
        label = self.labels[idx]
        
        # 提取 target 信息
        target = int(file_path.split('_')[-1].split('.')[0])
        if self.transform:
            image = self.transform(image)
        if image.shape != (16, 80, 80):
            print("size wrong", image.shape)
            image = np.transpose(image, (1, 0, 2))
        if image.shape != (16, 80, 80):
            raise ValueError(f"Unexpected image shape: {image.shape}, expected (16, 80, 80)")
        return image, label, target

def random_rotation(image, angle_range=(-30, 30)):
    angle = np.random.uniform(*angle_range)
    return np.array([cv2.rotate(slice, cv2.ROTATE_90_CLOCKWISE) for slice in image])

def random_blur(image, ksize=3):
    return np.array([cv2.GaussianBlur(slice, (ksize, ksize), 0) for slice in image])

def custom_transform(image):
    image = random_rotation(image)
    image = random_blur(image)
    return image

def load_dataset(path, num_classes=2, split_ratio=0.8):
    train_transform = custom_transform
    val_transform = None
    dataset = TiffDataset(path, transform=None)
    
    # 按 8:2 比例划分训练集和验证集
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 为训练集和验证集分别应用不同的转换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # 输出统计信息
    total_files = len(dataset)
    class0_count = sum(1 for label in dataset.labels if label == 0)
    class1_count = sum(1 for label in dataset.labels if label == 1)
    
    print(f"总文件数: {total_files}")
    print(f"Class 0 文件数: {class0_count}")
    print(f"Class 1 文件数: {class1_count}")
    print(f"训练集文件数: {train_size}")
    print(f"验证集文件数: {val_size}")
    
    return train_dataset, val_dataset

def check_image_sizes(dataset, expected_shape=(16, 80, 80)):
    for idx in range(len(dataset)):
        image, label, _ = dataset[idx]
        if image.shape != expected_shape:
            return False
    return True

def check_dataloader_account(loader):
    train_class_counts = count_samples_per_class(loader)
    print(f"训练集每个类别的样本数量: {train_class_counts}")

def count_samples_per_class(data_loader):
    class_counts = {}
    for _, labels, targets in data_loader:
        for label, target in zip(labels, targets):
            label = label.item()
            target = int(target)  # 确保 target 是整数
            if label not in class_counts:
                class_counts[label] = {'total': 0, 'targets': {}}
            class_counts[label]['total'] += 1
            if target in class_counts[label]['targets']:
                class_counts[label]['targets'][target] += 1
            else:
                class_counts[label]['targets'][target] = 1
    return class_counts

def collate_fn(batch):
    images, labels, targets = zip(*batch)
    images = [torch.tensor(image).float() / 255.0 for image in images]
    images = torch.stack(images, dim=0)
    if images.shape[1:] != (16, 80, 80):
        images = images.permute(0, 2, 1, 3)  # 调整维度
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(targets, dtype=torch.float32)
    return images, labels, targets





# import os
# import tifffile as tiff
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from sklearn.model_selection import train_test_split
# import numpy as np
# import torchvision.transforms as transforms
# import cv2

# class TiffDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.file_paths = []
#         self.labels = []
        
#         for file_name in os.listdir(root_dir):
#             if file_name.endswith('.tif'):
#                 file_path = os.path.join(root_dir, file_name)
#                 self.file_paths.append(file_path)
                
#                 try:
#                     class_num = int(file_name.split('_')[-1].split('.')[0])
#                     label = 0 if class_num in [1, 2] else 1
#                 except ValueError:
#                     continue
                
#                 self.labels.append(label)
                
#     def __len__(self):
#         return len(self.file_paths)
    
#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         image = tiff.imread(file_path)
#         label = self.labels[idx]
        
#         # 提取 target 信息
#         target = int(file_path.split('_')[-1].split('.')[0])
#         if self.transform:
#             image = self.transform(image)
#         if image.shape != (16, 80, 80):
#             print("size wrong", image.shape)
#             image = np.transpose(image, (1, 0, 2))
#         if image.shape != (16, 80, 80):
#             raise ValueError(f"Unexpected image shape: {image.shape}, expected (16, 80, 80)")
#         return image, label, target

#     def check_image_names(self):
#         for file_path, label in zip(self.file_paths, self.labels):
#             file_name = os.path.basename(file_path)
#             try:
#                 class_num = int(file_name.split('_')[-1].split('.')[0])
#                 if (label == 0 and class_num not in [1, 2, 3]) or (label == 1 and class_num not in [0, 4, 5, 6]):
#                     return False
#             except ValueError:
#                 return False
#         return True

# def random_rotation(image, angle_range=(-30, 30)):
#     angle = np.random.uniform(*angle_range)
#     return np.array([cv2.rotate(slice, cv2.ROTATE_90_CLOCKWISE) for slice in image])

# def random_blur(image, ksize=3):
#     return np.array([cv2.GaussianBlur(slice, (ksize, ksize), 0) for slice in image])

# def custom_transform(image):
#     image = random_rotation(image)
#     image = random_blur(image)
#     return image

# def load_dataset(path, num_classes=2, split_ratio=0.8):
#     train_transform = custom_transform
#     val_transform = None
#     dataset = TiffDataset(path, transform=None)
    
#     # 按 8:2 比例划分训练集和验证集
#     train_size = int(split_ratio * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
#     # 为训练集和验证集分别应用不同的转换
#     train_dataset.dataset.transform = train_transform
#     val_dataset.dataset.transform = val_transform
    
#     # 输出统计信息
#     total_files = len(dataset)
#     class0_count = sum(1 for label in dataset.labels if label == 0)
#     class1_count = sum(1 for label in dataset.labels if label == 1)
    
#     print(f"总文件数: {total_files}")
#     print(f"Class 0 文件数: {class0_count}")
#     print(f"Class 1 文件数: {class1_count}")
#     print(f"训练集文件数: {train_size}")
#     print(f"验证集文件数: {val_size}")
    
#     return train_dataset, val_dataset

# def check_image_sizes(dataset, expected_shape=(16, 80, 80)):
#     for idx in range(len(dataset)):
#         image, label = dataset[idx]
#         if image.shape != expected_shape:
#             return False
#     return True

# def check_dataloader_account(loader):
#     train_class_counts = count_samples_per_class(loader)
#     print(f"训练集每个类别的样本数量: {train_class_counts}")


# def count_samples_per_class(data_loader):
#     class_counts = {}
#     for _, labels, targets in data_loader:
#         for label, target in zip(labels, targets):
#             label = label.item()
#             target = int(target)  # 确保 target 是整数
#             if label not in class_counts:
#                 class_counts[label] = {'total': 0, 'targets': {}}
#             class_counts[label]['total'] += 1
#             if target in class_counts[label]['targets']:
#                 class_counts[label]['targets'][target] += 1
#             else:
#                 class_counts[label]['targets'][target] = 1
#     return class_counts


# def collate_fn(batch):
#     images, labels, targets = zip(*batch)
#     # Convert numpy arrays to tensors and ensure they have the correct shape
#     images = [torch.tensor(image).float() / 255.0 for image in images]  # 归一化到 [0, 1]
#     # Stack images to form a batch
#     images = torch.stack(images, dim=0)
#     labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # 确保标签是 float32 类型并添加一个维度
#     targets = torch.tensor(targets, dtype=torch.float32)  # 确保 targets 是 float32 类型
#     return images, labels, targets
