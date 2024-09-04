from dataset import load_dataset
from dataset import check_image_sizes
from model import CustomViTForImageClassification
from model import train_model 
from model import validate_model
import torch
import os
from torch import nn, optim
from torch.utils.data import DataLoader
import tifffile as tiff
from dataset import TiffDataset
from dataset import check_dataloader_account
from imagesave import create_run_folder
from imagesave import save_plots
from model import clear_gpu_memory
from dataset import collate_fn
import wandb
import torch.nn.functional as F
class WeightedLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(WeightedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, labels, sample_weights):
        # 交叉熵损失
        bce_loss = F.binary_cross_entropy(outputs, labels, reduction='none')
        
        # 焦点损失
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # 加权损失
        weighted_loss = sample_weights * (bce_loss + focal_loss)
        
        return weighted_loss.mean()
def main():
    wandb.init(project="tiger_1,2,3", config={
        "epochs": 200,
        "imgsz": (16, 80, 80),
        "batch_size": 256,
        "learning_rate": 0.005,
        "dropout_rate": 0.1,
    })
    path = "./ALL_DATA/data"
    train_dataset, val_dataset = load_dataset(path)

    print("------------------------------")
    print("检查cuda")
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print("device 是", device)
    print("------------------------------")
    print("创建数据加载器")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    print("检查loader数量")
    check_dataloader_account(train_loader)
    check_dataloader_account(val_loader)
    print("清理GPU")
    clear_gpu_memory()
    print("创建模型")
    model = CustomViTForImageClassification()
    model = model.to(device)

    #criterion = nn.BCELoss()
    criterion = WeightedLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True)
    print("训练开始:")
    run_folder = create_run_folder()
    print("save to", run_folder)
    num_epochs = 200

    metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, run_folder=run_folder)
        
    # 创建运行文件夹并保存图表
    save_plots(run_folder, metrics)
    print("save to", run_folder)

if __name__ == '__main__':
    main()



# from dataset import load_dataset
# from dataset import check_image_sizes
# from model import CustomViT
# from model import train_model 
# from model import validate_model
# import torch
# import os
# from torch import nn, optim
# from torch.utils.data import DataLoader
# import tifffile as tiff
# from dataset import TiffDataset
# from dataset import check_dataloader_account
# from imagesave import create_run_folder
# from imagesave import save_plots
# from model import clear_gpu_memory
# from dataset import collate_fn
# import wandb
# def main():
#     wandb.init(project="tiger_1,2_V2", config={
#         "epochs": 200,
#         "imgsz": (16, 80, 80),
#         "batch_size": 512,
#         "learning_rate": 0.005,
#         "dropout_rate": 0.1,
#     })
#     path = "./ALL_DATA/data"
#     train_dataset, val_dataset = load_dataset(path)
#     # dataset = TiffDataset(path)

#     # if dataset.check_image_names():
#     #     print("所有图像名称均符合要求")
#     # else:
#     #     print("一些图像名称不符合要求")
#     # if check_image_sizes(train_dataset) and check_image_sizes(val_dataset):
#     #     print("所有图像尺寸均符合要求")
#     # else:
#     #     print("一些图像尺寸不符合要求")

#     print("------------------------------")
#     print("检查cuda")
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#     print("device 是", device)
#     print("------------------------------")
#     print("创建数据加载器")
#     train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,collate_fn=collate_fn)
#     print("检查loader数量")
#     check_dataloader_account(train_loader)
#     check_dataloader_account(val_loader)
#     print("清理GPU")
#     clear_gpu_memory()
#     print("创建模型")
#     #pretrainPth = r"C:\Users\16074\Desktop\tiger\pretrained_weights"
#     model = CustomViT()
#     model = model.to(device)

#     #criterion = nn.CrossEntropyLoss()
#     criterion = nn.BCELoss()
#     optimizer = optim.AdamW(model.parameters(), lr=0.005)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, verbose=True)
#     print("训练开始:")
#     run_folder = create_run_folder()
#     print("save to",run_folder)
#     num_epochs = 200

#     metrics = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs,run_folder=run_folder)
        
#         # 创建运行文件夹并保存图表
#     save_plots(run_folder, metrics)
#     print("save to",run_folder)

# if __name__ == '__main__':
#     main()