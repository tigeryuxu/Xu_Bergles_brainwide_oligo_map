import torch
import numpy as np
import tifffile
from torch import nn
from transformers import ViTForImageClassification, ViTConfig

class CustomViTForImageClassification(nn.Module):
    def __init__(self):
        super(CustomViTForImageClassification, self).__init__()
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        config.image_size = 80
        config.num_channels = 16 
        self.model = ViTForImageClassification(config)
        self.model.classifier = nn.Linear(config.hidden_size, 1) 

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        return logits
    
def normalize_image(image, mean, std):
    image = (image - mean) / std
    return torch.tensor(image, dtype=torch.float32)

def predict(model, image, mean, std, device):
    model.eval()
    with torch.no_grad():
        image = normalize_image(image, mean, std)
        image = image.unsqueeze(0)
        image = image.to(device)
        output = model(image)
        prediction = torch.sigmoid(output).item()
    return prediction

def load_tif_image(filepath):
    image = tifffile.imread(filepath)
    return image

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomViTForImageClassification()
    #model_1,2,3
    #model.load_state_dict(torch.load(r"./model_1,2,3.pth",map_location=device))
    model.load_state_dict(torch.load(r"./model_1,2.pth",map_location=device))
    model = model.to(device)

    
    image_path = r'./5otx5_2yo_fused_MaMuT_corrected_CROP_LARGEOL_36_val_1.tif'
    image = load_tif_image(image_path)
    print(image.shape)
    
    #if you using model 1,2,3 please use here
    #mean = 373.46493034947963
    #std = 168.13003627997972
    #if you using model 1,2please use here
    mean = 376.16833740270033
    std = 169.88323430016285

    prediction = predict(model, image, mean, std, device)
    print("prediction is", prediction)
    if(prediction < 0.5):
        print("Prediction: ", 0)
    else:
        print("Prediction: ",1)

