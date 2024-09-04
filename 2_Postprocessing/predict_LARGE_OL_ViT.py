import torch
import numpy as np
from torch import nn
from transformers import ViTModel, ViTConfig
#Vit model 
class CustomViT(nn.Module):
    def __init__(self, image_size=(16, 80, 80)):
        super(CustomViT, self).__init__()
        
        self.config = ViTConfig(
            num_channels=image_size[0],  
            image_size=image_size[1],    
            patch_size=16,
            num_labels=1,  
            hidden_size=512,
            num_hidden_layers=16,
            num_attention_heads=16,
            intermediate_size=3072,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            dropout_rate=0.1,
            #attention_dropout_rate=0.1
        )
        self.vit = ViTModel(self.config)
        
        self.classifier = nn.Linear(self.config.hidden_size, 1)  
    
    def forward(self, x):
        outputs = self.vit(pixel_values=x)

        cls_output = outputs.last_hidden_state[:, 0, :]
    
        logits = self.classifier(cls_output)
        
        return logits
    

# load model
def load_model(model_path, device):
    model = CustomViT().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# input model, image 16*80*80, this method will tranfer image to tensor and predict by model
# output a predicted result
def predict(model, image, device):
    image = torch.tensor(image).unsqueeze(0).to(device).float() 

    with torch.no_grad():
        output = model(image)
        probabilities = torch.sigmoid(output).cpu().numpy()
        predicted = probabilities.round()
    confidence = probabilities.mean()
    return predicted, confidence


# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     #model address
#     model_path = 'best_val_loss_model.pth'
#     model = load_model(model_path, device)
#     #random image
#     random_image = np.random.rand(16, 80, 80).astype(np.float32)
#     prediction = predict(model, random_image, device)

#     print(f'Prediction: {prediction}')

# if __name__ == '__main__':
#     main()
