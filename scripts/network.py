#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2


class Network(nn.Module):
    def __init__(self, input_channels=3, use_pretrained=True, freeze_backbone=False):
        super(Network, self).__init__()
        
        # ResNet18 backbone
        if use_pretrained:
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet18 = models.resnet18(weights=None)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Practical regression head for E2E driving (inspired by NVIDIA PilotNet)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 50) 
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize FC layers for regression
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        torch.nn.init.constant_(self.fc2.bias, 0.0)
        torch.nn.init.constant_(self.fc3.bias, 0.0)
        torch.nn.init.constant_(self.fc4.bias, 0.0)
        
        self.regression_head = nn.Sequential(
            self.fc1,
            self.relu,
            self.dropout,
            self.fc2, 
            self.relu,
            self.dropout,
            self.fc3,
            self.relu,
            self.fc4  # No activation on final layer
        )
    
    def forward(self, x):
        x = self.backbone(x)      # ResNet18 feature extraction + avg pool
        x = self.flatten(x)       # Flatten to 512-dim vector  
        x = self.regression_head(x)  # 4-layer MLP: 512→100→50→10→1
        return x
    
    def save_torchscript(self, model_path, input_size=(3, 224, 224)):
        self.eval()
        dummy_input = torch.randn(1, *input_size)
        traced_model = torch.jit.trace(self, dummy_input)
        
        traced_model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        self.train()
    
    @staticmethod
    def load_torchscript(model_path, device='cuda'):
        model = torch.jit.load(model_path)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded from: {model_path}")
        return model
    
    @staticmethod
    def preprocess_image(image, target_size=(224, 224)):
        """
        推論時の画像前処理：中央クロップ方式
        訓練時と同様の480x300→224x224クロップを実行
        """
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            target_h, target_w = target_size
            
            # 元画像が想定サイズ（480x300周辺）の場合は中央クロップ
            if w >= target_w and h >= target_h:
                # 中央から切り出し（shift_sign=0.0相当）
                x_start = (w - target_w) // 2
                y_start = (h - target_h) // 2
                cropped = image[y_start:y_start+target_h, x_start:x_start+target_w]
            else:
                # 小さい画像の場合は従来通りリサイズ
                cropped = cv2.resize(image, target_size)
            
            normalized = cropped.astype(np.float32) / 255.0

            if len(normalized.shape) == 3:
                normalized = np.transpose(normalized, (2, 0, 1))

            if len(normalized.shape) == 3:
                normalized = np.expand_dims(normalized, axis=0)
            return torch.from_numpy(normalized)
        else:
            raise ValueError("Input image must be numpy array")


def create_model(input_channels=3, use_pretrained=True, freeze_backbone=False):
    return Network(input_channels, use_pretrained, freeze_backbone)


if __name__ == "__main__":
    print("🧪 Testing Network...")
    
    # Test different configurations
    configs = [
        {"use_pretrained": True, "freeze_backbone": False, "name": "Pretrained + Trainable"},
        {"use_pretrained": True, "freeze_backbone": True, "name": "Pretrained + Frozen"},
        {"use_pretrained": False, "freeze_backbone": False, "name": "Random Init + Trainable"}
    ]
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    for config in configs:
        print(f"\n📋 Testing: {config['name']}")
        model = create_model(use_pretrained=config['use_pretrained'], 
                           freeze_backbone=config['freeze_backbone'])
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output value: {output.item():.6f}")
        
        # Check if backbone is frozen
        backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
        head_trainable = any(p.requires_grad for p in model.regression_head.parameters())
        print(f"  Backbone trainable: {backbone_trainable}")
        print(f"  Head trainable: {head_trainable}")
    
    # Test TorchScript save/load
    print(f"\n💾 Testing TorchScript save/load...")
    model = create_model()
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.pt")
        model.save_torchscript(model_path)
        
        loaded_model = Network.load_torchscript(model_path, device='cpu')
        loaded_output = loaded_model(dummy_input)
        
        print(f"  Original output: {output.item():.6f}")
        print(f"  Loaded output: {loaded_output.item():.6f}")
        print(f"  Difference: {abs(output.item() - loaded_output.item()):.8f}")
    
    # Test image preprocessing
    print(f"\n🖼️  Testing image preprocessing...")
    test_img_480x300 = np.random.randint(0, 255, (300, 480, 3), dtype=np.uint8)
    test_img_small = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
    
    processed_large = Network.preprocess_image(test_img_480x300)
    processed_small = Network.preprocess_image(test_img_small)
    
    print(f"  480x300 input -> {processed_large.shape} tensor (center crop)")
    print(f"  200x150 input -> {processed_small.shape} tensor (resize)")
    
    print("✅ All tests passed!")