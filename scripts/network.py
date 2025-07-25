#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2


class Network(nn.Module):
    def __init__(self, input_channels=3):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(1152, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)

        self.cnn_layer = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.conv5,
            self.relu,
            self.flatten
        )

        self.fc_layer = nn.Sequential(
            self.fc1,
            self.relu,
            self.dropout,
            self.fc2,
            self.relu,
            self.dropout,
            self.fc3,
            self.relu,
            self.dropout,
            self.fc4,
            self.relu,
            self.dropout,
            self.fc5
        )
    
    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.fc_layer(x)
        return x
    
    def save_torchscript(self, model_path, input_size=(3, 66, 200)):
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
    def preprocess_image(image, target_size=(200, 66)):

        if isinstance(image, np.ndarray):
            resized = cv2.resize(image, target_size)
            normalized = resized.astype(np.float32) / 255.0

            if len(normalized.shape) == 3:
                normalized = np.transpose(normalized, (2, 0, 1))

            if len(normalized.shape) == 3:
                normalized = np.expand_dims(normalized, axis=0)
            return torch.from_numpy(normalized)
        else:
            raise ValueError("Input image must be numpy array")


def create_model(input_channels=3):
    return Network(input_channels)


if __name__ == "__main__":
    print("🧪 Testing Network...")
    
    model = create_model()
    
    dummy_input = torch.randn(1, 3, 66, 200)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item():.6f}")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "test_model.pt")
        model.save_torchscript(model_path)
        
        loaded_model = Network.load_torchscript(model_path, device='cpu')
        loaded_output = loaded_model(dummy_input)
        
        print(f"Loaded output: {loaded_output.item():.6f}")
        print(f"Difference: {abs(output.item() - loaded_output.item()):.8f}")
    
    print("✅ All tests passed!")