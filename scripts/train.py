#!/usr/bin/env python3

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ament_index_python.packages import get_package_share_directory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils.dataset_loader import DatasetLoader
from network import Network


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            package_dir = get_package_share_directory('e2enav')
            config_path = os.path.join(package_dir, 'config', 'training.yaml')
            params_path = os.path.join(package_dir, 'config', 'params.yaml')
        else:
            config_dir = os.path.dirname(config_path)
            params_path = os.path.join(config_dir, 'params.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['train']
        
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)['simple_inference_node']['ros__parameters']
            config['image_height'] = params['image_height']
            config['image_width'] = params['image_width']
        
        self.package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.result_dir = os.path.join(self.package_dir, 'logs', 'training_result')
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.model_dir = os.path.join(self.package_dir, 'config', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.shuffle = config.get('shuffle', True)
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.model_filename = config['model_filename']
        
        self.shift_signs = config.get('shift_signs', [-2.0, -1.0, 0.0, 1.0, 2.0])
        self.vel_offset = config.get('vel_offset', 0.2)
        
        # ResNet18 settings
        self.use_pretrained_resnet = config.get('use_pretrained_resnet', True)
        self.freeze_resnet_backbone = config.get('freeze_resnet_backbone', False)
        
        print(f"Training configuration loaded from: {config_path}")
        print(f"Results will be saved to: {self.result_dir}")
        print(f"Models will be saved to: {self.model_dir}")



class Trainer:
    def __init__(self, config, dataset):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.loader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            # num_workers=os.cpu_count() // 2,
            num_workers=0,
            pin_memory=True
        )
        
        self.model = Network(
            input_channels=3,
            use_pretrained=config.use_pretrained_resnet,
            freeze_backbone=config.freeze_resnet_backbone
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.writer = SummaryWriter(log_dir=config.result_dir)
        
        self.loss_log = []
        self.epoch_losses = []
        
        print(f"Training setup complete:")
        print(f"  Dataset samples: {dataset.samples_count}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Epochs: {config.epochs}")
        print(f"  ResNet18 pretrained: {config.use_pretrained_resnet}")
        print(f"  ResNet18 backbone frozen: {config.freeze_resnet_backbone}")
    
    def train(self):
        scaler = GradScaler()
        torch.backends.cudnn.benchmark = True
        
        print("Starting training...")
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            batch_iter = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", leave=False)
            
            for i, batch in enumerate(batch_iter):
                images, targets = [x.to(self.device) for x in batch]
                
                self.optimizer.zero_grad()
                
                with autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions, targets)
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                loss_value = loss.item()
                self.loss_log.append(loss_value)
                epoch_loss += loss_value
                batch_count += 1
                
                batch_iter.set_postfix({
                    'loss': f'{loss_value:.6f}',
                    'avg_loss': f'{epoch_loss/batch_count:.6f}'
                })
                
                global_step = epoch * 1000 + i
                self.writer.add_scalar('Loss/batch', loss_value, global_step)
            
            avg_epoch_loss = epoch_loss / batch_count
            self.epoch_losses.append(avg_epoch_loss)
            
            self.writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
            self.writer.flush()
            
            print(f"Epoch {epoch+1}/{self.config.epochs} - Average Loss: {avg_epoch_loss:.6f}")
            
            if (epoch + 1) % 10 == 0:
                self.save_intermediate_model(epoch + 1)
        
        self.save_results()
        self.writer.close()
        
        print("Training completed!")
    
    def save_intermediate_model(self, epoch):
        self.model.eval()
        
        dummy_input = torch.randn(1, 3, self.config.image_height, self.config.image_width).to(self.device)
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        base_filename = os.path.splitext(self.config.model_filename)[0]
        extension = os.path.splitext(self.config.model_filename)[1] or '.pt'
        intermediate_filename = f"{base_filename}_{epoch}ep{extension}"
        model_path = os.path.join(self.config.model_dir, intermediate_filename)
        
        traced_model.save(model_path)
        print(f"Intermediate model saved: {model_path}")
        
        self.model.train()
    
    def save_results(self):
        self.model.eval()
        
        dummy_input = torch.randn(1, 3, self.config.image_height, self.config.image_width).to(self.device)
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        model_path = os.path.join(self.config.model_dir, self.config.model_filename)
        traced_model.save(model_path)
        print(f"Final model saved: {model_path}")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_log)
        plt.title("Training Loss (per batch)")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_losses)
        plt.title("Training Loss (per epoch)")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True)
        
        loss_curve_path = os.path.join(self.config.result_dir, 'loss_curves.png')
        plt.tight_layout()
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"Loss curves saved: {loss_curve_path}")
        stats = {
            "final_loss": float(self.epoch_losses[-1]),
            "min_loss": float(np.min(self.epoch_losses)),
            "total_epochs": len(self.epoch_losses),
            "total_batches": len(self.loss_log),
            "model_path": model_path,
            "config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "image_size": [self.config.image_height, self.config.image_width],
                "model_type": "ResNet18_Regression",
                "use_pretrained": self.config.use_pretrained_resnet,
                "freeze_backbone": self.config.freeze_resnet_backbone
            }
        }
        
        import json
        stats_path = os.path.join(self.config.result_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Training stats saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Train simple imitation learning model')
    parser.add_argument('dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--config', type=str, default=None, help='Training config file path')
    parser.add_argument('--visflag', action='store_true', help='Enable visualization to logs/visualize_images')
    
    args = parser.parse_args()
    
    config = Config(config_path=args.config)
    
    dataset_dir = args.dataset
    webdataset_dir = os.path.join(dataset_dir, 'webdataset')
    
    if not os.path.exists(webdataset_dir):
        raise ValueError(f"WebDataset directory not found: {webdataset_dir}")
    
    # 可視化ディレクトリの設定
    visualize_dir = None
    if args.visflag:
        visualize_dir = os.path.join(config.package_dir, 'logs', 'visualize_images')
        os.makedirs(visualize_dir, exist_ok=True)
        print(f"Visualization enabled: {visualize_dir}")
    
    dataset = DatasetLoader(
        dataset_dir=webdataset_dir,
        input_size=(224, 224),
        visualize_dir=visualize_dir,
        shift_signs=config.shift_signs,
        vel_offset=config.vel_offset,
        enable_random_sampling=args.visflag
    )
    
    detected_height, detected_width = dataset.input_size
    config.image_height = detected_height
    config.image_width = detected_width
    print(f"Updated config with detected image size: {detected_width}x{detected_height}")
    
    print(f"Dataset loaded: {dataset.samples_count} samples")
    print(f"  Horizontal shift options: {config.shift_signs}")
    print(f"  Angular velocity offset: {config.vel_offset}")
    print(f"  Horizontal shift augmentation enabled")
    
    trainer = Trainer(config, dataset)
    trainer.train()


if __name__ == '__main__':
    main()