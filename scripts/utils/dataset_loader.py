import os
import torch
import numpy as np
from torch.utils.data import IterableDataset
import cv2
import json
import webdataset as wds
from io import BytesIO
import random

class DatasetLoader(IterableDataset):
    def __init__(self, dataset_dir, input_size=None, visualize_dir=None, 
                 shift_signs=None, vel_offset=0.2, shuffle_buffer_size=None):
        self.dataset_dir = dataset_dir
        self.visualize_dir = visualize_dir
        
        if shift_signs is None:
            self.shift_signs = [-1.0, 0.0, 1.0]
        else:
            self.shift_signs = shift_signs
        self.vel_offset = vel_offset
        
        self.input_size = self._determine_input_size(input_size)
        
        import glob
        shard_files = glob.glob(os.path.join(dataset_dir, "shard_*.tar*"))
        if not shard_files:
            raise ValueError(f"No shard files found in {dataset_dir}")
        self.shard_pattern = shard_files
        
        self.samples_count = self._count_samples()
        
        if shuffle_buffer_size is None:
            self.shuffle_buffer_size = 2000
        else:
            self.shuffle_buffer_size = shuffle_buffer_size
        
        print(f"Found {len(shard_files)} shard files")
        print(f"Total samples: {self.samples_count}")
        print(f"Shuffle buffer size: {self.shuffle_buffer_size}")
        
        if self.visualize_dir:
            os.makedirs(self.visualize_dir, exist_ok=True)
            
        self._sample_counter = 0
    
    def _determine_input_size(self, input_size):
        if input_size is not None:
            print(f"Using specified input size: {input_size}")
            return input_size
        
        stats_file = os.path.join(self.dataset_dir, "dataset_stats.json")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"Dataset stats file not found: {stats_file}")
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        if 'image_size' not in stats:
            raise KeyError(f"'image_size' not found in dataset stats: {stats_file}")
        
        detected_size = tuple(stats['image_size'])
        print(f"Auto-detected input size from dataset stats: {detected_size}")
        return detected_size
    
    def _handle_sample_format(self, sample):
        if "metadata.json" in sample:
            metadata_data = sample["metadata.json"]
            angle_data = sample["angle.json"]
            
            if isinstance(metadata_data, (str, bytes)):
                metadata_info = json.loads(metadata_data)
            else:
                metadata_info = metadata_data
                
            if isinstance(angle_data, (str, bytes)):
                angle_info = json.loads(angle_data)
            else:
                angle_info = angle_data
        else:
            raise ValueError("Missing metadata.json in sample")
        
        img_array = sample["npy"]
        
        if isinstance(img_array, bytes):
            img_array = np.load(BytesIO(img_array))
        
        return (img_array, angle_info, metadata_info)
    
    def _create_webdataset(self):
        import random
        shuffled_shards = self.shard_pattern.copy()
        random.shuffle(shuffled_shards)
        
        return (
            wds.WebDataset(shuffled_shards, shardshuffle=10)
            .shuffle(self.shuffle_buffer_size)
            .map(self._handle_sample_format)
        )
    
    def _count_samples(self):
        stats_file = os.path.join(self.dataset_dir, "dataset_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                return stats['total_samples']
        return 1000
    
    def _apply_horizontal_shift(self, img, shift_sign):

        if shift_sign != 0.0:
            h, w = img.shape[:2]
            shift = int(shift_sign * w * 0.1)
            
            trans_mat = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
            img = cv2.warpAffine(img, trans_mat, (w, h), 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=(0, 0, 0))
        return img
    
    def __iter__(self):
        dataset = self._create_webdataset()
        
        for img_array, angle_info, metadata_info in dataset:
            shift_sign = random.choice(self.shift_signs)
            
            img_uint8 = cv2.resize(img_array, self.input_size[::-1])
            angle = float(angle_info['angle'])
            
            img_uint8 = self._apply_horizontal_shift(img_uint8, shift_sign)
            adjusted_angle = angle - shift_sign * self.vel_offset
            
            if self.visualize_dir and self._sample_counter < 100:
                save_path = os.path.join(self.visualize_dir, 
                                       f"{self._sample_counter:05d}_shift{shift_sign:.1f}_angle{adjusted_angle:.3f}.png")
                cv2.imwrite(save_path, img_uint8)
            
            self._sample_counter += 1
            
            img_tensor = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0
            angle_tensor = torch.tensor([adjusted_angle], dtype=torch.float32)
            
            yield img_tensor, angle_tensor