import os
import torch
import numpy as np
from torch.utils.data import IterableDataset
import cv2
import json
import webdataset as wds
from io import BytesIO
import random
from .dataset_augment import DatasetAugmenter

class DatasetLoader(IterableDataset):
    def __init__(self, dataset_dir, input_size=None, visualize_dir=None, 
                 shift_signs=None, vel_offset=0.2, shuffle_buffer_size=None,
                 projection_signs=None, projection_offset=0.2, enable_random_sampling=False):
        self.dataset_dir = dataset_dir
        self.visualize_dir = visualize_dir
        self.enable_random_sampling = enable_random_sampling
        
        self.augmenter = DatasetAugmenter(
            shift_signs=shift_signs,
            vel_offset=vel_offset,
            projection_signs=projection_signs,
            projection_offset=projection_offset
        )
        
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
            
        if self.enable_random_sampling and self.visualize_dir:
            self.sample_indices = set(random.sample(range(self.samples_count), min(100, self.samples_count)))
            
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
    
    
    def __iter__(self):
        dataset = self._create_webdataset()
        
        for img_array, angle_info, metadata_info in dataset:
            angle = float(angle_info['angle'])
            
            (transformed_img, adjusted_angle, transform_type,
             transform_sign) = self.augmenter.apply_augmentation(img_array, angle, self.input_size)
            
            # 新しい横シフト処理：affine変換は既に224x224にクロップ済み
            if transform_type == 'affine':
                # 既に目標サイズにクロップ済みのため、そのまま使用
                img_uint8 = transformed_img
            else:
                # projectionやnoneの場合は既存処理を継続
                target_aspect_ratio = self.input_size[1] / self.input_size[0]
                cropped_img = self.augmenter._apply_center_crop(
                    transformed_img, target_aspect_ratio)
                img_uint8 = cv2.resize(cropped_img, self.input_size[::-1])
            
            should_visualize = (self.visualize_dir and
                                self.enable_random_sampling and
                                self._sample_counter in self.sample_indices)
            
            if should_visualize:
                filename = (f"{self._sample_counter:05d}_{transform_type}"
                           f"{transform_sign:.1f}_angle{adjusted_angle:.3f}.png")
                save_path = os.path.join(self.visualize_dir, filename)
                cv2.imwrite(save_path, img_uint8)
            
            self._sample_counter += 1
            
            img_tensor = (torch.from_numpy(img_uint8)
                          .permute(2, 0, 1).float() / 255.0)
            angle_tensor = torch.tensor([adjusted_angle], dtype=torch.float32)
            
            yield img_tensor, angle_tensor
