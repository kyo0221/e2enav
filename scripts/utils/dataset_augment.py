import numpy as np
import cv2
import random


class DatasetAugmenter:
    
    def __init__(self, shift_signs=None, vel_offset=0.2):
        self.shift_signs = shift_signs if shift_signs else [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.vel_offset = vel_offset
    
    def apply_augmentation(self, img, angle, target_size=(224, 224)):
        transform_sign = random.choice(self.shift_signs)
        transformed_img = self._apply_horizontal_crop(img, transform_sign, target_size)
        adjusted_angle = angle + transform_sign * self.vel_offset
        
        return transformed_img, adjusted_angle, 'affine', transform_sign
    
    def _apply_horizontal_crop(self, img, shift_sign, target_size=(224, 224)):
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        max_x_shift = w - target_w
        max_y_shift = h - target_h

        center_x = max_x_shift // 2  # 128
        x_offset = int((shift_sign / 2.0) * center_x)
        x_start = center_x + x_offset
        
        y_start = max_y_shift // 2  # 38
        
        # 範囲チェック
        x_start = max(0, min(x_start, max_x_shift))
        y_start = max(0, min(y_start, max_y_shift))
        
        cropped = img[y_start:y_start+target_h, x_start:x_start+target_w]
        
        return cropped
        
    
    def get_augmentation_info(self):
        return {
            'shift_signs': self.shift_signs,
            'vel_offset': self.vel_offset,
            'has_affine': True
        }
