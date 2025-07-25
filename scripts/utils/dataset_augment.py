import numpy as np
import cv2
import random


class DatasetAugmenter:
    
    def __init__(self, shift_signs=None, vel_offset=0.2, 
                 projection_signs=None, projection_offset=0.2):
        self.shift_signs = shift_signs if shift_signs else []
        self.vel_offset = vel_offset
        self.projection_signs = projection_signs if projection_signs else []
        self.projection_offset = projection_offset
        
        self.transform_options = []
        if self.projection_signs:
            for sign in self.projection_signs:
                self.transform_options.append(('projection', sign))
        if self.shift_signs:
            for sign in self.shift_signs:
                self.transform_options.append(('affine', sign))
        
        if not self.transform_options:
            self.transform_options.append(('none', 0.0))
    
    def apply_augmentation(self, img, angle):
        if not self.transform_options:
            return img, angle, 'none', 0.0
        
        transform_type, transform_sign = random.choice(self.transform_options)
        
        if transform_type == 'affine':
            transformed_img = self._apply_horizontal_shift(img, transform_sign)
            adjusted_angle = angle - transform_sign * self.vel_offset
        elif transform_type == 'projection':
            transformed_img = self._apply_projection_transform(img, transform_sign)
            adjusted_angle = angle + transform_sign * self.projection_offset
        else:
            transformed_img = img
            adjusted_angle = angle
        
        return transformed_img, adjusted_angle, transform_type, transform_sign
    
    def _apply_horizontal_shift(self, img, shift_sign):
        if shift_sign == 0.0:
            return img
            
        h, w = img.shape[:2]
        shift = int(shift_sign * w * 0.1)
        
        trans_mat = np.array([[1, 0, shift], [0, 1, 0]], dtype=np.float32)
        img = cv2.warpAffine(img, trans_mat, (w, h), 
                           borderMode=cv2.BORDER_CONSTANT, 
                           borderValue=(0, 0, 0))
        return img
    
    def _apply_projection_transform(self, img, projection_sign):
        if projection_sign == 0.0:
            return img
            
        h, w = img.shape[:2]
        
        focal_length_mm = 2.2
        pixel_size_um = 3.0
        focal_length_px = (focal_length_mm * 1000) / pixel_size_um
        
        scale_x = w / 1920.0
        scale_y = h / 1200.0
        fx = focal_length_px * scale_x
        fy = focal_length_px * scale_y
        
        K = np.array([
            [fx, 0,  w / 2],
            [0,  fy, h / 2], 
            [0,  0,      1]
        ], dtype=np.float32)
        
        angle_deg = projection_sign * 5.0
        angle_rad = np.deg2rad(angle_deg)
        
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0,                 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ], dtype=np.float32)
        
        K_inv = np.linalg.inv(K)
        H = K @ R @ K_inv
        
        warped = cv2.warpPerspective(img, H, (w, h), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))
        
        mask = np.any(warped != 0, axis=2)
        coords = np.column_stack(np.where(mask))
        
        if len(coords) == 0:
            return img
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        valid_region = warped[y_min:y_max+1, x_min:x_max+1]
        
        return cv2.resize(valid_region, (w, h))
    
    def get_augmentation_info(self):
        return {
            'shift_signs': self.shift_signs,
            'vel_offset': self.vel_offset,
            'projection_signs': self.projection_signs,
            'projection_offset': self.projection_offset,
            'total_transform_options': len(self.transform_options),
            'has_affine': bool(self.shift_signs),
            'has_projection': bool(self.projection_signs)
        }