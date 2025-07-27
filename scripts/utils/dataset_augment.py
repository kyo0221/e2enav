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
    
    def apply_augmentation(self, img, angle, target_size=None):
        if not self.transform_options:
            return img, angle, 'none', 0.0
        
        transform_type, transform_sign = random.choice(self.transform_options)
        
        if transform_type == 'affine':
            if target_size is not None:
                transformed_img = self._apply_horizontal_crop(img, transform_sign, target_size)
            else:
                transformed_img = self._apply_horizontal_crop(img, transform_sign)
            adjusted_angle = angle + transform_sign * self.vel_offset
        elif transform_type == 'projection':
            transformed_img = self._apply_projection_transform(img, transform_sign)
            adjusted_angle = angle - transform_sign * self.projection_offset
        else:
            transformed_img = img
            adjusted_angle = angle
        
        return transformed_img, adjusted_angle, transform_type, transform_sign
    
    def _apply_horizontal_crop(self, img, shift_sign, target_size=(224, 224)):
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        max_x_shift = w - target_w
        max_y_shift = h - target_h

        center_x = max_x_shift // 2  # 128
        x_offset = int((shift_sign / 2.0) * center_x)  # -128 to +128
        x_start = center_x + x_offset
        
        y_start = max_y_shift // 2  # 38
        
        # 範囲チェック
        x_start = max(0, min(x_start, max_x_shift))
        y_start = max(0, min(y_start, max_y_shift))
        
        cropped = img[y_start:y_start+target_h, x_start:x_start+target_w]
        
        return cropped
        
    def _apply_horizontal_shift(self, img, shift_sign):
        return self._apply_horizontal_crop(img, shift_sign)
    
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
        
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        corners_h = cv2.perspectiveTransform(corners[None, :, :], H)[0]

        x_coords = corners_h[:, 0]
        y_coords = corners_h[:, 1]

        x_min = max(0, np.ceil(x_coords.min()).astype(int))
        x_max = min(w, np.floor(x_coords.max()).astype(int))
        y_min = max(0, np.ceil(y_coords.min()).astype(int))
        y_max = min(h, np.floor(y_coords.max()).astype(int))

        warped_full = cv2.warpPerspective(img, H, (w, h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))

        # パディングが発生しない範囲でクロップ
        cropped = warped_full[y_min:y_max, x_min:x_max]
        
        return cropped
    
    def _apply_center_crop(self, img, target_aspect_ratio):
        h, w = img.shape[:2]
        current_aspect_ratio = w / h
        
        if current_aspect_ratio > target_aspect_ratio:
            new_w = int(h * target_aspect_ratio)
            new_h = h
        else:
            new_w = w
            new_h = int(w / target_aspect_ratio)

        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        cropped = img[start_y:start_y+new_h, start_x:start_x+new_w]
        
        return cropped
    
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
