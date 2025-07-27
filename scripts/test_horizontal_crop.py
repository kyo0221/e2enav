#!/usr/bin/env python3

import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset_augment import DatasetAugmenter

def create_test_image(width=480, height=300):
    """テスト用の480x300画像を作成（グリッドパターン）"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # グリッドパターンを描画
    grid_size = 50
    for i in range(0, width, grid_size):
        cv2.line(img, (i, 0), (i, height), (100, 100, 100), 1)
    for i in range(0, height, grid_size):
        cv2.line(img, (0, i), (width, i), (100, 100, 100), 1)
    
    # 中央に赤い円を描画
    center_x, center_y = width // 2, height // 2
    cv2.circle(img, (center_x, center_y), 30, (0, 0, 255), -1)
    
    # 左上に番号を描画
    cv2.putText(img, f'{width}x{height}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def test_horizontal_crop():
    """横シフトクロップ処理のテスト"""
    print("🧪 テスト開始: 横シフト+直接クロップ処理")
    
    # テスト用画像作成
    test_img = create_test_image(480, 300)
    
    # DatasetAugmenter初期化
    shift_signs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    augmenter = DatasetAugmenter(
        shift_signs=shift_signs,
        vel_offset=0.4
    )
    
    # 結果保存ディレクトリ
    output_dir = "/tmp/horizontal_crop_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # 元画像を保存
    cv2.imwrite(os.path.join(output_dir, "original_480x300.png"), test_img)
    
    print(f"📁 結果保存先: {output_dir}")
    print(f"🔧 テスト対象shift_signs: {shift_signs}")
    
    # 各shift_signでテスト
    for shift_sign in shift_signs:
        print(f"\n📋 Testing shift_sign = {shift_sign}")
        
        # 新しい横シフトクロップ処理
        cropped_img = augmenter._apply_horizontal_crop(
            test_img, shift_sign, target_size=(224, 224)
        )
        
        print(f"  元画像サイズ: {test_img.shape}")
        print(f"  切り出し後サイズ: {cropped_img.shape}")
        
        # 切り出し位置の計算（確認用）
        h, w = test_img.shape[:2]
        max_x_shift = w - 224  # 256
        center_x = max_x_shift // 2  # 128
        x_offset = int((shift_sign / 2.0) * center_x)
        x_start = center_x + x_offset
        y_start = (h - 224) // 2  # 38
        
        print(f"  計算上の切り出し位置: x={x_start}, y={y_start}")
        print(f"  切り出し範囲: [{y_start}:{y_start+224}, {x_start}:{x_start+224}]")
        
        # 結果画像に情報を追加
        result_img = cropped_img.copy()
        cv2.putText(result_img, f'shift={shift_sign}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_img, f'pos=({x_start},{y_start})', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 保存
        filename = f"cropped_shift_{shift_sign:+.1f}.png"
        cv2.imwrite(os.path.join(output_dir, filename), result_img)
        
        # 角速度オフセットテスト
        original_angle = 0.5
        adjusted_angle = original_angle - shift_sign * augmenter.vel_offset
        print(f"  角速度: {original_angle:.3f} → {adjusted_angle:.3f} (オフセット: {-shift_sign * augmenter.vel_offset:.3f})")
    
    print(f"\n✅ テスト完了！結果は {output_dir} に保存されました。")
    print("\n📊 期待される結果:")
    print("  - shift_sign = -2.0: 左端から切り出し (x=0)")
    print("  - shift_sign = 0.0:  中央から切り出し (x=128)") 
    print("  - shift_sign = +2.0: 右端から切り出し (x=256)")
    print("  - 全て224x224サイズ")
    print("  - パディングなし")

def test_augmentation_workflow():
    """完全なデータ拡張ワークフローのテスト"""
    print("\n🔄 テスト: 完全なデータ拡張ワークフロー")
    
    test_img = create_test_image(480, 300)
    
    augmenter = DatasetAugmenter(
        shift_signs=[-1.0, 0.0, 1.0],
        vel_offset=0.4
    )
    
    # 複数回実行してランダム性を確認
    for i in range(5):
        transformed_img, adjusted_angle, transform_type, transform_sign = \
            augmenter.apply_augmentation(test_img, 0.5, target_size=(224, 224))
        
        print(f"  実行{i+1}: type={transform_type}, sign={transform_sign:+.1f}, "
              f"angle={adjusted_angle:.3f}, size={transformed_img.shape}")

if __name__ == "__main__":
    test_horizontal_crop()
    test_augmentation_workflow()