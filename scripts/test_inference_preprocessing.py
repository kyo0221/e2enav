#!/usr/bin/env python3

import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network import Network

def create_test_image(width=480, height=300):
    """テスト用の480x300画像を作成"""
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
    
    # 方向を示す矢印を描画
    cv2.arrowedLine(img, (center_x, center_y), (center_x, center_y - 50), (0, 255, 0), 3)
    
    # サイズ情報を描画
    cv2.putText(img, f'{width}x{height}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def test_inference_preprocessing():
    """推論時の前処理テスト"""
    print("🧪 テスト開始: 推論時画像前処理（中央クロップ）")
    
    # 異なるサイズでテスト
    test_cases = [
        (480, 300, "standard_training_size"),
        (640, 480, "larger_image"),
        (320, 240, "smaller_image"),
        (224, 224, "exact_target_size")
    ]
    
    output_dir = "/tmp/inference_preprocessing_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 結果保存先: {output_dir}")
    
    for width, height, case_name in test_cases:
        print(f"\n📋 Testing {case_name}: {width}x{height}")
        
        # テスト画像作成
        test_img = create_test_image(width, height)
        
        # 推論時前処理実行
        processed_tensor = Network.preprocess_image(test_img, target_size=(224, 224))
        
        # テンソルをnumpy配列に戻す（可視化用）
        processed_np = processed_tensor.squeeze(0).permute(1, 2, 0).numpy()
        processed_uint8 = (processed_np * 255).astype(np.uint8)
        
        print(f"  入力サイズ: {test_img.shape}")
        print(f"  出力テンソルサイズ: {processed_tensor.shape}")
        print(f"  出力画像サイズ: {processed_uint8.shape}")
        
        # 処理方法を判定
        if width >= 224 and height >= 224:
            x_start = (width - 224) // 2
            y_start = (height - 224) // 2
            print(f"  処理方法: 中央クロップ (位置: {x_start}, {y_start})")
        else:
            print(f"  処理方法: リサイズ")
        
        # 元画像を保存
        cv2.imwrite(os.path.join(output_dir, f"{case_name}_original.png"), test_img)
        
        # 処理後画像を保存
        cv2.imwrite(os.path.join(output_dir, f"{case_name}_processed.png"), processed_uint8)
        
        # サイドバイサイド比較画像を作成
        if test_img.shape[0] != processed_uint8.shape[0]:
            # サイズが異なる場合は元画像をリサイズ
            original_resized = cv2.resize(test_img, (224, 224))
        else:
            original_resized = test_img
            
        comparison = np.hstack([original_resized, processed_uint8])
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Processed", (234, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_dir, f"{case_name}_comparison.png"), comparison)
    
    print(f"\n✅ テスト完了！結果は {output_dir} に保存されました。")
    print("\n📊 期待される結果:")
    print("  - 480x300: 中央クロップ (x=128, y=38)")
    print("  - 640x480: 中央クロップ (x=208, y=128)")
    print("  - 320x240: リサイズ処理")
    print("  - 224x224: そのまま使用")

def test_training_consistency():
    """訓練時と推論時の一貫性テスト"""
    print("\n🔄 テスト: 訓練時と推論時の一貫性")
    
    # 訓練時と同じ画像を作成
    test_img = create_test_image(480, 300)
    
    # 推論時前処理（中央クロップ）
    inference_tensor = Network.preprocess_image(test_img, target_size=(224, 224))
    
    # 訓練時前処理をシミュレート（shift_sign=0.0）
    from utils.dataset_augment import DatasetAugmenter
    augmenter = DatasetAugmenter(shift_signs=[0.0])
    training_processed = augmenter._apply_horizontal_crop(test_img, 0.0, target_size=(224, 224))
    
    print(f"  推論時テンソル形状: {inference_tensor.shape}")
    print(f"  訓練時画像形状: {training_processed.shape}")
    
    # 訓練時画像をテンソル形式に変換
    training_tensor = (torch.from_numpy(training_processed)
                      .permute(2, 0, 1).float() / 255.0)
    training_tensor = training_tensor.unsqueeze(0)
    
    # 差分を計算
    diff = torch.abs(inference_tensor - training_tensor).mean()
    print(f"  前処理の差分（平均絶対誤差）: {diff.item():.6f}")
    
    if diff.item() < 1e-6:
        print("  ✅ 訓練時と推論時の前処理が一致しています")
    else:
        print("  ❌ 訓練時と推論時の前処理に差異があります")

if __name__ == "__main__":
    import torch
    test_inference_preprocessing()
    test_training_consistency()