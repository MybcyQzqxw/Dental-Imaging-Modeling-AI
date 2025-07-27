"""
模型推理脚本
用于对新图像进行预测
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def predict_oral_lesion(model_path, image_path, target_size=(90, 90)):
    """预测口腔病变类别"""
    import tensorflow as tf
    
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    
    # 加载和预处理图像
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # 预测
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence, predictions[0]


def visualize_prediction(image_path, predicted_class, confidence, 
                        all_predictions, class_names):
    """可视化预测结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示原图
    image = Image.open(image_path)
    ax1.imshow(image)
    ax1.set_title(f'输入图像\\n预测: {class_names[predicted_class]}\\n'
                  f'置信度: {confidence:.2f}')
    ax1.axis('off')
    
    # 显示预测概率
    ax2.bar(range(len(class_names)), all_predictions)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_ylabel('预测概率')
    ax2.set_title('各类别预测概率')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--image_path', type=str, required=True,
                       help='待预测图像路径')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    
    args = parser.parse_args()
    
    # 口腔病变类别
    class_names = [
        'Gingivits', 'Cold_Sores', 'Canker_Sores', 'Periodontitis',
        'Receding_Gum', 'abfraction', 'Thrush', 'Gingival_Cyst'
    ]
    
    print("模型推理系统")
    print("="*40)
    print(f"模型路径: {args.model_path}")
    print(f"图像路径: {args.image_path}")
    print("="*40)
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在 {args.image_path}")
        return
    
    # 进行预测
    try:
        predicted_class, confidence, all_predictions = predict_oral_lesion(
            args.model_path, args.image_path
        )
        
        print(f"\\n预测结果:")
        print(f"类别: {class_names[predicted_class]}")
        print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print(f"\\n所有类别概率:")
        for i, (class_name, prob) in enumerate(
            zip(class_names, all_predictions)
        ):
            print(f"{i}: {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # 可视化结果
        if args.visualize:
            visualize_prediction(
                args.image_path, predicted_class, confidence,
                all_predictions, class_names
            )
        
    except Exception as e:
        print(f"推理过程中发生错误: {e}")


if __name__ == "__main__":
    main()
