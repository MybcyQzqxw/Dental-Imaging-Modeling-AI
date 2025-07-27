"""
模型评估脚本
用于评估训练好的模型性能
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate_classification_model(model_path, test_data_path, class_names):
    """评估分类模型"""
    import tensorflow as tf
    from utils.data_preprocessing import load_oral_lesions_data
    
    print("加载分类模型...")
    model = tf.keras.models.load_model(model_path)
    
    print("加载测试数据...")
    images, labels = load_oral_lesions_data(
        test_data_path, class_names, target_size=(90, 90)
    )
    
    print("进行预测...")
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 分类报告
    print("\\n分类报告:")
    report = classification_report(
        labels, predicted_classes, target_names=class_names
    )
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = model_path.replace('.h5', '_confusion_matrix.png')
    plt.savefig(save_path)
    print(f"混淆矩阵保存到: {save_path}")
    plt.show()
    
    # 计算准确率
    accuracy = np.mean(labels == predicted_classes)
    print(f"\\n测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return report, accuracy


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='测试数据路径')
    parser.add_argument('--model_type', choices=['classification', 'detection'],
                       default='classification', help='模型类型')
    
    args = parser.parse_args()
    
    print("模型评估系统")
    print("="*40)
    print(f"模型路径: {args.model_path}")
    print(f"测试数据路径: {args.test_data_path}")
    print(f"模型类型: {args.model_type}")
    print("="*40)
    
    if args.model_type == 'classification':
        class_names = [
            'Cold_Sores', 'Canker_Sores', 'abfraction', 'Thrush', 'Gingival_Cyst'
        ]
        evaluate_classification_model(
            args.model_path, args.test_data_path, class_names
        )
    
    print("评估完成!")


if __name__ == "__main__":
    main()
