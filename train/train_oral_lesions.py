"""
口腔病变识别模型训练脚本
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import AVAILABLE_CLASSES, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PATHS
from models.neural_networks import create_oral_lesions_cnn
from utils.data_preprocessing import load_oral_lesions_data, apply_data_augmentation
from utils.training_utils import create_callbacks, plot_training_history
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_oral_lesions_model(data_path, epochs=50, batch_size=32):
    """
    训练口腔病变识别模型
    
    Args:
        data_path: 数据路径
        epochs: 训练轮数
        batch_size: 批次大小
    
    Returns:
        训练好的模型和历史记录
    """
    print("开始训练口腔病变识别模型...")
    print(f"可用类别: {AVAILABLE_CLASSES}")
    print(f"类别数量: {len(AVAILABLE_CLASSES)}")
    
    # 加载数据
    print("正在加载数据...")
    images, labels = load_oral_lesions_data(
        data_path, 
        AVAILABLE_CLASSES, 
        target_size=DATA_CONFIG['target_size']
    )
    
    if len(images) == 0:
        print("错误: 没有找到任何图像数据！")
        print(f"请检查数据路径: {data_path}")
        print(f"预期的类别目录: {AVAILABLE_CLASSES}")
        return None, None
    
    print(f"加载了 {len(images)} 张图像")
    
    # 数据标准化
    images = np.array(images) / 255.0
    labels = np.array(labels)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, 
        test_size=DATA_CONFIG['test_split'], 
        random_state=42, 
        stratify=labels
    )
    
    # 再次划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=DATA_CONFIG['validation_split'], 
        random_state=42, 
        stratify=y_train
    )
    
    print(f"训练集: {len(X_train)} 张")
    print(f"验证集: {len(X_val)} 张")
    print(f"测试集: {len(X_test)} 张")
    
    # 数据增强
    if TRAINING_CONFIG['use_data_augmentation']:
        print("应用数据增强...")
        X_train_aug, y_train_aug = apply_data_augmentation(X_train, y_train)
        X_train = np.vstack([X_train, X_train_aug])
        y_train = np.hstack([y_train, y_train_aug])
        print(f"增强后训练集: {len(X_train)} 张")
    
    # 创建模型
    print("创建模型...")
    model = create_oral_lesions_cnn(
        input_shape=MODEL_CONFIG['input_shape'],
        num_classes=len(AVAILABLE_CLASSES)
    )
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 创建回调函数
    callbacks = create_callbacks(
        model_save_path=os.path.join(PATHS['models_dir'], 'oral_lesions_best.h5'),
        log_dir=os.path.join(PATHS['logs_dir'], 'oral_lesions'),
        patience=MODEL_CONFIG['early_stopping_patience']
    )
    
    # 训练模型
    print("开始训练...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    print("\\n评估模型...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试准确率: {test_accuracy:.4f}")
    
    # 生成分类报告
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\\n分类报告:")
    print(classification_report(
        y_test, y_pred_classes, 
        target_names=AVAILABLE_CLASSES
    ))
    
    # 绘制训练历史
    plot_training_history(history, save_path=os.path.join(PATHS['results_dir'], 'oral_lesions_training_history.png'))
    
    # 保存最终模型
    final_model_path = os.path.join(PATHS['models_dir'], 'oral_lesions_final.h5')
    model.save(final_model_path)
    print(f"模型已保存到: {final_model_path}")
    
    return model, history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练口腔病变识别模型')
    parser.add_argument('--data_path', type=str, 
                       default=DATA_CONFIG['processed_data_path'],
                       help='数据路径')
    parser.add_argument('--epochs', type=int, 
                       default=MODEL_CONFIG['epochs'],
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, 
                       default=DATA_CONFIG['batch_size'],
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    os.makedirs(PATHS['logs_dir'], exist_ok=True)
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    
    print("口腔病变识别模型训练")
    print("="*50)
    print(f"数据路径: {args.data_path}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print("="*50)
    
    # 训练模型
    model, history = train_oral_lesions_model(
        args.data_path, 
        epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    if model is not None:
        print("\\n训练完成！")
    else:
        print("\\n训练失败！")


if __name__ == "__main__":
    main()
