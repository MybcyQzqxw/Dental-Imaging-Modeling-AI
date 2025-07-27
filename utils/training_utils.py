"""
训练工具模块.

包含训练过程中使用的回调函数和工具
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)


def create_callbacks(model_save_path, log_dir, patience=10):
    """
    创建训练回调函数.
    
    Args:
        model_save_path: 模型保存路径
        log_dir: 日志目录
        patience: 早停耐心值
    
    Returns:
        callbacks: 回调函数列表
    """
    callbacks = []
    
    # 模型检查点
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 早停
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 学习率衰减
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(lr_scheduler)
    
    # TensorBoard
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    callbacks.append(tensorboard)
    
    return callbacks


def plot_training_history(history, save_path=None):
    """
    绘制训练历史.
    
    Args:
        history: 训练历史对象
        save_path: 保存路径（可选）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制准确率
    ax1.plot(history.history['accuracy'], label='训练准确率')
    ax1.plot(history.history['val_accuracy'], label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制损失
    ax2.plot(history.history['loss'], label='训练损失')
    ax2.plot(history.history['val_loss'], label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'训练历史图表已保存到: {save_path}')
    
    plt.show()


def save_model_summary(model, save_path):
    """
    保存模型摘要到文件.
    
    Args:
        model: Keras模型
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\\n'))
    
    print(f'模型摘要已保存到: {save_path}')


def calculate_model_size(model_path):
    """
    计算模型文件大小.
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        文件大小（MB）
    """
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    return 0
