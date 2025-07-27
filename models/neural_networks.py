"""
神经网络模型定义.

包含各种深度学习模型的架构定义
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input,
    concatenate, UpSampling2D, Conv2DTranspose
)
from tensorflow.keras.applications import VGG16, ResNet50
import tensorflow.keras.backend as K


def create_oral_lesions_cnn(input_shape=(90, 90, 3), num_classes=8):
    """
    创建口腔病变分类CNN模型.
    
    Args:
        input_shape: 输入图像形状
        num_classes: 分类数量
    
    Returns:
        编译好的Keras模型
    """
    model = Sequential([
        # 第一个卷积块
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # 第二个卷积块
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # 第三个卷积块
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # 第四个卷积块
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu'),
        GlobalAveragePooling2D(),
        
        # 全连接层
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_transfer_learning_model(input_shape=(90, 90, 3), num_classes=8, 
                                 base_model='vgg16'):
    """
    创建基于迁移学习的模型.
    
    Args:
        input_shape: 输入图像形状
        num_classes: 分类数量
        base_model: 基础模型类型 ('vgg16', 'resnet50')
    
    Returns:
        编译好的Keras模型
    """
    # 选择基础模型
    if base_model == 'vgg16':
        base = VGG16(weights='imagenet', include_top=False, 
                     input_shape=input_shape)
    elif base_model == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, 
                        input_shape=input_shape)
    else:
        raise ValueError(f"不支持的基础模型: {base_model}")
    
    # 冻结基础模型的权重
    base.trainable = False
    
    # 添加自定义头部
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_detection_model(input_shape=(224, 224, 3), num_classes=8):
    """
    创建目标检测模型.
    
    Args:
        input_shape: 输入图像形状
        num_classes: 分类数量
    
    Returns:
        编译好的Keras模型
    """
    inputs = Input(shape=input_shape)
    
    # 特征提取骨干网络
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # 分类头
    cls_head = GlobalAveragePooling2D()(x)
    cls_head = Dense(128, activation='relu')(cls_head)
    cls_head = Dropout(0.5)(cls_head)
    cls_output = Dense(num_classes, activation='softmax', name='classification')(cls_head)
    
    # 边界框回归头
    bbox_head = GlobalAveragePooling2D()(x)
    bbox_head = Dense(128, activation='relu')(bbox_head)
    bbox_head = Dropout(0.5)(bbox_head)
    bbox_output = Dense(4, activation='sigmoid', name='bbox_regression')(bbox_head)
    
    model = Model(inputs=inputs, outputs=[cls_output, bbox_output])
    
    return model


def iou_loss(y_true, y_pred):
    """
    IoU损失函数.
    
    Args:
        y_true: 真实边界框
        y_pred: 预测边界框
    
    Returns:
        IoU损失
    """
    # 计算交集
    x1 = K.maximum(y_true[:, 0], y_pred[:, 0])
    y1 = K.maximum(y_true[:, 1], y_pred[:, 1])
    x2 = K.minimum(y_true[:, 2], y_pred[:, 2])
    y2 = K.minimum(y_true[:, 3], y_pred[:, 3])
    
    intersection = K.maximum(0.0, x2 - x1) * K.maximum(0.0, y2 - y1)
    
    # 计算并集
    area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union = area_true + area_pred - intersection
    
    # 计算IoU
    iou = intersection / K.maximum(union, K.epsilon())
    
    return 1.0 - iou


def iou_metric(y_true, y_pred):
    """
    IoU评估指标.
    
    Args:
        y_true: 真实边界框
        y_pred: 预测边界框
    
    Returns:
        IoU值
    """
    return 1.0 - iou_loss(y_true, y_pred)


def create_unet_model(input_shape=(256, 256, 3), num_classes=1):
    """
    创建U-Net分割模型.
    
    Args:
        input_shape: 输入图像形状
        num_classes: 分割类别数量
    
    Returns:
        编译好的Keras模型
    """
    inputs = Input(input_shape)
    
    # 编码器（下采样路径）
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # 瓶颈层
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # 解码器（上采样路径）
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model
