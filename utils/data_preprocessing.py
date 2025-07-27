"""
数据预处理工具
包含图像处理、数据加载和数据增强功能
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import glob
import xmltodict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf


def imflatfield(I, sigma):
    """
    Python实现的imflatfield功能
    用于图像平场校正
    
    Args:
        I: 输入图像，BGR格式，uint8类型
        sigma: 高斯滤波参数
    
    Returns:
        校正后的图像
    """
    tic = time.perf_counter()
    
    A = I.astype(np.float32) / 255
    Ihsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
    A = Ihsv[:, :, 2]
    
    filterSize = int(2*np.ceil(2*sigma) + 1)
    
    # 计算阴影
    shading = cv2.GaussianBlur(A, (filterSize, filterSize), sigma, borderType=cv2.BORDER_REFLECT)
    
    meanVal = np.mean(A)
    
    # 限制最小值为1e-6
    shading = np.maximum(shading, 1e-6)
    
    # 执行平场校正
    B = A / shading
    B = B * meanVal
    
    # 转换回原始格式
    Ihsv[:, :, 2] = B
    B = cv2.cvtColor(Ihsv, cv2.COLOR_HSV2BGR)
    B = np.clip(B * 255, 0, 255).astype(np.uint8)
    
    toc = time.perf_counter()
    print(f"图像平场校正完成，耗时: {toc - tic:0.4f} 秒")
    
    return B


def load_oral_lesions_data(data_path, class_names, target_size=(90, 90)):
    """
    加载口腔病变分类数据
    
    Args:
        data_path: 数据路径
        class_names: 类别名称列表
        target_size: 目标图像尺寸
    
    Returns:
        images: 图像数组
        labels: 标签数组
    """
    images = []
    labels = []
    
    print("开始加载口腔病变数据...")
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"警告: 类别路径不存在 {class_path}")
            continue
            
        # 支持多种图像格式
        image_extensions = ['*.jpeg', '*.jpg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        for image_file in image_files:
            try:
                image = Image.open(image_file)
                image = image.convert('RGB')
                image = image.resize(target_size)
                image_array = np.array(image) / 255.0
                
                images.append(image_array)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"加载图像失败 {image_file}: {e}")
                continue
    
    print(f"数据加载完成: {len(images)} 张图像，{len(set(labels))} 个类别")
    
    return np.array(images), np.array(labels)


def load_oral_conditions_data(images_path, annotations_path, target_size=(480, 480)):
    """
    加载口腔疾病检测数据（YOLO格式或XML格式）
    
    Args:
        images_path: 图像文件路径模式
        annotations_path: 标注文件路径模式
        target_size: 目标图像尺寸
    
    Returns:
        images: 图像数组
        targets: 目标数组（包含边界框和类别）
    """
    images = []
    bboxes = []
    classes_raw = []
    
    print("开始加载口腔疾病检测数据...")
    
    image_files = glob.glob(images_path)
    
    for image_file in image_files:
        try:
            # 加载图像
            image = Image.open(image_file)
            image = image.resize(target_size)
            image_array = np.array(image) / 255.0
            
            # 查找对应的标注文件
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            
            # 尝试XML格式
            xml_file = os.path.join(annotations_path, base_name + '.xml')
            txt_file = os.path.join(annotations_path, base_name + '.txt')
            
            if os.path.exists(xml_file):
                # 解析XML格式
                x = xmltodict.parse(open(xml_file, 'rb'))
                bndbox = x['annotation']['object']['bndbox']
                bbox = np.array([
                    int(bndbox['xmin']), int(bndbox['ymin']),
                    int(bndbox['xmax']), int(bndbox['ymax'])
                ]) / target_size[0]  # 归一化
                
                class_name = x['annotation']['object']['name']
                
                images.append(image_array)
                bboxes.append(bbox)
                classes_raw.append(class_name)
                
            elif os.path.exists(txt_file):
                # 解析YOLO格式
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为边界框格式
                        x_min = x_center - width/2
                        y_min = y_center - height/2
                        x_max = x_center + width/2
                        y_max = y_center + height/2
                        
                        bbox = [x_min, y_min, x_max, y_max]
                        
                        images.append(image_array)
                        bboxes.append(bbox)
                        classes_raw.append(str(class_id))
                        break  # 只处理第一个边界框
            
        except Exception as e:
            print(f"加载数据失败 {image_file}: {e}")
            continue
    
    # 编码类别
    encoder = LabelBinarizer()
    classes_onehot = encoder.fit_transform(classes_raw)
    
    # 合并边界框和类别
    targets = np.concatenate([np.array(bboxes), classes_onehot], axis=1)
    
    print(f"检测数据加载完成: {len(images)} 张图像")
    
    return np.array(images), targets


def create_data_augmentation():
    """
    创建数据增强管道
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", input_shape=(90, 90, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    return data_augmentation


def visualize_oral_lesions_data(images, labels, class_names, num_samples=9):
    """
    可视化口腔病变数据样本
    
    Args:
        images: 图像数组
        labels: 标签数组
        class_names: 类别名称
        num_samples: 显示样本数量
    """
    plt.figure(figsize=(12, 12))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_class_distribution(labels, class_names):
    """
    可视化类别分布
    
    Args:
        labels: 标签数组
        class_names: 类别名称
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), [counts[i] if i in unique else 0 for i in range(len(class_names))])
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylabel('样本数量')
    plt.title('各类别样本分布')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def predict_and_draw_bbox(model, images, output_dir, target_size=(480, 480)):
    """
    预测并绘制边界框
    
    Args:
        model: 训练好的模型
        images: 图像数组
        output_dir: 输出目录
        target_size: 目标图像尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = model.predict(images)
    
    for i, (img, pred) in enumerate(zip(images, predictions)):
        # 恢复边界框坐标
        bbox = pred[:4] * target_size[0]
        
        # 转换图像格式
        img_display = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_display, 'RGB')
        
        # 绘制边界框
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle(bbox, outline="red", width=3)
        
        # 保存结果
        save_path = os.path.join(output_dir, f'prediction_{i+1}.png')
        img_pil.save(save_path)
    
    print(f"预测结果已保存到: {output_dir}")


def prepare_training_data(images, labels, test_size=0.2, random_state=42):
    """
    准备训练数据
    
    Args:
        images: 图像数组
        labels: 标签数组
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        训练和测试数据集
    """
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state
    )
    
    print(f"数据划分完成:")
    print(f"训练集: {x_train.shape[0]} 样本")
    print(f"测试集: {x_test.shape[0]} 样本")
    print(f"图像形状: {x_train.shape[1:]}")
    
    return x_train, x_test, y_train, y_test
