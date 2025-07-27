"""
Dental Imaging AI 配置文件.

包含项目的所有配置参数
"""

# 实际可用的口腔病变类别
AVAILABLE_CLASSES = [
    'abfraction',
    'Canker_Sores',
    'Cold_Sores',
    'Gingival_Cyst',
    'Thrush'
]

# 原计划的完整类别（为将来扩展保留）
PLANNED_CLASSES = [
    'Gingivits', 'Cold_Sores', 'Canker_Sores', 'Periodontitis',
    'Receding_Gum', 'abfraction', 'Thrush', 'Gingival_Cyst'
]

# 数据配置
DATA_CONFIG = {
    'raw_data_path': 'Data/oral lesions_raw',
    'processed_data_path': 'Data/oral_lesions_raw',
    'target_size': (90, 90),
    'batch_size': 32,
    'test_split': 0.2,
    'validation_split': 0.2
}

# 模型配置
MODEL_CONFIG = {
    'num_classes': len(AVAILABLE_CLASSES),
    'input_shape': (90, 90, 3),
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 10
}

# 训练配置
TRAINING_CONFIG = {
    'use_data_augmentation': True,
    'save_best_only': True,
    'monitor': 'val_accuracy',
    'mode': 'max'
}

# 文件路径
PATHS = {
    'models_dir': 'trained_models',
    'logs_dir': 'logs',
    'results_dir': 'results'
}
