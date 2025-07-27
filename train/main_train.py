"""
主训练脚本
用于启动不同模型的训练任务
"""

import os
import sys
import argparse


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI牙科影像检测模型训练')
    parser.add_argument('task', choices=['lesions', 'conditions', 'both'],
                       help='训练任务: lesions(口腔病变分类), '
                            'conditions(口腔疾病检测), both(两者都训练)')
    parser.add_argument('--data_path', type=str, default='./Data',
                       help='数据集根路径 (默认: ./Data)')
    
    args, unknown_args = parser.parse_known_args()
    
    print("AI牙科影像检测模型训练系统")
    print("="*50)
    print(f"选择的训练任务: {args.task}")
    print(f"数据路径: {args.data_path}")
    print("="*50)
    
    # 构建训练脚本的完整路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.task in ['lesions', 'both']:
        print("\\n开始口腔病变分类模型训练...")
        lesions_script = os.path.join(script_dir, 'train_oral_lesions.py')
        lesions_data_path = os.path.join(args.data_path, 'oral_lesions_raw')
        
        cmd = [
            sys.executable, lesions_script,
            '--data_path', lesions_data_path
        ] + unknown_args
        
        print(f"执行命令: {' '.join(cmd)}")
        os.system(' '.join(cmd))
    
    if args.task in ['conditions', 'both']:
        print("\\n开始口腔疾病检测模型训练...")
        conditions_script = os.path.join(script_dir, 'train_oral_conditions.py')
        images_path = os.path.join(args.data_path, 'teeth_raw', '*.JPG')
        annotations_path = os.path.join(args.data_path, 'annotations')
        
        cmd = [
            sys.executable, conditions_script,
            '--images_path', images_path,
            '--annotations_path', annotations_path
        ] + unknown_args
        
        print(f"执行命令: {' '.join(cmd)}")
        os.system(' '.join(cmd))
    
    print("\\n所有训练任务完成!")


if __name__ == "__main__":
    main()
