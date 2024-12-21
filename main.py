from ultralytics import YOLO
import cv2
import torch
import time
import numpy as np
from PIL import Image
import albumentations as A
from pathlib import Path

class FaceDetectionSystem:
    def __init__(self):
        print("正在初始化人脸检测系统...")
        # 加载预训练的人脸检测模型
        self.face_detector = YOLO('yolov5s.pt')
        
        # 定义图像增强转换
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        
        # 人脸特征数据库（实际应用中应该使用数据库存储）
        self.face_database = {}
        
    def apply_augmentation(self, image):
        """应用图像增强"""
        augmented = self.augmentation(image=image)
        return augmented['image']
    
    def train_on_dataset(self, dataset_path):
        """在数据集上训练和验证"""
        print(f"开始处理数据集: {dataset_path}")
        dataset_path = Path(dataset_path)
        
        # 处理每个人的文件夹
        for person_dir in dataset_path.glob("*"):
            if person_dir.is_dir():
                person_name = person_dir.name
                print(f"处理 {person_name} 的图像...")
                
                # 处理该人的所有图像
                for img_path in person_dir.glob("*.jpg"):
                    # 读取图像
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    # 应用图像增强（用于验证增强效果）
                    augmented_image = self.apply_augmentation(image)
                    
                    # 保存增强后的图像（用于对比）
                    aug_save_path = img_path.parent / f"aug_{img_path.name}"
                    cv2.imwrite(str(aug_save_path), augmented_image)
                    
                    # 这里可以添加特征提取和存储逻辑
                    
    def detect_face_realtime(self):
        """实时人脸检测和识别"""
        print("启动实时人脸检测...")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 人脸检测
            results = self.face_detector(frame)
            
            # 处理检测结果
            if len(results) > 0:
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 提取人脸区域
                        face_roi = frame[y1:y2, x1:x2]
                        
                        # 这里可以添加人脸识别和性别分类的代码
                        # 示例标注
                        label = "未知人员 (性别: 未知)"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('实时人脸检测', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    system = FaceDetectionSystem()
    
    # 1. 首先在数据集上训练和验证
    dataset_path = "face_dataset"  # 替换为你的数据集路径
    system.train_on_dataset(dataset_path)
    
    # 2. 启动实时检测
    system.detect_face_realtime()

if __name__ == '__main__':
    main()
