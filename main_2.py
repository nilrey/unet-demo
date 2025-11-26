import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision

class BetterVideoObjectDetector:
    def __init__(self, model_name='fasterrcnn_resnet50_fpn', confidence_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Загружаем предобученную модель детекции объектов
        if model_name == 'fasterrcnn_resnet50_fpn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif model_name == 'fasterrcnn_mobilenet_v3_large':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # COCO классы
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Целевые классы (люди и транспорт)
        self.target_classes = {
            1: 'person',    # человек
            2: 'bicycle',   # велосипед
            3: 'car',       # машина
            4: 'motorcycle',# мотоцикл
            5: 'airplane',  # самолет
            6: 'bus',       # автобус
            7: 'train',     # поезд
            8: 'truck'      # грузовик
        }
        
        self.colors = {
            0: (0, 255, 255),   # Желтый для всех
            1: (0, 255, 0),    # Зеленый для людей
            3: (255, 0, 0),    # Красный для машин
            6: (0, 0, 255),    # Синий для автобусов
            8: (128, 0, 128)   # Фиолетовый для грузовиков
        }
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def detect_objects(self, frame):
        """Детекция объектов в кадре"""
        # Конвертируем BGR в RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Детекция
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Обработка результатов
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.confidence_threshold and label in self.target_classes:
                detections.append({
                    'box': box.astype(int),
                    'score': score,
                    'label': label,
                    'class_name': self.target_classes[label]
                })
        
        return detections

    def draw_boxes(self, frame, detections):
        """Отрисовка bounding boxes на кадре"""
        result_frame = frame.copy()
        
        for detection in detections:
            box = detection['box']
            label = 0 #detection['label']
            score = detection['score']
            class_name = detection['class_name']
            
            color = self.colors.get(label, (255, 255, 255))
            
            # Рисуем bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 1)
            
            # Рисуем подпись
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(result_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_frame

    def process_video(self, video_path, output_path=None, show_preview=True):
        """Обработка видеофайла"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return
        
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Детекция объектов
            detections = self.detect_objects(frame)
            
            # Отрисовка результатов
            result_frame = self.draw_boxes(frame, detections)
            
            # Добавляем информацию
            # cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(result_frame, f"Detections: {len(detections)}", (10, 60),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if output_path:
                out.write(result_frame)
            
            if show_preview:
                cv2.imshow('Object Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            print(f"Обработано кадров: {frame_count}, Обнаружено: {len(detections)}", end='\r')
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nОбработка завершена. Кадров: {frame_count}")

# Использование
if __name__ == "__main__":
    detector = BetterVideoObjectDetector(confidence_threshold=0.5)
    input_file = "cars_1"
    detector.process_video(f"data/input/{input_file}.mp4", 
                          f"data/output/out-{input_file}-unet-detection.avi", 
                          show_preview=False)