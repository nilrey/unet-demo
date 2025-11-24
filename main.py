import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision
from collections import defaultdict
from datetime import datetime
from clearml import Task, Dataset, Logger

class BetterVideoObjectDetector:
    def __init__(self, model_name='fasterrcnn_resnet50_fpn', confidence_threshold=0.7):
        self.model_name = model_name
        # Инициализируем ClearML Task
        self.task = Task.init(
            project_name="U-Net",
            task_name=f"car_tracking_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=Task.TaskTypes.inference
        )
        # Устанавливаем параметры задачи
        self.task.set_parameter("model", "U-Net")
        # self.task.set_parameter("confidence_threshold", "N\A")
        # self.task.set_parameter("iou_threshold", 0.4)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Загружаем предобученную модель детекции объектов
        if self.model_name == 'fasterrcnn_resnet50_fpn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif self.model_name == 'fasterrcnn_mobilenet_v3_large':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        
        self.model.to(self.device)
        self.model.eval()
        
        # COCO классы
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck'
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
        self.task.set_parameter("allowed_classes", self.target_classes)
        
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
            label_id = 0 # detection['label']
            score = detection['score']
            class_name = detection['class_name']
            
            color = self.colors.get(label_id, (255, 255, 255))
            
            # Рисуем bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 1)
            
            # Рисуем подпись
            label_text = f"{class_name}"
            cv2.putText(result_frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_frame

    def process_video(self, video_path, output_path=None, show_preview=False):
        """Обработка видеофайла"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return
        
        # Инициализация видеозаписи для output_path
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Видео будет сохранено в: {output_path}")
        
        object_counts = []  # для гистограммы
        frame_changes = []  # для анализа стабильности
        previous_count = 0
        frame_count = 0
        total_objects = 0
        objects_per_class = defaultdict(int)
        
        print("Начата обработка видео...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Детекция объектов
            detections = self.detect_objects(frame)
            
            # Подсчет объектов
            objects_in_frame = len(detections)
            total_objects += objects_in_frame
            
            # Подсчет объектов по классам
            for detection in detections:
                class_name = detection['class_name']
                objects_per_class[class_name] += 1
            
            # Вывод информации о текущем кадре в консоль
            print(f"Frame {frame_count}: objects: {objects_in_frame}")
            
            # Отрисовка результатов
            result_frame = self.draw_boxes(frame, detections)
            
            # Добавляем информацию на кадр
            # cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(result_frame, f"Detections: {objects_in_frame}", (10, 60),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Сохранение результата в файл
            if output_path:
                out.write(result_frame)
            
            if show_preview:
                cv2.imshow('Object Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1

            # Сохраняем для гистограммы
            object_counts.append(objects_in_frame)
            
            # Анализ изменений между фреймами
            if frame_count > 0:
                change = abs(objects_in_frame - previous_count)
                frame_changes.append(change)
            
            previous_count = objects_in_frame 

            # Логируем количество объектов для текущего фрейма
            Logger.current_logger().report_scalar(
                title="Object Detection Statistics",
                series="Objects per Frame",
                value=objects_in_frame,
                iteration=frame_count
            )
            
            # Логируем накопленную статистику
            Logger.current_logger().report_scalar(
                title="Object Detection Statistics", 
                series="Total Objects Detected",
                value=total_objects,
                iteration=frame_count
            )
            
            # Логируем среднее количество объектов на фрейм
            if frame_count > 0:
                avg_objects = total_objects / (frame_count + 1)
                Logger.current_logger().report_scalar(
                    title="Object Detection Statistics",
                    series="Average Objects per Frame",
                    value=avg_objects,
                    iteration=frame_count
                )

            # Запись обработанного кадра в выходное видео
            out.write(frame) 
            
            # Периодический вывод в консоль для отладки
            if frame_count % 30 == 0:  # Каждые 30 фреймов
                print(f"Frame {frame_count}: {objects_in_frame} objects detected")


        
        cap.release()
        if output_path:
            out.release()
            print(f"Результат сохранен в файл: {output_path}")
        cv2.destroyAllWindows()

        # После завершения видео - логируем PLOTS
        if object_counts:
            # Гистограмма распределения объектов
            Logger.current_logger().report_histogram(
                title="Object Detection Analysis",
                series=f"Objects per Frame - {self.model_name}",
                values=object_counts,
                xaxis="Number of Objects",
                yaxis="Number of Frames"
            )
            
            # Гистограмма стабильности трекинга
            if frame_changes:
                Logger.current_logger().report_histogram(
                    title="Tracking Stability Analysis", 
                    series=f"Frame-to-Frame Changes - {self.model_name}",
                    values=frame_changes,
                    xaxis="Objects Change Count", 
                    yaxis="Frequency"
                )


        # Сохраняем итоговую статистику
        self.task.get_logger().report_single_value("Total Frames Processed", frame_count)
        self.task.get_logger().report_single_value("Total Objects Detected", total_objects)
        self.task.get_logger().report_single_value("Average Objects per Frame", total_objects / max(frame_count, 1))

        # Загружаем обработанное видео как артефакт
        self.task.upload_artifact("processed_video", output_path)        
        
        # Вывод итоговой статистики
        print("\n" + "="*50)
        print("="*50)
        print(f"Total Frames: {frame_count}")
        print(f"Total Objects: {total_objects}")
        # print("\nРаспределение по классам:")
        # for class_name, count in objects_per_class.items():
        #     print(f"  {class_name}: {count}")
        print("="*50)

# Использование
if __name__ == "__main__":
    detector = BetterVideoObjectDetector(confidence_threshold=0.5)
    input_file = "cars_1_1"
    time_start = datetime.now()
    detector.process_video(f"data/input/{input_file}.mp4", 
                          f"data/output/out-{input_file}-resnet50-002.mp4", 
                          show_preview=False)
    print(f'Время работы: {datetime.now() - time_start} сек.')