from ultralytics import YOLO

# โหลดโมเดลมาตรฐาน YOLOv8
model = YOLO('yolov8n.pt')  # น้ำหนักน้อย
# บันทึกโมเดลเป็น .pt file
model.save('yolov8/best.pt')