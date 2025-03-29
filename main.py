import os
import cv2
import torch
import time
import queue
import threading
from datetime import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk
import RPi.GPIO as GPIO
import subprocess
import warnings
import requests
import io
import numpy as np
import datetime

from autoTransform.transform import process_auto_transform
from splitImage.split import process_split_image
from readLicense.read import process_read_license

warnings.filterwarnings("ignore")

# Constants
YOLO_WIDTH = 640
YOLO_HEIGHT = 640
OCR_SIZE = 224
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FPS = 60


API_URL = "http://10.240.67.29:3000"


# GPIO pins
RED_LIGHT_PIN = 16
GREEN_LIGHT_PIN = 18
IR_SENSOR_PIN = 22

# Trigger Zone
ZONE_LEFT = 30
ZONE_RIGHT = 70
ZONE_TOP = 0
ZONE_BOTTOM = 100

stop_event = threading.Event()
last_successful_plate = None
current_frame = None


def setup_gpio():
    """Initialize GPIO settings"""
    try:
        GPIO.cleanup()
    except:
        pass
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(RED_LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(GREEN_LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IR_SENSOR_PIN, GPIO.IN)
    
    print("GPIO setup completed")

def control_lights(red_state, green_state):
    """Control LED lights"""
    GPIO.output(RED_LIGHT_PIN, GPIO.HIGH if red_state else GPIO.LOW)
    GPIO.output(GREEN_LIGHT_PIN, GPIO.HIGH if green_state else GPIO.LOW)

def turn_red():
    """Turn on red light, turn off green light"""
    control_lights(True, False)

def turn_green():
    """Turn on green light, turn off red light"""
    control_lights(False, True)

def cleanup_gpio():
    """ทำความสะอาด GPIO เมื่อจบโปรแกรม"""
    try:
        GPIO.setmode(GPIO.BOARD)  # ต้องตั้ง mode ก่อน cleanup
        GPIO.cleanup()  # ทำความสะอาด GPIO ทั้งหมด
    except Exception as e:
        print(f"GPIO cleanup error: {str(e)}")

# เพิ่มฟังก์ชันใหม่สำหรับจัดการเวลา
def get_formatted_time():
    """คืนค่าเวลาปัจจุบันในรูปแบบ HH:MM:SS"""
    return datetime.datetime.now().strftime("%H:%M:%S")


def send_exit_request(license_plate: str, ocr_results_queue):
    """Send exit request to server and handle the response"""
    try:
        data = {"licensePlate": license_plate}
        response = requests.post(API_URL+"/parking/exit", json=data)
        response_data = response.json()

        if response.status_code == 200:
            status_message = "อนุญาตให้ออกได้"
            print(f"✅ Exit request successful for: {license_plate}")
            ocr_results_queue.put({
                'status': status_message,
                'payment': "ชำระแล้ว"
            })
            turn_green()
            return True
        elif response.status_code == 400:
            if 'paymentDetails' in response_data:
                amount_due = response_data['paymentDetails']['amountDue']
                status_message = "ไม่สามารถออกได้"
                payment_message = f"ต้องชำระ {amount_due} บาท"
                
                ocr_results_queue.put({
                    'status': status_message,
                    'payment': payment_message
                })
            else:
                status_message = "เกิดข้อผิดพลาด"
                ocr_results_queue.put({
                    'status': status_message,
                    'payment': "ไม่สามารถตรวจสอบได้"
                })
            turn_red()
            return False
        else:
            print(f"❌ Exit request failed: {response_data}")
            status_message = "เกิดข้อผิดพลาด"
            ocr_results_queue.put({
                'status': status_message,
                'payment': "ไม่สามารถตรวจสอบได้"
            })
            turn_red()
            return False

    except Exception as e:
        print(f"❌ Error in exit request: {str(e)}")
        status_message = "เกิดข้อผิดพลาด"
        ocr_results_queue.put({
            'status': status_message,
            'payment': "ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์"
        })
        turn_red()
        return False

def ir_light_controller():
    """Control lights based on IR sensor status"""
    last_state = None
    
    # Start with red light
    control_lights(True, False)
    print("System initialized - Red light on")
    
    while not stop_event.is_set():
        try:
            current_ir_state = GPIO.input(IR_SENSOR_PIN)
            
            if current_ir_state != last_state:
                if current_ir_state == GPIO.HIGH:  # Object detected
                    print("IR Sensor: Object detected - Waiting 2 seconds")
                    time.sleep(2)
                    print("2 seconds passed - Red light on")
                    control_lights(True, False)
                last_state = current_ir_state
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"IR controller error: {str(e)}")
            time.sleep(1)

def update_gui(ocr_results_queue, plate_label, time_label, status_label, payment_label):
    while True:
        try:
            result = ocr_results_queue.get_nowait()
            
            if isinstance(result, dict):
                if 'plate' in result:
                    plate_text = result.get('plate', '')
                    time_text = result.get('time', '')
                    plate_label.config(text=f"ทะเบียน: {plate_text}")
                    time_label.config(text=f"เวลา: {time_text}")
                
                if 'status' in result:
                    status_text = result.get('status', '')
                    payment_text = result.get('payment', '')
                    status_label.config(text=status_text)
                    payment_label.config(text=payment_text)

        except queue.Empty:
            pass
        except Exception as e:
            print(f"GUI update error: {str(e)}")
        
        time.sleep(0.1)

def add_padding_to_bbox(xmin, ymin, xmax, ymax, frame_height, frame_width, padding_percent=10):

    # คำนวณขนาดของ padding
    width = xmax - xmin
    height = ymax - ymin
    padding_x = int(width * padding_percent / 100)
    padding_y = int(height * padding_percent / 100)
    
    # เพิ่ม padding และตรวจสอบขอบเขต
    new_xmin = max(0, xmin - padding_x)
    new_ymin = max(0, ymin - padding_y)
    new_xmax = min(frame_width, xmax + padding_x)
    new_ymax = min(frame_height, ymax + padding_y)
    
    return new_xmin, new_ymin, new_xmax, new_ymax


def capture_frame(cap, frame_queue):
    # ตั้งค่า buffer size ของ OpenCV
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # ตั้งค่า FPS ของกล้อง
    cap.set(cv2.CAP_PROP_FPS, 1)  # ปรับเป็น 30 FPS
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            stop_event.set()
            break
        
        # ล้าง frame เก่าออกจาก queue
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
        
        frame_queue.put(frame)

def process_frame(frame_queue, model, last_ocr_time, trigger_zone, model_path, font_path, ocr_results_queue):
    global last_successful_plate
    
    object_in_zone = False
    last_successful_plate = None
    waiting_for_exit = False
    trigger_start_time = 0
    last_ocr_time = time.time()
    min_confidence_threshold = 0.6
    
    while not stop_event.is_set():
        try:
            original_frame = frame_queue.get(timeout=1)
            
            # Resize frame for YOLO processing
            yolo_frame = cv2.resize(original_frame.copy(), (YOLO_WIDTH, YOLO_HEIGHT))
            
            # Calculate scaling factors
            height, width = original_frame.shape[:2]
            scale_x = width / YOLO_WIDTH
            scale_y = height / YOLO_HEIGHT
            
            # Calculate actual trigger zone coordinates
            actual_trigger_zone = (
                (int(trigger_zone[0][0] * scale_x), int(trigger_zone[0][1] * scale_y)),
                (int(trigger_zone[1][0] * scale_x), int(trigger_zone[1][1] * scale_y))
            )
            
            # Draw initial trigger zone
            cv2.rectangle(original_frame, actual_trigger_zone[0], actual_trigger_zone[1], (255, 0, 0), 2)

            # Run YOLO detection
            results = model.predict(yolo_frame, verbose=False)[0]
            
            # Process detection results
            plate_in_zone = False
            if len(results.boxes) > 0:
                for box in results.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)

                    if conf < model.conf:
                        continue
                    
                    # Calculate real coordinates
                    xmin, ymin, xmax, ymax = coords
                    real_xmin = int(xmin * scale_x)
                    real_ymin = int(ymin * scale_y)
                    real_xmax = int(xmax * scale_x)
                    real_ymax = int(ymax * scale_y)

                    if cls in [0, 1]:  # car or license plate
                        # Draw bounding box
                        cv2.rectangle(original_frame, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
                        label = f"{['car', 'licenseplate'][cls]}: {conf:.2%}"
                        cv2.putText(original_frame, label, (real_xmin, real_ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(scale_x, scale_y), (0, 255, 0), 2)

                        # Check if license plate is in trigger zone
                        in_trigger_zone = (
                            real_xmin >= actual_trigger_zone[0][0] and
                            real_xmax <= actual_trigger_zone[1][0] and
                            real_ymin >= actual_trigger_zone[0][1] and
                            real_ymax <= actual_trigger_zone[1][1]
                        )
                        
                        if cls == 1 and in_trigger_zone:  # license plate in zone
                            plate_in_zone = True
                            if not waiting_for_exit:
                                if not object_in_zone:
                                    object_in_zone = True
                                    trigger_start_time = time.time()
                                elif time.time() - trigger_start_time >= 1:
                                    current_time = time.time()
                                    if current_time - last_ocr_time >= 1:  # 1 second delay between OCR attempts
                                        # Perform OCR
                                        
                                        # เพิ่ม padding ให้กับ bounding box
                                        padded_xmin, padded_ymin, padded_xmax, padded_ymax = add_padding_to_bbox(
                                            real_xmin, real_ymin, real_xmax, real_ymax,
                                            original_frame.shape[0], original_frame.shape[1],
                                            padding_percent=10 # ปรับค่านี้ตามความเหมาะสม
                                        )
                                        
                                        # ตัดภาพด้วย padding
                                        plate_img = original_frame[padded_ymin:padded_ymax, padded_xmin:padded_xmax]
                                        

                                        if plate_img.size > 0:  # Check if plate image is valid
                                            try:

                                                # preprocessing
                                                plate_img = cv2.resize(plate_img, (OCR_SIZE, OCR_SIZE))
                                                plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                                                plate_img_eq = cv2.equalizeHist(plate_img_gray)
                                                plate_img = cv2.cvtColor(plate_img_eq, cv2.COLOR_GRAY2BGR)
                                                
                                                # transform
                                                transformed_img = process_auto_transform(plate_img)
                                                
                                                # split
                                                top_img, _ = process_split_image(transformed_img)
                                                results, confidence = process_read_license(top_img, model_path, font_path)
                                                text = ''.join([char for char, _, _ in results])
                                                
                                                if text and confidence >= min_confidence_threshold:
                                                    current_time = get_formatted_time()
                                                    
                                                    if text != last_successful_plate:
                                                        print(f"\nDetected new license plate: {text}")
                                                        print(f"Confidence: {confidence:.2%}")
                                                        print(f"Time: {current_time}")
                                                        
                                                        # Update GUI with plate info
                                                        ocr_results_queue.put({
                                                            'plate': text,
                                                            'time': current_time,
                                                            'confidence': confidence
                                                        })
                                                        
                                                        # Send exit request automatically
                                                        send_exit_request(text, ocr_results_queue)
                                                        
                                                        last_successful_plate = text
                                                        waiting_for_exit = True
                                                    
                                                    last_ocr_time = time.time()
                                            except Exception as e:
                                                print(f"OCR Error: {str(e)}")
                                                continue

            # Update trigger zone color based on detection
            trigger_zone_color = (0, 0, 255) if plate_in_zone else (255, 0, 0)
            cv2.rectangle(original_frame, actual_trigger_zone[0], actual_trigger_zone[1], trigger_zone_color, 2)

            # Reset waiting state if no plate in zone
            if not plate_in_zone and waiting_for_exit:
                waiting_for_exit = False
                print(f"License plate {last_successful_plate} has left the trigger zone")
                last_successful_plate = None

            # Display the frame
            display_frame = cv2.resize(original_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow("CCTV Stream", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopping...")
                stop_event.set()
                break

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {str(e)}")
            continue

def main():
    try:
        model_path = "readLicense/EfficientNet_model3.pth"
        yolo_model_path = "detectCar/yolov8/best_v8_2.pt"
        font_path = "AnantasonReno-SemiExpanded-Italic.otf"
        capture_dir = "captured_plates"
        os.makedirs(capture_dir, exist_ok=True)

        # Initialize GPIO
        setup_gpio()

        # Setup ML model
        model = YOLO(yolo_model_path)
        model.conf = 0.5
        model.max_det = 1

        frame_queue = queue.Queue(maxsize=1)
        ocr_results_queue = queue.Queue()

        # Create GUI
        root = tk.Tk()
        root.title("ระบบตรวจสอบการออก")
        
        # Create labels with larger font
        plate_label = ttk.Label(root, text="ทะเบียน: -", font=("Helvetica", 80))
        plate_label.pack(pady=20)
        
        time_label = ttk.Label(root, text="เวลา: -", font=("Helvetica", 80))
        time_label.pack(pady=20)
        
        status_label = ttk.Label(root, text="สถานะ: -", font=("Helvetica", 80))
        status_label.pack(pady=20)
        
        payment_label = ttk.Label(root, text="การชำระเงิน: -", font=("Helvetica", 80))
        payment_label.pack(pady=20)

        # Start threads
        gui_thread = threading.Thread(target=update_gui, 
                                   args=(ocr_results_queue, plate_label, time_label, 
                                         status_label, payment_label), 
                                   daemon=True)
        ir_thread = threading.Thread(target=ir_light_controller, daemon=True)
        ir_thread.start()
        
        gui_thread.start()
        

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open camera")
            
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Calculate trigger zone
        trigger_zone = (
            (int(YOLO_WIDTH * ZONE_LEFT/100), int(YOLO_HEIGHT * ZONE_TOP/100)),
            (int(YOLO_WIDTH * ZONE_RIGHT/100), int(YOLO_HEIGHT * ZONE_BOTTOM/100))
        )
        
        last_ocr_time = 0
        stop_event.clear()

        # Start image processing threads
        capture_thread = threading.Thread(target=capture_frame, args=(cap, frame_queue), daemon=True)
        process_thread = threading.Thread(target=process_frame, 
                                       args=(frame_queue, model, last_ocr_time, trigger_zone, 
                                             model_path, font_path, ocr_results_queue), 
                                       daemon=True)
        
        capture_thread.start()
        process_thread.start()

        root.mainloop()

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        stop_event.set()
    finally:
        stop_event.set()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        cleanup_gpio()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by Ctrl+C")
    finally:
        stop_event.set()
        cleanup_gpio()