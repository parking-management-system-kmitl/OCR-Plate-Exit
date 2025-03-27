import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def enhance_image(image):
    """เพิ่มการปรับแต่งภาพเพื่อช่วยในการตรวจจับตัวอักษร"""
    if len(image.shape) == 3:
        # แปลงเป็นภาพระดับสีเทา
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # ปรับความคมชัด
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # ลดสัญญาณรบกวน
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # ปรับความสว่างและคอนทราสต์
    alpha = 1.2  # คอนทราสต์ (1.0-3.0)
    beta = 10    # ความสว่าง (0-100)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return enhanced

def process_image_for_ocr(image):
    """ประมวลผลภาพเพื่อตรวจจับตัวอักษรป้ายทะเบียน"""
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # เก็บภาพต้นฉบับไว้
    original = image.copy()
    
    # เพิ่มการปรับแต่งภาพ
    enhanced = enhance_image(image)
    
    # ปรับให้เป็นไบนารี่
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # บันทึกภาพไบนารี่ก่อนประมวลผล
    binary_original = binary.copy()
    
    # ประมวลผลภาพเพื่อช่วยให้ตรวจจับตัวอักษรได้ดีขึ้น
    
    # 1. กำจัดสัญญาณรบกวนด้วย morphological operations
    kernel_noise = np.ones((3, 3), np.uint8)  # เพิ่มขนาด kernel
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    # ทำ opening ซ้ำเพื่อกำจัดจุดเล็กๆ
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    
    # 2. เชื่อมส่วนในแนวตั้ง
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_vertical)
    
    # 3. เชื่อมส่วนในแนวนอน
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
    
    # ค้นหา contours
    contours1, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1 + contours2
    
    # คำนวณความสูงเฉลี่ยของ contours
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avg_height = np.mean(heights) if heights else 0
    
    # เก็บตำแหน่งของทุก region ที่พบ
    all_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # ตรวจสอบเงื่อนไข:
        # 1. ขนาดพื้นที่ต้องมากพอ
        # 2. ความสูงต้องใกล้เคียงกับความสูงเฉลี่ย
        # 3. ความกว้างต้องมากกว่าค่าขั้นต่ำ
        if (w * h > 50 and  # ขนาดพื้นที่ขั้นต่ำ
            h > 0.4 * avg_height and  # ความสูงขั้นต่ำ
            w > 3):  # ความกว้างขั้นต่ำ
            all_regions.append((x, y, w, h))
    
    # กรองซ้ำซ้อนและรวม regions
    unique_regions = []
    for region in all_regions:
        x1, y1, w1, h1 = region
        is_duplicate = False
        
        for unique_region in unique_regions:
            x2, y2, w2, h2 = unique_region
            # คำนวณพื้นที่ซ้อนทับ
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            # พื้นที่ของกรอบทั้งสอง
            area1 = w1 * h1
            area2 = w2 * h2
            smaller_area = min(area1, area2)
            
            if overlap_area > 0.8 * smaller_area:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_regions.append(region)
    
    # รวม regions ที่อาจเป็นส่วนหนึ่งของตัวอักษรเดียวกัน
    merged_regions = []
    used_regions = set()
    
    for i, (x1, y1, w1, h1) in enumerate(unique_regions):
        if i in used_regions:
            continue
            
        current_region = [i]
        for j, (x2, y2, w2, h2) in enumerate(unique_regions[i+1:], i+1):
            if j in used_regions:
                continue
            
            x_overlap = (x1 < x2 + w2) and (x2 < x1 + w1)
            if y2 > y1:
                vertical_gap = y2 - (y1 + h1)
            else:
                vertical_gap = y1 - (y2 + h2)
            
            x_center1 = x1 + w1/2
            x_center2 = x2 + w2/2
            x_distance = abs(x_center1 - x_center2)
            
            if x_distance < max(w1, w2) * 0.7 and vertical_gap < max(h1, h2) * 0.7:
                current_region.append(j)
                used_regions.add(j)
        
        if current_region:
            # รวม regions
            min_x = min(unique_regions[i][0] for i in current_region)
            min_y = min(unique_regions[i][1] for i in current_region)
            max_x = max(unique_regions[i][0] + unique_regions[i][2] for i in current_region)
            max_y = max(unique_regions[i][1] + unique_regions[i][3] for i in current_region)
            
            merged_w = max_x - min_x
            merged_h = max_y - min_y
            
            # กรองด้วยอัตราส่วน
            ratio = merged_w/merged_h
            if 0.1 < ratio < 1.5 and merged_w * merged_h > 50:
                padding_x = int(merged_w * 0.07)
                padding_y = int(merged_h * 0.07)
                
                min_x = max(0, min_x - padding_x)
                min_y = max(0, min_y - padding_y)
                merged_w = min(image.shape[1] - min_x, merged_w + padding_x * 2)
                merged_h = min(image.shape[0] - min_y, merged_h + padding_y * 2)
                
                merged_regions.append((min_x, min_y, merged_w, merged_h))

    return image, merged_regions

def predict_image(image_region, model, transform, class_names, device):
    """ทำนายตัวอักษรเดี่ยว"""
    # แปลงเป็น PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB))
    
    # ประมวลผลด้วย transform และส่งเข้าโมเดล
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)
    
    predicted_class = class_names[predicted.item()]
    confidence = probability[0][predicted.item()].item()
    
    return predicted_class, confidence

def post_process_prediction(predictions):
    """ประมวลผลหลังการทำนาย - ตรวจสอบรูปแบบป้ายทะเบียนไทย"""
    if not predictions:
        return predictions
    
    # กฎการตรวจสอบรูปแบบป้ายทะเบียนไทยพื้นฐาน
    # ป้ายทะเบียนรถยนต์ส่วนบุคคลมักมีรูปแบบ: กก-1234
    
    processed = []
    
    # รวมตัวอักษรที่ทำนายได้
    chars = [p[0] for p in predictions]
    text = ''.join(chars)
    
    # กฎตรวจสอบเบื้องต้น
    for i, (char, conf, pos) in enumerate(predictions):
        # ถ้าอยู่ในตำแหน่งที่ควรเป็นตัวเลข (ตามรูปแบบป้ายทะเบียนไทย)
        if i >= 2 and char.isalpha() and conf < 0.8:
            # ตรวจสอบว่าตัวเลขอาจถูกทำนายผิดเป็นตัวอักษร
            # (อาจเพิ่มตรรกะเพิ่มเติมตรงนี้)
            continue
        
        # ถ้าตำแหน่งแรกหรือสองไม่ใช่ตัวอักษรไทย และความเชื่อมั่นต่ำ
        if i < 2 and not char.isalpha() and conf < 0.8:
            continue
            
        processed.append((char, conf, pos))
    
    return processed

def process_read_license(image, model_path, font_path=None):
    """ประมวลผลการอ่านป้ายทะเบียน"""
    # คลาสทั้งหมดที่โมเดลสามารถจำแนกได้
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                  'ก', 'ข', 'ค', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ฌ', 'ญ', 
                  'ฎ', 'ฐ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 
                  'บ', 'ป', 'ผ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 
                  'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ']
    
    # การเตรียมข้อมูล (transform) สำหรับโมเดล
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ตรวจสอบการใช้ GPU หรือ CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # โหลดโมเดล
    model = Model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # ประมวลผลภาพเพื่อตรวจจับตัวอักษร
    image, char_regions = process_image_for_ocr(image)
    
    # ตรวจสอบว่ามีตัวอักษรที่ตรวจพบหรือไม่
    if not char_regions:
        return [], 0.0
    
    # เตรียม preprocessing สำหรับแต่ละตัวอักษร
    def preprocess_char_image(char_img):
        """ปรับปรุงคุณภาพของภาพตัวอักษรก่อนส่งเข้าโมเดล"""
        # แปลงเป็นภาพระดับสีเทา
        if len(char_img.shape) == 3:
            gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = char_img
            
        # ปรับความคมชัดและคอนทราสต์
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # ขยายภาพและเพิ่ม padding (ช่วยให้โมเดลทำนายได้ดีขึ้น)
        border_size = max(5, int(min(char_img.shape[0], char_img.shape[1]) * 0.1))
        padded = cv2.copyMakeBorder(
            enhanced, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, 
            value=255
        )
        
        # แปลงกลับเป็นภาพสี RGB (เพื่อให้ใช้กับ PIL ได้)
        rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        return rgb
    
    # ทำนายตัวอักษรแต่ละตัว
    results = []
    for x, y, w, h in char_regions:
        char_image = image[y:y+h, x:x+w]
        
        # ข้ามพื้นที่ที่อาจไม่ใช่ตัวอักษร (เช่น สิ่งรบกวน)
        if char_image.size == 0 or w < 5 or h < 5:
            continue
        
        # ประมวลผลภาพตัวอักษรก่อนส่งเข้าโมเดล
        processed_char = preprocess_char_image(char_image)
        
        # ทำนายตัวอักษร
        predicted_class, confidence = predict_image(processed_char, model, transform, class_names, device)
        
        # เก็บผลลัพธ์ทั้งหมด
        results.append((predicted_class, confidence, (x, y, w, h)))

    # ตรวจสอบความสมบูรณ์ของผลลัพธ์
    if len(results) < 7 and len(char_regions) < 7:  # ป้ายทะเบียนไทยส่วนใหญ่มี 7 ตัวอักษร
        for x, y, w, h in char_regions:
            char_image = image[y:y+h, x:x+w]
            
            # ถ้าส่วนนี้ยังไม่ได้ถูกตรวจจับ (อาจเป็นเพราะความมั่นใจต่ำ)
            if not any(abs(pos[0] - x) < 5 for _, _, pos in results):
                processed_char = preprocess_char_image(char_image)
                predicted_class, confidence = predict_image(processed_char, model, transform, class_names, device)
                
                # ใช้ threshold ที่ต่ำลงเพื่อเพิ่มโอกาสในการตรวจจับตัวสุดท้าย
                if confidence >= 0.3:  # ลดจาก 0.5 เป็น 0.3
                    results.append((predicted_class, confidence, (x, y, w, h)))

    # จัดเรียงผลลัพธ์จากซ้ายไปขวา
    results = sorted(results, key=lambda r: r[2][0])

    # กรองเฉพาะผลลัพธ์ที่มีความแม่นยำมากกว่า threshold
    filtered_results = [r for r in results if r[1] >= 0.5]

    # คำนวณความแม่นยำรวม
    total_confidence = sum(conf for _, conf, _ in filtered_results) / len(filtered_results) if filtered_results else 0

    # ถ้าไม่มีผลลัพธ์ที่ผ่านเกณฑ์ แต่มีผลลัพธ์ทั้งหมด ให้ใช้ผลลัพธ์ที่ดีที่สุด
    if not filtered_results and results:
        best_results = []
        sorted_by_x = sorted(results, key=lambda r: r[2][0])
        current_pos = None
        best_for_pos = None

        for result in sorted_by_x:
            char, conf, (x, y, w, h) = result

            if current_pos is None or abs(x - current_pos) > w / 2:
                if best_for_pos:
                    best_results.append(best_for_pos)
                current_pos = x
                best_for_pos = result
            elif conf > best_for_pos[1]:  # หากเป็นตำแหน่งเดิมแต่ความมั่นใจมากกว่า
                best_for_pos = result

        if best_for_pos:
            best_results.append(best_for_pos)

        filtered_results = best_results

        total_confidence = sum(conf for _, conf, _ in filtered_results) / len(filtered_results) if filtered_results else 0

    return filtered_results, total_confidence