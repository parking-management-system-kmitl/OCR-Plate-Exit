import cv2
import numpy as np

def process_split_image(image):
    """Process image splitting and return top and bottom images"""
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise Exception("ไม่สามารถอ่านรูปภาพได้")
    else:
        img = image

    # แบ่งภาพเป็นส่วนบนและล่าง
    height = img.shape[0]
    top_height = int(height * 2/3)
    
    top_image = img[0:top_height, :]
    bottom_image = img[top_height:, :]
    
    return top_image, bottom_image