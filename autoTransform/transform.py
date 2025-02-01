import cv2
import numpy as np

def detect_plate_auto(img):
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(blurred, 30, 200)
   
   kernel = np.ones((3,3), np.uint8)
   edges = cv2.dilate(edges, kernel, iterations=1)
   
   contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
   
   for contour in contours:
       peri = cv2.arcLength(contour, True)
       approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
       if len(approx) >= 4 and len(approx) <= 6:
           return approx[:4].reshape(4, 2)
           
   h, w = img.shape[:2]
   return np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])

def order_points(points):
   rect = np.zeros((4, 2), dtype="float32")
   s = points.sum(axis=1)
   rect[0] = points[np.argmin(s)]
   rect[2] = points[np.argmax(s)]

   diff = np.diff(points, axis=1)
   rect[1] = points[np.argmin(diff)]
   rect[3] = points[np.argmax(diff)]

   return rect

def perspective_transform_auto(img, detected_points):
   rect = order_points(detected_points)
   (tl, tr, br, bl) = rect

   widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
   widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
   maxWidth = max(int(widthA), int(widthB))

   heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
   heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
   maxHeight = max(int(heightA), int(heightB))

   dst = np.array([
       [0, 0],
       [maxWidth - 1, 0],
       [maxWidth - 1, maxHeight - 1],
       [0, maxHeight - 1]], dtype="float32")

   M = cv2.getPerspectiveTransform(rect, dst)
   warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
   return warped

def process_auto_transform(image):
   """Process image transformation and return the transformed image"""
   if isinstance(image, str):
       img = cv2.imread(image)
       if img is None:
           raise Exception("ไม่สามารถโหลดรูปภาพได้")
   else:
       img = image

   detected_points = detect_plate_auto(img)
   warped_image = perspective_transform_auto(img, detected_points)
   
   # Add resize operation
   resized_image = cv2.resize(warped_image, (224, 224))
   
   return resized_image
