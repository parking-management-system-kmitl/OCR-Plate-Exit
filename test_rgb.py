import RPi.GPIO as GPIO
import time

# กำหนดขาที่ใช้ควบคุม Relay
in1 = 16  # GPIO 16
in2 = 18  # GPIO 18

# กำหนดขาที่ใช้รับสัญญาณ IR
IR_PIN = 22  # GPIO 22 (Active Low)

# ตั้งค่า GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(IR_PIN, GPIO.IN)

GPIO.output(in1, False)
GPIO.output(in2, False)

def set_led(red_on, green_on):
    GPIO.output(in1, GPIO.HIGH if red_on else GPIO.LOW)  # ไฟแดง
    GPIO.output(in2, GPIO.HIGH if green_on else GPIO.LOW)  # ไฟเขียว

# ฟังก์ชันอ่านสัญญาณ IR (สำหรับ Active Low)
def read_ir_signal():
    signal = GPIO.input(IR_PIN)
    return signal

try:
    while True:
        ir_signal = read_ir_signal()

        if ir_signal == GPIO.LOW:  # เมื่อมีสัญญาณ IR (IR Active Low)
            print("ได้รับสัญญาณ IR")
            # เปิดไฟสีแดง
            set_led(red_on=True, green_on=False)
        else:  # เมื่อไม่มีสัญญาณ IR
            print("ไม่ได้รับสัญญาณ IR")
            # เปิดไฟสีเขียว
            set_led(red_on=False, green_on=True)

        time.sleep(1)  # อ่านสัญญาณทุก 1 วินาที

except KeyboardInterrupt:
    print("\nโปรแกรมหยุดทำงาน")
    GPIO.cleanup()
