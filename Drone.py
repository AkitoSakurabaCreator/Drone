#必要なパッケージのインポート
import cv2
import mediapipe as mp
import numpy as np
from djitellopy import Tello
import time
import threading
import tensorflow as tf
import queue
import asyncio

class message :
    frameImage: any = None
    position = 0

global switch
switch = True

net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt.txt", "./MobileNetSSD_deploy.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

colors = np.random.uniform(0, 255, size=(21, 3))

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0)


input_buffer = queue.LifoQueue()
input_buffer2 = queue.LifoQueue()


drone = Tello()
time.sleep(2.0)
print("Connecting......")
drone.connect()
print("BATTERY: ")
print(drone.get_battery())
time.sleep(1.0)
print("Loading......")
drone.streamon()
print("Takeoff......")
drone.takeoff()
cap = drone.get_video_capture()


def main():
        global m
        m = message()
        thread = threading.Thread(target=pose, args=(m,))
        thread.start()
        thread2 = threading.Thread(target=box, args=(m,))
        thread2.start()
        while(True):
            if cap.isOpened:
                success, image = cap.read()
                input_buffer.put(image)
                input_buffer2.put(image)
                cv2.imshow('Drone Viewer', image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                switch = False
                drone.streamoff()
                drone.land()
                print("Landing......")
                print("BATTERY: ")
                print(drone.get_battery())
                drone.end()
                cap.release()
                cv2.destroyAllWindows()
                break

def box(m):
    F_WEIGHT = 960
    F_HEIGHT = 720
    rifX = F_WEIGHT/2
    rifY = F_HEIGHT/2

    Kp_X = 0.1
    Ki_X = 0.0
    Kp_Y = 0.2
    Ki_Y = 0.0

    Tc = 0.05

    integral_X = 0
    error_X = 0
    integral_Y = 0
    error_Y = 0

    centroX_pre = rifX
    centroY_pre = rifY

    while (switch):
        
        if input_buffer2.empty():
            continue
        image = input_buffer2.get()
        input_buffer2.queue = []

        h,w,channels = image.shape
        blob = cv2.dnn.blobFromImage(image,
        0.007843, (180, 180), (0,0,0),True, crop=False)

        net.setInput(blob)
        detections = net.forward()
        
        idx = int(detections[0, 0, 0, 1])
        confidence = detections[0, 0, 0, 2]

        if idx == 15 and confidence > 0.5:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            centroX = (startX + endX)/2
            centroY = (2*startY + endY)/3

            centroX_pre = centroX
            centroY_pre = centroY

            error_X = -(rifX - centroX)
            error_Y = rifY - centroY

            y = startY - 15 if startY - 15 > 15 else startY + 15

            integral_X = integral_X + error_X*Tc
            uX = Kp_X*error_X + Ki_X*integral_X
            
            integral_Y = integral_Y + error_Y*Tc
            uY = Kp_Y*error_Y + Ki_Y*integral_Y

            drone.send_rc_control(0,m.position,round(uY),round(uX))
        else:
            centroX = centroX_pre
            centroY = centroY_pre

            error_X = -(rifX - centroX)
            error_Y = rifY - centroY

            integral_X = integral_X + error_X*Tc
            uX = Kp_X*error_X + Ki_X*integral_X
            
            integral_Y = integral_Y + error_Y*Tc
            uY = Kp_Y*error_Y + Ki_Y*integral_Y
            
            drone.send_rc_control(0,0,round(uY),round(uX))

def pose(m):
    while(switch):
        if input_buffer.empty():
            continue
        image = input_buffer.get()
        input_buffer.queue = []
        results = holistic.process(image)
            
        m.position = 0
        if results.pose_landmarks is not None:
            marks = results.pose_landmarks.landmark
            z_index1 = marks[11].z
            z_index2 = marks[12].z
            prepare = float(str(np.mean(z_index1 + z_index2))[1:5])

            if 0.15 >= prepare:
                m.position = 20
            elif 0.4 >= prepare:
                m.position = 15
            elif 0.5 >= prepare:
                m.position = 5
            elif 0.67 >= prepare:
                m.position = 0
            elif 0.68 >= prepare:
                m.position = -5
            elif 0.73 >= prepare:
                m.position = -15
            elif 0.74 <= prepare:
                m.position = -30

if __name__ == "__main__":
    main()
