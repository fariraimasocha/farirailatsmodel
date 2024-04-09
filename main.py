import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

# Load your YOLO model
model = YOLO('Weights/best.pt')

# ESP32-CAM camera URL
url = 'http://192.168.199.13/cam-mid.jpg'

while True:
    img_resp = urllib.request.urlopen(url)
    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    # Perform object detection using YOLO model
    results = model(source=frame, show=True, conf=0.4, save=False)

    # Display the frame with object detection results
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
