import cv2
import numpy as np

# Define the paths to the model files
prototxt_path = r"C:\Users\jaimani choudhary\Desktop\shoe web\mkdir models\deploy.prototxt"
caffemodel_path = r"C:\Users\jaimani choudhary\Desktop\shoe web\mkdir models\mobilenet_iter_73000.caffemodel"

# Load pre-trained model and class labels
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Capture video from phone camera
cap_phone = cv2.VideoCapture("http://192.168.1.26:8080/video")

while True:
    ret2, frame2 = cap_phone.read()
    if ret2:
        (h, w) = frame2.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "phone":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame2, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"{CLASSES[idx]}: {confidence:.2f}"
                    cv2.putText(frame2, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Phone Camera', frame2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_phone.release()
cv2.destroyAllWindows()


