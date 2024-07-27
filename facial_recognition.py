import cv2
import dlib

# Load pre-trained models for face detection and shape prediction
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Capture video from laptop camera
cap_laptop = cv2.VideoCapture(0)

while True:
    ret1, frame1 = cap_laptop.read()
    if ret1:
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            
            for n in range(36, 48):  # Eye landmarks range
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame1, (x, y), 2, (255, 0, 0), -1)
        
        cv2.imshow('Laptop Camera', frame1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_laptop.release()
cv2.destroyAllWindows()
