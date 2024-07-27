import cv2
import dlib 

# Initialize dlib's face detector (HOG-based) and create facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Capture video from laptop camera
cap_laptop = cv2.VideoCapture(0)

# Capture video from phone camera (use IP Webcam app on phone for this example)
cap_phone = cv2.VideoCapture("http://192.168.1.2:8080/video")

def detect_faces_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        # Draw circles around eyes
        for n in range(36, 48):  # Eye landmarks range
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    return frame

while True:
    ret1, frame1 = cap_laptop.read()
    ret2, frame2 = cap_phone.read()

    if ret1 and ret2:
        frame1 = detect_faces_and_eyes(frame1)
        frame2 = detect_faces_and_eyes(frame2)

        cv2.imshow('Laptop Camera', frame1)
        cv2.imshow('Phone Camera', frame2)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to capture image from one of the cameras")

cap_laptop.release()
cap_phone.release()
cv2.destroyAllWindows()
