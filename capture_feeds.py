import cv2


cap_laptop = cv2.VideoCapture(0)

cap_phone = cv2.VideoCapture("http://192.168.1.26:8080/video")

while True:
    ret1, frame1 = cap_laptop.read()
    ret2, frame2 = cap_phone.read()
    
    if ret1 and ret2:
        
        cv2.imshow('Laptop Camera', frame1)
        cv2.imshow('Phone Camera', frame2)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_laptop.release()
cap_phone.release()
cv2.destroyAllWindows()
