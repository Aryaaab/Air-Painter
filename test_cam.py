import cv2

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Webcam", frame)

    # wait for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

# release the camera and close window
cap.release()
cv2.destroyAllWindows()
