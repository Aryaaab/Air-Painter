import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Index finger tip landmark number in Mediapipe
INDEX_FINGER_TIP = 8  

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get height & width of frame
            h, w, c = frame.shape

            # Extract index fingertip coordinates
            x = int(hand_landmarks.landmark[INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[INDEX_FINGER_TIP].y * h)

            # Draw a circle at fingertip
            cv2.circle(frame, (x, y), 15, (0, 0, 255), cv2.FILLED)

            # Also draw the full skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Index Finger Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
