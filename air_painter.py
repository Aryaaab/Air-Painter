import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

INDEX_FINGER_TIP = 8  

canvas = None
brush_color = (255, 0, 0)  # Default Blue
brush_thickness = 10       # Default thickness

# Define color palette + clear
palette = [
    (10, 10, 60, 60, (0, 0, 255), "R"),       # Red
    (70, 10, 120, 60, (0, 255, 0), "G"),      # Green
    (130, 10, 180, 60, (255, 0, 0), "B"),     # Blue
    (190, 10, 240, 60, (0, 255, 255), "Y"),   # Yellow
    (250, 10, 300, 60, (255, 255, 255), "W"), # White
    (310, 10, 390, 60, (50, 50, 50), "CLR")   # Clear
]

# Brush size options
sizes = [
    (410, 10, 460, 60, 5, "S"),   # Small
    (470, 10, 520, 60, 10, "M"),  # Medium
    (530, 10, 590, 60, 20, "L"),  # Large
]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    fingertip = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[INDEX_FINGER_TIP].y * h)
            fingertip = (x, y)

            cv2.circle(frame, (x, y), 10, brush_color, cv2.FILLED)

            # Draw only if not clicking menu
            if y > 70:  
                cv2.circle(canvas, (x, y), brush_thickness, brush_color, -1)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw color palette
    for (x1, y1, x2, y2, color, label) in palette:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, label, (x1+10, y2-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,0,0) if sum(color) > 400 else (255,255,255), 2)

    # Draw brush sizes
    for (x1, y1, x2, y2, size, label) in sizes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200,200,200), -1)
        cv2.putText(frame, label, (x1+10, y2-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.circle(frame, (x1+25, y1+25), size, (0,0,0), -1)

    # Handle fingertip menu selection
    if fingertip:
        fx, fy = fingertip

        # Color/clear selection
        for (x1, y1, x2, y2, color, label) in palette:
            if x1 < fx < x2 and y1 < fy < y2:
                if label == "CLR":
                    canvas = np.zeros((h, w, 3), np.uint8)
                else:
                    brush_color = color

        # Brush size selection
        for (x1, y1, x2, y2, size, label) in sizes:
            if x1 < fx < x2 and y1 < fy < y2:
                brush_thickness = size

    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.putText(combined, "Tap Palette to Change Color | Tap S/M/L for Brush | S: Save | Q: Quit",
                (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Air Painter", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"✅ Saved drawing as {filename}")

cap.release()
cv2.destroyAllWindows()
