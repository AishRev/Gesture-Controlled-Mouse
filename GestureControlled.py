import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

def fingers_up(landmarks):
    fingers = []
    
    # Thumb (landmark 4 vs 2): Check if thumb tip is to the right (for right hand)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other 4 fingers: Tip landmark y < PIP joint landmark y => finger up
    tip_ids = [8, 12, 16, 20]
    for id in tip_ids:
        if landmarks[id].y < landmarks[id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers  # [Thumb, Index, Middle, Ring, Pinky]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_detector.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Get index finger tip position
            index_tip = landmarks[8]
            x = int(index_tip.x * frame.shape[1])
            y = int(index_tip.y * frame.shape[0])
            screen_x = np.interp(x, (0, frame.shape[1]), (0, screen_w))
            screen_y = np.interp(y, (0, frame.shape[0]), (0, screen_h))
            pyautogui.moveTo(screen_x, screen_y)

            # Detect finger states
            fingers = fingers_up(landmarks)

            # Left Click: All 5 fingers up
            if fingers == [1, 1, 1, 1, 1]:
                pyautogui.click()
                pyautogui.sleep(0.3)

            # Right Click: Only thumb up
            elif fingers == [1, 0, 0, 0, 0]:
                pyautogui.click(button='right')
                pyautogui.sleep(0.3)

    # Show frame
    cv2.imshow("Gesture Mouse", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
