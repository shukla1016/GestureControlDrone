import cv2
import mediapipe as mp
import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

def count_fingers(landmarks, hand_label):
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]

    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP]

    count = 0
    for tip, pip in zip(tips, pips):
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            count += 1

    if hand_label == "Right":
        if landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            count += 1
    else:
        if landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            count += 1

    return count

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        total_fingers = 0
        label = "None"

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                count = count_fingers(hand_landmarks, hand_label)
                total_fingers += count

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if total_fingers == 0:
                label = "Stop"
            elif total_fingers == 1:
                label = "Up"
                client.moveByVelocityAsync(0, 0, -1, duration=0.1)
            elif total_fingers == 2:
                label = "Down"
                client.moveByVelocityAsync(0, 0, 1, duration=0.1)
            elif total_fingers == 3:
                label = "Left"
                client.moveByVelocityAsync(0, -1, 0, duration=0.1)
            elif total_fingers == 4:
                label = "Right"
                client.moveByVelocityAsync(0, 1, 0, duration=0.1)
            elif total_fingers == 5:
                label = "Forward"
                client.moveByVelocityAsync(1, 0, 0, duration=0.1)
            elif total_fingers == 6:
                label = "Backward"
                client.moveByVelocityAsync(-1, 0, 0, duration=0.1)
            elif total_fingers == 7:
                label = "Rotate Left"
                client.rotateByYawRateAsync(-30, duration=0.5)
            elif total_fingers == 8:
                label = "Rotate Right"
                client.rotateByYawRateAsync(30, duration=0.5)
            elif total_fingers == 9:
                label = "Land"
                client.landAsync()
            elif total_fingers == 10:
                label = "Takeoff"
                client.takeoffAsync()

        cv2.putText(frame, f'Fingers: {total_fingers} | Action: {label}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Finger Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

client.armDisarm(False)
client.enableApiControl(False)