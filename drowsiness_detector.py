"""Driver Drowsiness Detection - simple version"""

import time
import cv2
import mediapipe as mp
import numpy as np

# eye landmark points (from mediapipe face mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25      # below this = eyes closed
CLOSED_SECONDS = 1.5      # how long eyes must stay closed to alert

face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_ear(landmarks, eye_points):
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


# open webcam
cap = cv2.VideoCapture(0)

eyes_closed_start = None

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from camera")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # convert landmarks to pixel coordinates
        landmarks = []
        for point in face.landmark:
            landmarks.append((point.x * w, point.y * h))

        left_ear = get_ear(landmarks, LEFT_EYE)
        right_ear = get_ear(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2

        cv2.putText(frame, f"EAR: {round(ear, 2)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if ear < EAR_THRESHOLD:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()

            closed_time = time.time() - eyes_closed_start

            if closed_time >= CLOSED_SECONDS:
                cv2.putText(frame, "DROWSINESS ALERT!", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("\a")  # beep sound
        else:
            eyes_closed_start = None

    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()