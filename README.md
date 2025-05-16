# DRIVER-DROWSINESS-DETECTION-SYSTEM
Objectives -To build a system that monitors a driver’s alertness using a webcam. To detect closed eyes or signs of sleepiness. To give a warning sound and on-screen alert when drowsiness is detected. To provide a simple and user-friendly interface that anyone can use.

# DRIVER DROWSINESS DETECTION SYSTEM

import cv2
import winsound
import tkinter as tk
from tkinter import messagebox

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def start_detection():
    # Try to open the camera
    cap = cv2.VideoCapture(0)  # Try camera index 0
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        messagebox.showerror("Error", "Could not access camera. Please check if it is in use or connected properly.")
        return

    drowsy_count = 0  # Counter for drowsiness
    THRESHOLD_FRAMES = 20  # Frames before alarm triggers

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            drowsy_count += 1  # Increment drowsy count if no faces detected
        else:
            # Loop through each face detected
            for (x, y, w, h) in faces:
                # Extract the region of interest (ROI) where the face is detected
                roi_gray = gray[y:y + h, x:x + w]

                # Detect eyes within the face region
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) == 0:
                    drowsy_count += 1  # Increment drowsy count if no eyes detected
                else:
                    drowsy_count = 0  # Reset drowsy count if eyes are detected

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Draw rectangles around the eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # If drowsy count exceeds the threshold, trigger the alarm
        if drowsy_count >= THRESHOLD_FRAMES:
            cv2.putText(frame, "WAKE UP!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            winsound.Beep(1000, 500)

        cv2.imshow("Driver Alertness", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_info():
    messagebox.showinfo("Instructions", "Press 'q' to stop the detection.")


# Create Tkinter GUI
root = tk.Tk()
root.title("Driver Alertness System")

label = tk.Label(root, text="Driver Alertness Detection", font=("Arial", 14))
label.pack(pady=10)

start_button = tk.Button(root, text="Start Detection", command=start_detection, font=("Arial", 12))
start_button.pack(pady=5)

info_button = tk.Button(root, text="Instructions", command=show_info, font=("Arial", 12))
info_button.pack(pady=5)

exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
exit_button.pack(pady=10)

root.mainloop()
