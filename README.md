import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import queue
from collections import deque

class HandDetectionCallSystem:
    def __init__(self):
        # --- Hand tracking setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # --- TTS setup (with background thread queue) ---
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.voice_queue = queue.Queue()
        self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self.voice_thread.start()

        # --- Call state ---
        self.incoming_call = False
        self.call_status = ""
        self.call_announced = False
        self.status_announced = False

        # --- Gesture smoothing ---
        self.gesture_history = deque(maxlen=15)

    def _voice_loop(self):
        """Background thread: continuously checks for text to speak."""
        while True:
            text = self.voice_queue.get()  # Blocks until message available
            self.engine.say(text)
            self.engine.runAndWait()
            self.voice_queue.task_done()

    def speak(self, text):
        """Send text to speech queue."""
        # Prevent repeating the same message rapidly
        if not self.voice_queue.empty():
            try:
                last = self.voice_queue.queue[-1]
                if last == text:
                    return  # Skip duplicate
            except IndexError:
                pass
        self.voice_queue.put(text)

    def count_fingers(self, hand_landmarks, handedness):
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        fingers_up = 0

        if handedness == "Right":
            if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
                fingers_up += 1
        else:
            if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
                fingers_up += 1

        for i in range(1, 5):
            if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y:
                fingers_up += 1
        return fingers_up

    def detect_gesture(self, fingers_up):
        if fingers_up >= 4:
            return "PALM_OPEN"
        elif fingers_up <= 1:
            return "FIST"
        else:
            return "NONE"

    def get_stable_gesture(self, current_gesture):
        self.gesture_history.append(current_gesture)
        if len(self.gesture_history) == self.gesture_history.maxlen:
            if all(g == self.gesture_history[0] for g in self.gesture_history):
                return self.gesture_history[0]
        return "NONE"

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        fingers_up = 0
        gesture = "NONE"

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label
                fingers_up = self.count_fingers(hand_landmarks, hand_label)
                gesture = self.detect_gesture(fingers_up)
                stable_gesture = self.get_stable_gesture(gesture)

                if self.incoming_call and not self.call_status:
                    if stable_gesture == "PALM_OPEN":
                        self.call_status = "CALL ACCEPTED"
                        self.speak("Call accepted")
                        self.status_announced = True
                        self.incoming_call = False
                    elif stable_gesture == "FIST":
                        self.call_status = "CALL REJECTED"
                        self.speak("Call rejected")
                        self.status_announced = True
                        self.incoming_call = False

        # --- Display visuals ---
        cv2.putText(frame, f"Fingers: {fingers_up}", (w - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.incoming_call and not self.call_status:
            cv2.putText(frame, "INCOMING CALL", (w//2 - 150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, "Open palm to ACCEPT", (w//2 - 180, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Close fist to REJECT", (w//2 - 180, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if not self.call_announced:
                self.speak("Incoming call")
                self.call_announced = True

        if self.call_status:
            color = (0, 255, 0) if "ACCEPTED" in self.call_status else (0, 0, 255)
            cv2.putText(frame, self.call_status, (w//2 - 150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.putText(frame, "Press 'Y' for incoming call", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'R' to reset | Press 'Q' to quit", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Hand Detection Call System Started")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow("Hand Detection Call System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('y'):
                if not self.incoming_call and not self.call_status:
                    self.incoming_call = True
                    self.call_announced = False
                    self.call_status = ""
                    self.status_announced = False
                    self.gesture_history.clear()
                    print("Incoming call initiated")
            elif key == ord('r'):
                self.incoming_call = False
                self.call_status = ""
                self.call_announced = False
                self.status_announced = False
                self.gesture_history.clear()
                print("Reset call status")

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    system = HandDetectionCallSystem()
    system.run()
