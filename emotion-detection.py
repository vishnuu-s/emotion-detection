import cv2
from cv2 import data
from deepface import DeepFace
import pyautogui #(For Getting screen size)


class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(data.haarcascades + "haarcascade_frontalface_default.xml")
        self.model = DeepFace.build_model("Emotion")
        self.emotion_labels = [
            'angry', 
            'disgust', 
            'fear', 
            'happy', 
            'sad', 
            'surprise',
            'neutral'
        ]

    def detect_emotion(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)

            preds = self.model.predict(reshaped_face)[0]
            emotion_idx = preds.argmax()
            emotion = self.emotion_labels[emotion_idx]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return frame

class WebcamCapture:
    def __init__(self, source=0):
        self.video_capture = cv2.VideoCapture(source)

    def read_frame(self):
        ret, frame = self.video_capture.read()
        return frame

    def release(self):
        self.video_capture.release()
        
        

if __name__ == "__main__":
    emotion_detector = EmotionDetector()
    webcam = WebcamCapture()

    cv2.namedWindow('Real-time Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Real-time Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        frame = webcam.read_frame()
        frame_with_emotion = emotion_detector.detect_emotion(frame)

        # Get screen resolution and set window size accordingly
        screen_resolution = pyautogui.size()  # Set your screen resolution here
        cv2.resizeWindow('Real-time Emotion Detection', screen_resolution[0], screen_resolution[1])

        cv2.imshow('Real-time Emotion Detection', frame_with_emotion)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()