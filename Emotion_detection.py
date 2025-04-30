from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model_compressed.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

class EmotionDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (150, 150))
            roi = np.expand_dims(roi, axis=0)
            prediction = model.predict(roi)
            emotion = class_names[np.argmax(prediction)]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title(":rainbow[Emotion Detection via Webcam]")

webrtc_streamer(key="emotion-detection", 
                mode=WebRtcMode.SENDRECV, 
                video_processor_factory=EmotionDetector)
