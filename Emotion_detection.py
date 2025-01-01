import cv2
import streamlit  as st
face_c=cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
import numpy as np
from tensorflow.python.keras.models import  load_model
import tensorflow
import base64
model = tensorflow.keras.models.load_model(r"model_compressed.h5")
st.title(":rainbow[Emotion-Detection]")
button=st.button("Click For Camera")
Stop=st.button("Click for End")
path=(r"emotion-detection-concept-vector-25603383.jpg")
with open(path,"rb") as file:
   Backimage=base64.b64encode(file.read()).decode()
page_element=f"""
<style>
[data-testid="stAppViewContainer"]
{{
  background-image:url("data:image;base64,{Backimage}");
  background-size:1500px 1000px;
  background-position:center;
  background-repeat:no-repeat;

 
}}
<style>
"""
st.markdown(page_element,unsafe_allow_html=True)
frame_window=st.image([])
<DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Capture</title>
</head>
<body>
    <h1>Camera Stream</h1>
    <video id="videoElement" width="640" height="480" autoplay></video>
    <script>
        const video = document.getElementById('videoElement');

        // Access the camera using getUserMedia
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function (stream) {
                    video.srcObject = stream;
                })
                .catch(function (error) {
                    console.error("Error accessing the camera: ", error);
                });
        } else {
            console.log("getUserMedia is not supported in this browser.");
        }
    </script>
</body>
</html>

camera=cv2.VideoCapture(0)
if button:
   try:
    while True:
      _,frame=camera.read()
      frame=cv2.flip(frame,1)
      gray=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
      faces=face_c.detectMultiScale(gray,1.7)
      for x,y,w,h in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
         roi_gray=gray[y:y+w,x:x+h]
         roi_gray=cv2.resize(roi_gray,(150,150))
         Image_fin=np.expand_dims(roi_gray,axis=0)
         prediction=model.predict(Image_fin)
         class_name=['Angry','Happy','Neutral','Sad','Surprise']
         pred=class_name[np.argmax(prediction)]
         cv2.putText(frame,pred,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame_window.image(frame_rgb)
   except:
      print("Error Occur")
if Stop:
    camera.release()
         
