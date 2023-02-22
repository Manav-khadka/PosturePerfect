import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
import time
import pandas as pd
import pickle
import os
import numpy as np

# Load the model

model_path = os.path.join(os.getcwd(), 'myapp',  'body_language.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

# Define the minimum detection and tracking confidences
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Import the holistic model from mediapipe
import mediapipe as mp
mp_holistic = mp.solutions.holistic

# Define a generator function that yields JPEG frames from the camera
def gen(camera):
    #load the model
    with mp_holistic.Holistic(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as holistic:
        while True:
            success, image = camera.read()
            if not success:
                break

            # Recolor the image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False   

            # Process the image with the holistic model
            results = holistic.process(image)

            # Recolor the image from RGB back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image.flags.writeable = True   
            

            # Draw the pose landmarks on the image
            mp_drawing = mp.solutions.drawing_utils
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.FACEMESH_TESSELATION)
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
             # 1. Draw face landmarks
            
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
             # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
            
            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
            
            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
           
            
            
            try:
                
                pose = results.pose_landmarks.landmark
                
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Concate rows
                row = pose_row+face_row
                
                
    #             # Append class name 
    #             row.insert(0, class_name)
                
    #             # Export to CSV
    #             with open('coords.csv', mode='a', newline='') as f:
    #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 csv_writer.writerow(row) 

                # Make Detections
               
                X = pd.DataFrame([row])
                
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                # print(body_language_class, body_language_prob)
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            except:
                pass
            # Ex
            # Encode the image as a JPEG and yield it
            _, jpeg = cv2.imencode('.jpg', image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# Decorate the view function with gzip compression
@gzip.gzip_page
def stream_video(request):
    # Set the response headers to stream the JPEG frames as a multipart response
    response = StreamingHttpResponse(gen(cap), content_type='multipart/x-mixed-replace; boundary=frame')

    return response
