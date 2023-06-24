import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical 
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from os.path import exists
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

new_model = keras.models.load_model('posizio_klasifikatzailea_0.h5')

column_names = ['0_x', '0_y', '0_z', '1_x', '1_y', '1_z', '2_x', '2_y', '2_z', '3_x', '3_y', '3_z', '4_x', '4_y', '4_z', 
                '5_x', '5_y', '5_z', '6_x', '6_y', '6_z', '7_x', '7_y', '7_z', '8_x', '8_y', '8_z', '9_x', '9_y', '9_z', 
                '10_x', '10_y', '10_z', '11_x', '11_y', '11_z', '12_x', '12_y', '12_z', '13_x', '13_y', '13_z', '14_x', 
                '14_y', '14_z', '15_x', '15_y', '15_z', '16_x', '16_y', '16_z', '17_x', '17_y', '17_z', '18_x', '18_y', 
                '18_z', '19_x', '19_y', '19_z', '20_x', '20_y', '20_z']

hand_mesh_names = {0: 'Eskurik gabe', 1:'Zabalik', 2:'Hatza igota', 3:'Heavy', 4:'Hatz potola', 5:'Itxita',6:'Itxita',7:'Erdiko hatza'}

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    landmark_csv = pd.DataFrame(columns=column_names)
    hand_landmark_result=0
    x_ardatza, y_ardatza, z_ardatza = 0, 0, 0
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      landmark_csv = pd.DataFrame(columns=column_names)
      for hand_landmarks in results.multi_hand_landmarks:
        
        points = []
        for n in enumerate(hand_landmarks.landmark):
          points.append(n[1].x)
          points.append(n[1].y)
          points.append(n[1].z)
        if len(points)==63:
          a_series = pd. Series(points, index = column_names)
          landmark_csv = landmark_csv.append(a_series, ignore_index=True)
          hand_landmark_result = int(np.argmax(new_model.predict(landmark_csv)))

          if hand_landmark_result == 2:
            x_hatz, y_hatz, z_hatz = points[8*3],points[8*3+1],points[8*3+2]
            x_ardatza, y_ardatza, z_ardatza = round(points[8*3],2),round(points[8*3+1],2),round(points[8*3+2],2)
            
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    image =cv2.flip(image, 1)
    cv2.putText(img=image, text=str(hand_mesh_names[hand_landmark_result]), org=(10, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(255, 0, 0),thickness=2)
    if hand_mesh_names[hand_landmark_result] == 'Hatza igota':
      cv2.putText(img=image, text= str(' x:' + str(1-x_ardatza) + ' y:' + str(1-y_ardatza) + ' z:' + str(-z_ardatza)), org=(10, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=1)
    cv2.imshow('MediaPipe Hands', image)
    k = cv2.waitKey(1)
    if  k%256 == 27 or k == ord('q'):
      break

cap.release()