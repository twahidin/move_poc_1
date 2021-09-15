import cv2
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)











# import streamlit as st #success this version does not support linear regression
# # To make things easier later, we're also importing numpy and pandas for
# # working with sample data. Need to upload this version to an ipad but we need to cut down on the face landmarks to save processing power
# import numpy as np
# import pandas as pd
# import threading
# from typing import Union
# import cv2
# import s3fs
# import av
# import mediapipe as mp
# import csv
# import os
# # import boto3
# # from boto.s3.connection import S3Connection
# from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
# import time
# from dotenv import load_dotenv
# from PIL import Image
# #from sklearn.model_selection import train_test_split
# #from sklearn.pipeline import make_pipeline 
# #from sklearn.preprocessing import StandardScaler 
# #from sklearn.linear_model import LogisticRegression, RidgeClassifier
# #from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score # Accuracy metrics 
# import pickle

# #variable declaration
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# #model_name = './data/finalise_model.sav' #sports name can be changed with s_option
# model_name = './data/finalise_model.sav' #sports name can be changed with s_option
# # cords = 'crds.csv'
# # model_name = 'model.sav'


# a_model = ''
# X = ''
# img_counter = 5

# #Functions to switched on camera
# #Function to change captured camera values/resolution
# cx = 720
# cy = 480

# # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:#high confidence - too high, you may not be able to track
# # Before pusing to Heroku this segment below will be changed to CONFIG file in Heroku

# # load_dotenv('.env')

# # aws3 = boto3.resource(
# #     service_name='s3',
# #     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
# #     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
# # )

# #code below for Heroku test



# # aws3 = boto3.resource(
# #     service_name='s3',
# #     aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
# #     aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
# # )

# #fs = s3fs.S3FileSystem(anon=False)

# # @st.cache(ttl=600)
# # def read_file(filename):
# #     with fs.open(filename) as f:
# #         return f

# #file = read_file("movev1/finalise_model.sav")



# #fs.download("movev1/finalise_model.csv", model_name)
# #cords = fs.download_file("movev1/coords.csv")

# #aws3.Bucket('movev1').download_file(Key='finalise_model.sav', Filename=model_name) #check if exist
# #aws3.Bucket('movev1').download_file(Key='coords.csv', Filename=cords) #check if exist

# #function declaration
# def calculate_angle(a,b,c): #calculation of the angles of the joints
#     a = np.array(a) # First
#     b = np.array(b) # Mid
#     c = np.array(c) # End
    
#     radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
#     angle = np.abs(radians*180.0/np.pi)
    
#     if angle >180.0:
#         angle = 360-angle
        
#     return angle    

# #App title 

# st.title('MOVE APP Prototype V1')

# #Side bar options

# df = pd.DataFrame({
#   'first column': ['Default', 'Fast', 'Medium', 'Slow'],
  

#   #choose a sport column
# })


# class_df = pd.DataFrame({
#     'sec column': ['Sec 1', 'Sec 2', 'Sec 3', 'Sec 4', 'Sec 5'],
#     'third column': ['1', '2', '3', '4', '5'],
#     })


# #sports option in side bar
# s_option = st.sidebar.selectbox( 
#     'Select anlysis speed :',
#     df['first column'])

# with st.sidebar.form("User Credentials"):
#     name = st.text_input('Enter your name:')
#     age = st.slider('How old are you?', min_value = 12, max_value = 17, value = 15, step=1)
#     level = st.selectbox('Select your level:',class_df['sec column'])
#     class_no = st.selectbox('Select your class:',class_df['third column'])
#     submit_button = st.form_submit_button()



# #model = pickle.load(open(model_name, 'rb'))

# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# def main():
#     class VideoTransformer(VideoProcessorBase):
#         # frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
#         # in_image: Union[np.ndarray, None]

#         def __init__(self) -> None:
#             # self.frame_lock = threading.Lock()
#             self.in_image = None
#             self.row = None

#         def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#             in_image = frame.to_ndarray(format="bgr24")
#             results = pose.process(in_image)
#             mp_drawing.draw_landmarks(in_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
#                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
#                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#                                      )
#             try:
#                 pose_row = results.pose_landmarks
#                 if pose_row is not None:
#                     row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_row.landmark]).flatten())
#                     self.row = row
#                 pass
#             except Exception as e:
#                 raise e
#             else:
#                 pass

#             # with self.frame_lock:
#             self.in_image = in_image
#             return av.VideoFrame.from_ndarray(in_image, format="bgr24")
#     ctx = webrtc_streamer(key="snapshot", video_processor_factory=VideoTransformer)
   
#     if st.button("Analyse"):
#         img_counter = 5
#         seconds = 5
#         row_list =[]
#         img_list = []
#         class_list = []
#         prob_list = []
#         with st.empty():
#             while seconds:
#                 st.header(f"Please get ready!, capturing in ⏳ {seconds} ")
#                 seconds -= 1
#                 time.sleep(1)
#             st.header("Move !!!")    
#         while img_counter > 0:
#             if ctx.video_processor:           
#                 with ctx.video_processor.frame_lock:
#                     in_image = ctx.video_processor.in_image
#                     row = ctx.video_processor.row
#                     img_list.append(in_image)
#                     row_list.append(row)
#                     #st.write(str(img_counter))
#                     if in_image is None: 
#                         st.warning("No frames available yet.")
#             img_counter -= 1
#             if s_option == 'Slow':
#                 time.sleep(0.5)
#             elif s_option == 'Medium':
#                 time.sleep(0.2)
#             elif s_option == 'Fast':
#                 time.sleep(0.1)
#             else:
#                 time.sleep(0.3)
#         #choose the sport if none, just take a series of images only 

#         try:
            
#             for i in range(len(img_list)):
#                 X = pd.DataFrame([row_list[i]])
#                 #st.text(X)
#                 body_language_class = model.predict(X)[0]
#                 body_language_prob = model.predict_proba(X)[0]        

#                 class_list.append(body_language_class)
#                 prob_list.append(str(round(body_language_prob[np.argmax(body_language_prob)],2)))

#                 cv2.rectangle(img_list[i], (0,0), (250, 60), (245, 117, 16), -1)

                
#                 # Display Class
#                 cv2.putText(img_list[i], 'CLASS'
#                             , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                 cv2.putText(img_list[i], body_language_class.split(' ')[0]
#                             , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
#                 # Display Probability
#                 cv2.putText(img_list[i], 'PROB'
#                             , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#                 in_image = cv2.putText(img_list[i], str(round(body_language_prob[np.argmax(body_language_prob)],2))
#                             , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             pass
#         except Exception as e:
#             raise e
#         else:
#             pass


#         #after the while loop
#         with st.container():
#             #st.write("This is inside a container "  + str(img_list) )
#             if img_list:
#                 col1, col2, col3, col4, col5 = st.columns(5)
#                 with col1:
#                     st.image(img_list[0], channels="BGR", use_column_width=True)
#                     st.text(class_list[0])
#                     st.text(prob_list[0])
#                 with col2:
#                     st.image(img_list[1], channels="BGR", use_column_width=True)
#                     st.text(class_list[1])
#                     st.text(prob_list[1]) 
#                 with col3:
#                     st.image(img_list[2], channels="BGR", use_column_width=True)
#                     st.text(class_list[2])
#                     st.text(prob_list[2])
#                 with col4:
#                     st.image(img_list[3], channels="BGR", use_column_width=True)
#                     st.text(class_list[3])
#                     st.text(prob_list[3]) 
#                 with col5:
#                     st.image(img_list[4], channels="BGR", use_column_width=True)
#                     st.text(class_list[4])
#                     st.text(prob_list[4])      



# if __name__ == "__main__":
#     main()
    






