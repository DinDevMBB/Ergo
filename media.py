import mediapipe as mp
import cv2
import math as m
import numpy as np


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree

# Calcualte angle with 3 points
def find3pointangle(a,b,c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        three_angle = np.degrees(np.arccos(cosine_angle))
        return int(three_angle)

# calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# Initilize medipipe selfie segmentation class.
# mp_pose = mp.solutions.pose
# mp_holistic = mp.solutions.holistic

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose



# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

def detectPose(image_pose,MIN_CONFIEDENCE):
    # Setup the Pose function for images - independently for the images standalone processing.
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=MIN_CONFIEDENCE)
    # Colors.
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
       
    h,w = image_pose.shape[:2]
       
    resultant = pose.process(image_pose)
    landmarked_image = image_pose.copy()

    if resultant.pose_landmarks:    
        # mp_drawing.draw_landmarks(image=landmarked_image, landmark_list=resultant.pose_landmarks,
        #                           connections=mp_pose.POSE_CONNECTIONS,
        #                           landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
        #                                                                        thickness=1, circle_radius=1),
        #                           connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
        #                                                                        thickness=1, circle_radius=1))

        # left_wrist_landmark = (resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * w,
        #                         resultant.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y *h)
        
        
        lm = resultant.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        
        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        # Left Wrist.
        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)

        # Left Elbow.
        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)

        # Right ear.
        r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
        r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)
       

        # Right shoulder.
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        # Right wrist.
        r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
        r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

        # Right Elbow.
        r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
        r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

        # Right index.
        r_index_x = int(lm.landmark[lmPose.RIGHT_INDEX].x * w)
        r_index_y = int(lm.landmark[lmPose.RIGHT_INDEX].y * h)

        # Right Hip.
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)

        # Left Hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)      

        # Left Knee.
        l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)  

        # Right Knee.
        r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
        r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h) 

        # Left AnkLe.
        l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h) 

        # Right AnkLe.
        r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)             
        
        # Calculate angles.
        neck_inclination =0
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)

        # calcualte neck tilt
        neck_tilt =0
        left_dist = findDistance(l_ear_x, l_ear_y, l_shldr_x, l_shldr_y)
        right_dist =findDistance(r_ear_x, r_ear_y, r_shldr_x, r_shldr_y)
        neck_tilt =left_dist/right_dist
        
        # calcuate trunk angle
        left_bend =0
        left_bend = 180- findAngle(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)

        # calcualte trunk twist
        trunk_twist =0
        left_dist = findDistance(r_shldr_x, r_shldr_y, l_hip_x, l_hip_y)
        right_dist =findDistance(l_shldr_x, l_shldr_y, r_hip_x, r_hip_y)
        trunk_twist =left_dist/right_dist

        # Calcualte knee angle
        a = np.array([l_hip_x,l_hip_y])
        b = np.array([l_knee_x,l_knee_y])
        c = np.array([l_ankle_x,l_ankle_y])
        left_leg_angle =find3pointangle(a,b,c)

        a = np.array([r_hip_x,r_hip_y])
        b = np.array([r_knee_x,r_knee_y])
        c = np.array([r_ankle_x,r_ankle_y])
        right_leg_angle =find3pointangle(a,b,c)

        leg_angle = max(left_leg_angle,right_leg_angle)
        leg_ratio = left_leg_angle/right_leg_angle


        # # mark dot
        #x1 =int(left_wrist_landmark[0])
        #y1 =int(left_wrist_landmark[1])
        # cv2.circle(landmarked_image,(x1,y1), 7, yellow, -1)

        # Upper Arm Angle
        rigth_uarm_bend =0
        rigth_uarm_bend = 180- findAngle(r_shldr_x, r_shldr_y, r_wrist_x, r_wrist_y)
    
        # lower Arm Angle
        rigth_larm_bend =0
        rigth_larm_bend = 180- findAngle(r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y)

        # Wrist Angle
        wrist_bend_angle =0
        wrist_bend_angle =180- findAngle(r_index_x, r_index_y, r_wrist_x, r_wrist_y)

        # Nose
        nose_x = int(lm.landmark[lmPose.NOSE].x * w)
        nose_y = int(lm.landmark[lmPose.NOSE].y * h) 

        # mid of shoulder
        m_shldr_x =int((r_shldr_x+l_shldr_x)/2)
        m_shldr_y =int((r_shldr_y+l_shldr_y)/2)

        # mid of ear
        m_ear_x =int((r_ear_x+l_ear_x)/2)
        m_ear_y =int((r_ear_y+l_ear_y)/2) 

        # mid of hip
        m_hip_x =int((r_hip_x+l_hip_x)/2)
        m_hip_y =int((r_hip_y+l_hip_y)/2)      





        
        cv2.circle(landmarked_image,(m_shldr_x,m_shldr_y), 3, yellow, -1)
        cv2.circle(landmarked_image,(m_ear_x,m_ear_y), 30, yellow, -1)
        cv2.circle(landmarked_image,(m_hip_x,m_hip_y), 3, yellow, -1)
        cv2.line(landmarked_image, (m_shldr_x,m_shldr_y), (m_ear_x,m_ear_y), green, 4)
        cv2.line(landmarked_image, (l_shldr_x,l_shldr_y), (r_shldr_x,r_shldr_y), blue, 4)
        cv2.line(landmarked_image, (m_shldr_x,m_shldr_y), (m_hip_x,m_hip_y), green, 4)
        cv2.line(landmarked_image, (l_hip_x,l_hip_y), (r_hip_x,r_hip_y), blue, 4)
        cv2.line(landmarked_image, (l_hip_x,l_hip_y), (l_knee_x,l_knee_y), blue, 4)
        cv2.line(landmarked_image, (r_hip_x,r_hip_y), (r_knee_x,r_knee_y), blue, 4)
        cv2.line(landmarked_image, (l_knee_x,l_knee_y), (l_ankle_x,l_ankle_y), blue, 4)
        cv2.line(landmarked_image, (r_knee_x,r_knee_y), (r_ankle_x,r_ankle_y), blue, 4)
        cv2.line(landmarked_image, (r_shldr_x,r_shldr_y), (r_elbow_x,r_elbow_y), blue, 4)
        cv2.line(landmarked_image, (r_wrist_x,r_wrist_y), (r_elbow_x,r_elbow_y), blue, 4)
        cv2.line(landmarked_image, (l_shldr_x,l_shldr_y), (l_elbow_x,l_elbow_y), blue, 4)
        cv2.line(landmarked_image, (l_wrist_x,l_wrist_y), (l_elbow_x,l_elbow_y), blue, 4)
        #cv2.circle(landmarked_image,(x1,y1), 7, yellow, -1)
    
    else:
        neck_inclination = 0
        neck_tilt= 1
        left_bend =0
        trunk_twist =0
        leg_angle =0
        leg_ratio =0
        rigth_uarm_bend =0
        rigth_larm_bend =0
        wrist_bend_angle =0

    return landmarked_image,neck_inclination,neck_tilt,left_bend,trunk_twist,leg_angle,leg_ratio,rigth_uarm_bend,rigth_larm_bend,wrist_bend_angle

    
      
    
        
     