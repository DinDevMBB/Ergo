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

def neck_incl(angle):
    if (angle >=0 and angle <=20):
        score =1
    else:
        score =2
    return score

def neck_tilt_score(ratio):
    if (ratio >=0.3 and ratio <= 1.3):
        score =0
    else:
        score =1
    return score  
def trunk_score(angle):
    if angle ==0:
        score =1
    elif (angle >0 and angle <=20):
        score =2
    elif (angle >20 and angle <=60):
        score =3
    elif angle <0:
        score =3
    else:
        score =4
    return score
def trunk_twist_score(ratio):
    if (ratio >=0.3 and ratio <= 1.3):
        score =0
    else:
        score =1
    return score
def uarm_score(angle):
    if angle ==0:
        score =1
    elif (angle >-20 and angle <=20):
        score =1
    elif (angle >20 and angle <=45):
        score =2
    elif (angle >45 and angle <=90):
        score =3
    elif (angle >90):
        score =4
    else:
        score =4
    return score
def larm_score(angle):
    if angle ==0:
        score =1
    elif (angle >0 and angle <=20):
        score =1
    else:
        score =2
    return score

def wrist_score(angle):
    if angle ==0:
        score =1
    elif (angle >-15 and angle <=15):
        score =1
    else:
        score =2
    return score

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
    
        # Left index.
        l_index_x = int(lm.landmark[lmPose.LEFT_INDEX].x * w)
        l_index_y = int(lm.landmark[lmPose.LEFT_INDEX].y * h)

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
                  
    #################################################################################    
        # Calculate angles.
        neck_inclination =0
        #neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        neck_angle = findAngle(m_shldr_x, m_shldr_y, m_ear_x, m_ear_y)

        # calcualte neck tilt
        neck_tilt =0
        left_dist = findDistance(l_ear_x, l_ear_y, l_shldr_x, l_shldr_y)
        right_dist =findDistance(r_ear_x, r_ear_y, r_shldr_x, r_shldr_y)
        neck_tilt =left_dist/right_dist

        neck_score = neck_incl(neck_angle)
        tilt_score = neck_tilt_score(neck_tilt)

    ####################################################################################  
        # calcuate trunk angle
        bend_angle =0
        #left_bend = 180- findAngle(l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
        bend_angle = 180- findAngle(m_shldr_x, m_shldr_y, m_hip_x, m_hip_y)

        # calcualte trunk twist
        trunk_twist =0
        left_dist = findDistance(r_shldr_x, r_shldr_y, l_hip_x, l_hip_y)
        right_dist =findDistance(l_shldr_x, l_shldr_y, r_hip_x, r_hip_y)
        trunk_twist =left_dist/right_dist

        bend_score = trunk_score(bend_angle)
        twist_score = trunk_twist_score(trunk_twist)

#################################################################################################
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

        leg_score = trunk_score(180-leg_angle)
        leg_twist_score = trunk_twist_score(leg_ratio)

#############################################################################################

        # Right Upper Arm Angle
        rigth_uarm_bend =0
        rigth_uarm_bend = 180- findAngle(r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y)
        
        # Left Upper Arm Angle
        left_uarm_bend =0
        left_uarm_bend = 180- findAngle(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)

        # Upper Arm Angle

        uarm_bend =max(rigth_uarm_bend,left_uarm_bend)

        upper_arm_score = uarm_score(uarm_bend)
        twist_score = trunk_twist_score(neck_tilt)
    
    ##################################################################################
        # lower Arm Angle
        rigth_larm_bend =0
        rigth_larm_bend = 180- findAngle(r_elbow_x, r_elbow_y, r_wrist_x, r_wrist_y)

        # lower Arm Angle
        left_larm_bend =0
        left_larm_bend = 180- findAngle(l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y)

        larm_bend =max(rigth_larm_bend,left_larm_bend)


        lower_arm_score = larm_score(larm_bend)
###########################################################################################
        # Wrist Angle
        r_wrist_bend_angle =0
        r_wrist_bend_angle =180- findAngle(r_index_x, r_index_y, r_wrist_x, r_wrist_y)

        l_wrist_bend_angle =0
        l_wrist_bend_angle =180- findAngle(l_index_x, l_index_y, l_wrist_x, l_wrist_y)

        wrist_bend_angle =max(r_wrist_bend_angle,l_wrist_bend_angle)

        wrist_angle_score = wrist_score(wrist_bend_angle)

##########################################################################################
       
    
    else:
        neck_inclination = 0
        neck_tilt= 1
        bend_angle =0
        trunk_twist =0
        leg_angle =0
        leg_ratio =0
        uarm_bend =0
        rigth_larm_bend =0
        wrist_bend_angle =0
        neck_score =0
        tilt_score =0
        bend_score =0
        twist_score =0
        leg_score = 0
        leg_twist_score = 0
        upper_arm_score = 0
        lower_arm_score =0
        wrist_angle_score =0
        larm_bend =0
        #neck_inclination,neck_tilt,neck_score,tilt_score,bend_angle,trunk_twist,bend_score,twist_score,leg_angle,leg_ratio,leg_score,leg_twist_score,uarm_bend,upper_arm_score,larm_bend,lower_arm_score,wrist_bend_angle,wrist_angle_score =0
        

    return landmarked_image,neck_inclination,neck_tilt,neck_score,tilt_score,bend_angle,trunk_twist,bend_score,twist_score,leg_angle,leg_ratio,leg_score,leg_twist_score,uarm_bend,upper_arm_score,larm_bend,lower_arm_score,wrist_bend_angle,wrist_angle_score

    
      
    
        
     