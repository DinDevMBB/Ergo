
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import media
import posture
import Score_table
import gc

def main():
    # intialize sample image and video
    DEMO_IMAGE = 'demo.jpg'
    DEMO_VIDEO = 'input.mp4'


    #@st.cache(allow_output_mutation=True) // updated in new version
    @st.cache_resource

    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized


    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 350px;
            margin-left: -350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    app_mode = st.sidebar.selectbox('Choose the App mode',
    ['About App','Run on Image','Run on Video'])

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')
    Force =st.sidebar.radio("Add Force/ Load Score",
    ('< 11 lbs (~5 Kgs)', '11 to 22 lbs ', '> 22 lbs (~10 Kgs)'))
    if Force == '< 11 lbs (~5 Kgs)':
        LoadScore = 0
    elif Force =='11 to 22 lbs ':
        LoadScore = 1
    else:
        LoadScore = 2

    #
    if app_mode =='About App':
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 250px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 250px;
            margin-left: -250px;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )



        #####################
        # Page Title
        ######################
        p=r'header.png'
        image = Image.open(p)

        st.image(image, use_column_width=True)

        st.write("""
        REBA (Rapid Entire Body Assessment) is a widely-used ergonomic tool that quickly evaluates whole-body postural risks in various work environments.
        This application automatically identifies posture issues and calculates the corresponding REBA score.
        ***
        """)

    elif app_mode =='Run on Image':
        neck_angle = 0
        
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 250px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 250px;
            margin-left: -250px;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )
        # upload an image
        image_path = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

        # open the image using cv2
        if image_path is not None:
            image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
            
        else:
            demo_image = DEMO_IMAGE
            image = cv2.imread(demo_image)

        # resize the image        
        r_image = image_resize(image =image, width=450)

        # convert image to RGB format
        image_in_RGB = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)

        # Here we will read our image from the specified path to detect the pose
        img,neck_angle,neck_tilt,bend,twist,leg_angle,leg_ratio,uarm_angle,larm_angle,wrist_angle= media.detectPose(image_pose=image_in_RGB,MIN_CONFIEDENCE=detection_confidence)
    
        st.image(img)
        st.button('Click Here to Update Score after loading new image',)
        def conclusion(x):
            if x ==0:
                msg ='Please intialize the model by clicking update button above'
                st.markdown(f'<h1 style="color:#0000FF;font-size:24px;">{"Please intialize the model by clicking update button above"}</h1>', unsafe_allow_html=True)
            elif(x==1):
                msg ='Negligible Risk'
                st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"Negligible Risk"}</h1>', unsafe_allow_html=True)
            elif(x>1 and x<=3):
                msg ='Low Risk, Change may be needed'
                st.markdown(f'<h1 style="color:#33ff33;font-size:24px;">{"Low Risk, Change may be needed"}</h1>', unsafe_allow_html=True)
            elif(x>3 and x<=7):
                msg ='Medium Risk, Further investigation and change soon'
                st.markdown(f'<h1 style="color:#FFBF00;font-size:24px;">{"Medium Risk, Further investigation and change soon"}</h1>', unsafe_allow_html=True)
            elif(x>7 and x<=10):
                msg ='High Risk, Investigate and implement change'
                st.markdown(f'<h1 style="color:#FF0000;font-size:24px;">{"High Risk, Investigate and implement change"}</h1>', unsafe_allow_html=True)
            else:
                msg ='Very High Risk, Implement change'
                st.markdown(f'<h1 style="color:#FF0000;font-size:24px;">{"Very High Risk, Implement change"}</h1>', unsafe_allow_html=True)
            return msg
        
        if 'TblC_Score' not in st.session_state:
            st.session_state['TblC_Score'] = 0


        # The message and nested widget will remain on the page
        st.write("REBA score:" )
        

        
        html_str = f"""
        <style>
        p.a {{
        font: bold {30}px Courier;
        }}
        </style>
        <p class="a">{st.session_state.TblC_Score}</p>
        """

        st.markdown(html_str, unsafe_allow_html=True)
        conc = conclusion(st.session_state.TblC_Score)
        
            


    ######################################################################################################
        st.subheader("Step 1: Locate Neck Position")
        col1, col2,col3,col4 = st.columns(4)
        #col1.header(r"$\textsf{\small Step 1: Locate Neck Position}$")
        col2.caption("Neck Angle")
        col3.caption("Angle Score")
        col4.caption("Tilt Score")

        p=r'neck_angle.png'
        col1.image(p, use_column_width=True)
        neck_score = posture.neck_inclination(bend)
        with col2:
            st.text(round(neck_angle))
        col3.text(neck_score)
        tilt_score = posture.neck_tilt_score(neck_tilt)
        col4.text(tilt_score)
        final_neck_score = neck_score + tilt_score
    ###############################################################################################
        # Step 2 bend angle
        st.subheader("Step 2: Locate Trunk Position")
        col1, col2,col3,col4 = st.columns(4)
        #col1.header(r"$\textsf{\small Step 1: Locate Neck Position}$")
        col2.caption("Bend Angle")
        col3.caption("Trunk Score")
        col4.caption("Twist Score")

        p=r'trunk.png'
        col1.image(p, use_column_width=True)
        bend_score = posture.trunk_score(bend)
        with col2:
            st.text(round(bend))
        col3.text(bend_score)
        twist_score = posture.trunk_twist_score(twist)
        col4.text(twist_score)
        final_trunk_score = bend_score + twist_score
    #################################################################################
        # Step 3 bend angle
        st.subheader("Step 3: Leg Score")
        col1, col2,col3,col4 = st.columns(4)
        col2.caption("Leg Angle")
        col3.caption("Leg Score")
        col4.caption("Adjust")
        p=r'legs.png'
        col1.image(p, use_column_width=True)
        leg_score = posture.trunk_score(180-leg_angle)
        with col2:
            st.text(round(180-leg_angle))
        col3.text(leg_score)
        leg_twist_score = posture.trunk_twist_score(twist)
        col4.text(leg_twist_score)
        final_leg_score = leg_score + leg_twist_score

    #####################################################################################
        # Get Table A score
        inputscore = final_neck_score*100 + final_leg_score*10 + final_trunk_score
        Posture_score = Score_table.GetA(inputscore)
        st.write("Table A Posture score:" , Posture_score)
        st.write("Force/ Load score:" ,LoadScore)
        tblA_score =Posture_score +LoadScore
        st.write("Final TableA Score:" ,tblA_score)

    ###############################################################################################
        # Step 7 Upper Arm bend angle
        st.subheader("Step 4: Locate Upper Arm Position")
        col1, col2,col3,col4 = st.columns(4)
        #col1.header(r"$\textsf{\small Step 1: Locate Neck Position}$")
        col2.caption("Upper Arm Angle")
        col3.caption("Angle Score")
        # col4.caption("Twist Score")

        p=r'upper_arm.png'
        col1.image(p, use_column_width=True)
        upper_arm_score = posture.uarm_score(uarm_angle)
        with col2:
            st.text(round(uarm_angle))
        col3.text(upper_arm_score)
        twist_score = posture.trunk_twist_score(twist)
        # col4.text(twist_score)
        # final_trunk_score = upper_arm_score + twist_score
    ################################################################################
    ###############################################################################################
        # Step 8 Lower Arm bend angle
        st.subheader("Step 5: Locate Lower Arm Position")
        col1, col2,col3,col4 = st.columns(4)
        #col1.header(r"$\textsf{\small Step 1: Locate Neck Position}$")
        col2.caption("Lower Arm Angle")
        col3.caption("Angle Score")
        # col4.caption("Twist Score")

        p=r'lower_arm.png'
        col1.image(p, use_column_width=True)
        lower_arm_score = posture.larm_score(larm_angle)
        with col2:
            st.text(round(larm_angle))
        col3.text(lower_arm_score)
        #twist_score = posture.trunk_twist_score(twist)
        # col4.text(twist_score)
        # final_trunk_score = upper_arm_score + twist_score
    ################################################################################
    ###############################################################################################
        # Step 9 Locate Wrist position
        st.subheader("Step 6: Locate Wrist Position")
        col1, col2,col3,col4 = st.columns(4)
        #col1.header(r"$\textsf{\small Step 1: Locate Neck Position}$")
        col2.caption("Wrist Angle")
        col3.caption("Angle Score")
        # col4.caption("Twist Score")

        p=r'wrist_angle.png'
        col1.image(p, use_column_width=True)
        wrist_angle_score = posture.wrist_score(wrist_angle)
        with col2:
            st.text(round(wrist_angle))
        col3.text(wrist_angle_score)
        #twist_score = posture.trunk_twist_score(twist)
        # col4.text(twist_score)
        # final_trunk_score = upper_arm_score + twist_score
    ################################################################################
    #####################################################################################
        # Get Table B score
        inputscore = upper_arm_score*100 + lower_arm_score*10 + wrist_angle_score
        Posture_score = Score_table.GetB(inputscore)
        st.write("Table B Posture score:" , Posture_score)
        st.write("Coupling score:" ,0)
        st.write("Final Table B Score:" ,Posture_score)

    #############################################################################################
        # Get Table C Score

        st.session_state.TblC_Score = Score_table.GetC(tblA_score,Posture_score)
        st.write("Table C Final score:" , st.session_state.TblC_Score)
        


    ###########################################################################################

        





    elif app_mode =='Run on Video':
        st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 250px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 250px;
            margin-left: -250px;
        }
        </style>
        """,
        unsafe_allow_html=True,
        )
        video_file_buffer  = st.sidebar.file_uploader(
        label = "Please upload a video preferably sideview.",type=["mp4", "mpeg"])
        st.set_option('deprecation.showfileUploaderEncoding', False)

        st.video(DEMO_VIDEO)

        #####
# def clear_cache():
#     keys = list(st.session_state.keys())
#     for key in keys:
#         st.session_state.pop(key)
# def clear():
#     st.button('Clear Cache', on_click=clear_cache)

if __name__ == "__main__":
    gc.enable()
    main()
    #clear_cache()
    # clear()
    gc.collect()
    


    