import sys
sys.path.append('../')

from models.model_architecture import model
import pandas as pd
import mediapipe as mp
import dlib
from imutils import face_utils
import utils as ut
import os
import json



def generate_video():
    ################################################### VARIABLES INITIALIZATION ###########################################################

    # Camera settings
    WIDTH = 1028//2
    HEIGHT = 720//2

    cap = ut.cv.VideoCapture(0)
    cap.set(ut.cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(ut.cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)


    # important keypoints (wrist + tips coordinates)
    # for training the model
    TRAINING_KEYPOINTS = [keypoint for keypoint in range(0, 21, 4)]


    # Mouse mouvement stabilization
    SMOOTH_FACTOR = 6
    PLOCX, PLOCY = 0, 0 # previous x, y locations
    CLOX, CLOXY = 0, 0 # current x, y locations


    # Hand detector
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75)

    # Hand landmarks drawing
    mp_drawing = mp.solutions.drawing_utils

    # Load saved model for hand gesture recognition
    GESTURE_RECOGNIZER_PATH = '../models/model.pth'
    model.load_state_dict(ut.torch.load(GESTURE_RECOGNIZER_PATH))

    # Load Label
    LABEL_PATH = '../data/label.csv'
    labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()


    # confidence threshold(required to translate gestures into commands)
    CONF_THRESH = 0.9

    # history to track the n last detected commands
    GESTURE_HISTORY = ut.deque([])

    # general counter (for volum up/down; forward/backward)
    GEN_COUNTER = 0

    # Face detection (absence feature)
    mp_face_detection = mp.solutions.face_detection

    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.75)

    ABSENCE_COUNTER = 0
    ABSENCE_COUNTER_THRESH = 20

    # Frontal face + face landmarks (sleepness feature)
    SHAPE_PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    SLEEP_COUNTER = 0
    SLEEP_COUNTER_THRESH = 20
    EAR_THRESH = 0.21 
    EAR_HISTORY = ut.deque([])

    STATE_PATH = '../data/player_state.json' 

    ################################################### INITIALIZATION END ###########################################################


    while True:
        
        # read camera
        has_frame, frame = cap.read()
        if not has_frame:
            break

        # horizontal flip and color conversion for mediapipe
        frame = ut.cv.flip(frame, 1)
        frame_rgb = ut.cv.cvtColor(frame, ut.cv.COLOR_BGR2RGB)

        # Detection and mouse zones
        det_zone, m_zone = ut.det_mouse_zones(frame)

    ############################################ GESTURE DETECTION ###########################################################
    
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # get landmarks coordinates
                coordinates_list = ut.calc_landmark_coordinates(frame_rgb, hand_landmarks)
                important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]


                # Conversion to relative coordinates and normalized coordinates
                preprocessed = ut.pre_process_landmark(important_points)
                
                # compute the needed distances to add to coordinate features
                d0 = ut.calc_distance(coordinates_list[0], coordinates_list[5])
                pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
                distances = ut.normalize_distances(d0, ut.get_all_distances(pts_for_distances)) 
                
                features = ut.np.concatenate([preprocessed, distances])
                
                # inference
                conf, pred = ut.predict(features, model)
                gesture = labels[pred]

                
    ####################################################### YOUTUBE PLAYER CONTROL ###########################################################                       
                    
                # check if middle finger mcp is inside the detection zone and prediction confidence is higher than a given threshold
                if ut.cv.pointPolygonTest(det_zone, coordinates_list[9], False) == 1 and conf >= CONF_THRESH: 

                    # track command history
                    gest_hist = ut.track_history(GESTURE_HISTORY, gesture)

                    if len(gest_hist) >= 2:
                        before_last = gest_hist[len(gest_hist) - 2]
                    else:
                        before_last = gest_hist[0]

                ############### mouse gestures ##################
                    if gesture == 'Move_mouse':
                        x, y = ut.mouse_zone_to_screen(coordinates_list[9], m_zone)
                        
                        # smoothe mouse movements
                        CLOX = PLOCX + (x - PLOCX) / SMOOTH_FACTOR
                        CLOXY = PLOCY + (y - PLOCY) / SMOOTH_FACTOR
                        ut.pyautogui.moveTo(CLOX, CLOXY)
                        PLOCX, PLOCY = CLOX, CLOXY

                    if gesture == 'Right_click' and before_last != 'Right_click':
                        ut.pyautogui.rightClick()

                    if gesture == 'Left_click' and before_last != 'Left_click':
                        ut.pyautogui.click()    


                ############### Other gestures ################## 
                    if gesture == 'Play_Pause' and before_last != 'Play_Pause':
                        ut.pyautogui.press('space')
                    
                    elif gesture == 'Vol_up_gen':
                        ut.pyautogui.press('volumeup')

                    elif gesture == 'Vol_down_gen':
                        ut.pyautogui.press('volumedown')

                    elif gesture == 'Vol_up_ytb':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            ut.pyautogui.press('up')

                    elif gesture == 'Vol_down_ytb':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            ut.pyautogui.press('down')

                    elif gesture == 'Forward':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            ut.pyautogui.press('right')
                    
                    elif gesture == 'Backward':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            ut.pyautogui.press('left')
                    
                    elif gesture == 'fullscreen' and before_last != 'fullscreen':
                        ut.pyautogui.press('f')
                    
                    elif gesture == 'Cap_Subt' and before_last != 'Cap_Subt':
                        ut.pyautogui.press('c')

                    elif gesture == 'Neutral':
                        GEN_COUNTER = 0 

                    # show detected gesture
                    ut.cv.putText(frame, f'{gesture} | {conf: .2f}', (int(WIDTH*0.05), int(HEIGHT*0.07)),
                        ut.cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, ut.cv.LINE_AA)

                    

        
    ##################################################### SLEEPNESS DETECTION ########################################################### 

        frame_gray = ut.cv.cvtColor(frame_rgb, ut.cv.COLOR_RGB2GRAY)
        
        # 1. Frontal face detection   
        faces = detector(frame_gray)
        for face in faces:
            
            # 2. Facial landmarks
            landmarks = predictor(frame_gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # 3. Eye landmarks
            leftEye = landmarks[lStart:lEnd]
            rightEye = landmarks[rStart:rEnd]

            
            # 4. Eye aspect ratio to detect when eyes are closed
            leftEAR = ut.eye_aspect_ratio(leftEye)
            rightEAR = ut.eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            EAR_HISTORY = ut.track_history(EAR_HISTORY, round(ear, 2), 20)
            mean_ear = sum(EAR_HISTORY) / len(EAR_HISTORY)
            

            # check if eyes are closed for a certain number of consecutive frames, then pause the video 
            if mean_ear < EAR_THRESH:
                SLEEP_COUNTER += 1

                ps = None
                # get player state (e.g. 1 -> playing, 2 -> pause)
                if os.path.exists(STATE_PATH):
                    with open(STATE_PATH) as json_file:
                        player_state = json.load(json_file)
                        player_state = json.loads(player_state)

                        ps = player_state['playerState']

                if (SLEEP_COUNTER  > SLEEP_COUNTER_THRESH and SLEEP_COUNTER % 3 == 0) and ps == 1:
                    ut.pyautogui.press('space')

            else:
                SLEEP_COUNTER = 0

            # show eye contours and mean EAR
            leftEyeHull = ut.cv.convexHull(leftEye)
            rightEyeHull = ut.cv.convexHull(rightEye)
            ut.cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            ut.cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            ut.cv.putText(frame, f'EAR: {mean_ear:.2f}',(int(WIDTH*0.90 ), int(HEIGHT*0.08)),
                            ut.cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, ut.cv.LINE_AA)


    ################################################# ABSENCE DETECTION ###########################################################

        # Based on media pipe face detection (more robust)
        results = face_detection.process(frame_rgb)

        # check if no face is detected, if so pause the video
        if results.detections == None:
            ABSENCE_COUNTER += 1

            ps = None
            # get player state (e.g. 1 -> playing, 2 -> pause)
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH) as json_file:
                    player_state = json.load(json_file)
                    player_state = json.loads(player_state)
                    ps = player_state['playerState']
  
            if (ABSENCE_COUNTER > ABSENCE_COUNTER_THRESH and ABSENCE_COUNTER % 3 == 0) and ps == 1:
                ut.pyautogui.press('space')  
    

        else:
            ABSENCE_COUNTER = 0


        # draw face bounding box
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                frame_height, frame_width = frame.shape[:2]
                x, y = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)
                w, h = int(bbox.width * frame_width), int(bbox.height * frame_height)
                ut.cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)



        _ , buffer = ut.cv.imencode('.jpg', frame)
        frame  = buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
