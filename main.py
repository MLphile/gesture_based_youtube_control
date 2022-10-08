from models.model_architecture import model
import pandas as pd
import mediapipe as mp
import dlib
from imutils import face_utils
from utils import *

################################################### VARIABLES INITIALIZATION ###########################################################

# Set to normal mode (=> no recording of data)
mode = 0
CSV_PATH = 'data/gestures.csv'

# Camera settings
WIDTH = 1028//2
HEIGHT = 720//2

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)


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
GESTURE_RECOGNIZER_PATH = 'models/model.pth'
model.load_state_dict(torch.load(GESTURE_RECOGNIZER_PATH))

# Load Label
LABEL_PATH = 'data/label.csv'
labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()


# confidence threshold(required to translate gestures into commands)
CONF_THRESH = 0.9

# history to track the n last detected commands
GESTURE_HISTORY = deque([])

# general counter (for volum up/down; forward/backward)
GEN_COUNTER = 0

# Face detection (absence feature)
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.75)

IS_ABSENT = None # only for testing purposes...change this by video status (paused or playing)
ABSENCE_COUNTER = 0
ABSENCE_COUNTER_THRESH = 20

# Frontal face + face landmarks (sleepness feature)
SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


IS_SLEEPING = None # only for testing purposes...change this by video status (paused or playing)
SLEEP_COUNTER = 0
SLEEP_COUNTER_THRESH = 20
EAR_THRESH = 0.21 
EAR_HISTORY = deque([])

################################################### INITIALIZATION END ###########################################################



while True:
    key = cv.waitKey(1) 
    if key == ord('q'):
        break

    
    # choose mode (normal or recording)
    mode = select_mode(key, mode=mode)

    # class id for recording
    class_id = get_class_id(key)

    # read camera
    has_frame, frame = cap.read()
    if not has_frame:
        break

    # horizontal flip and color conversion for mediapipe
    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detection and mouse zones
    det_zone, m_zone = det_mouse_zones(frame)

############################################ GESTURE DETECTION / TRAINING POINT LOGGING ###########################################################
 
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get landmarks coordinates
            coordinates_list = calc_landmark_coordinates(frame_rgb, hand_landmarks)
            important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]


            # Conversion to relative coordinates and normalized coordinates
            preprocessed = pre_process_landmark(important_points)
            
            # compute the needed distances to add to coordinate features
            d0 = calc_distance(coordinates_list[0], coordinates_list[5])
            pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
            distances = normalize_distances(d0, get_all_distances(pts_for_distances)) 
            

            # Write to the csv file "keypoint.csv"(if mode == 1)
            # logging_csv(class_id, mode, preprocessed)
            features = np.concatenate([preprocessed, distances])
            draw_info(frame, mode, class_id)
            logging_csv(class_id, mode, features, CSV_PATH)

            
            # inference
            conf, pred = predict(features, model)
            gesture = labels[pred]

            
####################################################### YOUTUBE PLAYER CONTROL ###########################################################                       
                
            # check if middle finger mcp is inside the detection zone and prediction confidence is higher than a given threshold
            if cv.pointPolygonTest(det_zone, coordinates_list[9], False) == 1 and conf >= CONF_THRESH: 

                # track command history
                gest_hist = track_history(GESTURE_HISTORY, gesture)

                if len(gest_hist) >= 2:
                    before_last = gest_hist[len(gest_hist) - 2]
                else:
                    before_last = gest_hist[0]

            ############### mouse gestures ##################
                if gesture == 'Move_mouse':
                    x, y = mouse_zone_to_screen(coordinates_list[9], m_zone)
                    
                    # smoothe mouse movements
                    CLOX = PLOCX + (x - PLOCX) / SMOOTH_FACTOR
                    CLOXY = PLOCY + (y - PLOCY) / SMOOTH_FACTOR
                    pyautogui.moveTo(CLOX, CLOXY)
                    PLOCX, PLOCY = CLOX, CLOXY

                if gesture == 'Right_click' and before_last != 'Right_click':
                    pyautogui.rightClick()

                if gesture == 'Left_click' and before_last != 'Left_click':
                    pyautogui.click()    


            ############### Other gestures ################## 
                if gesture == 'Play_Pause' and before_last != 'Play_Pause':
                    pyautogui.press('space')
                
                elif gesture == 'Vol_up_gen':
                    pyautogui.press('volumeup')

                elif gesture == 'Vol_down_gen':
                    pyautogui.press('volumedown')

                elif gesture == 'Vol_up_ytb':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('up')

                elif gesture == 'Vol_down_ytb':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('down')

                elif gesture == 'Forward':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('right')
                
                elif gesture == 'Backward':
                    GEN_COUNTER += 1
                    if GEN_COUNTER % 4 == 0:
                        pyautogui.press('left')
                
                elif gesture == 'fullscreen' and before_last != 'fullscreen':
                    pyautogui.press('f')
                
                elif gesture == 'Cap_Subt' and before_last != 'Cap_Subt':
                    pyautogui.press('c')

                elif gesture == 'Neutral':
                    GEN_COUNTER = 0 

                # show detected gesture
                cv.putText(frame, f'{gesture} | {conf: .2f}', (int(WIDTH*0.05), int(HEIGHT*0.07)),
                    cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)

                

    
##################################################### SLEEPNESS DETECTION ########################################################### 

    frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY)
    
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
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        EAR_HISTORY = track_history(EAR_HISTORY, round(ear, 2), 20)
        mean_ear = sum(EAR_HISTORY) / len(EAR_HISTORY)
        

        # check if eyes are closed for a certain number of consecutive frames, then pause the video 
        if mean_ear < EAR_THRESH:
            SLEEP_COUNTER += 1

            if SLEEP_COUNTER > SLEEP_COUNTER_THRESH and IS_SLEEPING == False:
                pyautogui.press('space')
                IS_SLEEPING = True

        else:
            SLEEP_COUNTER = 0
            IS_SLEEPING = False

        # show eye contours and mean EAR
        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)
        cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        cv.putText(frame, f'EAR: {mean_ear:.2f}',(int(WIDTH*0.90 ), int(HEIGHT*0.08)),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)


################################################# ABSENCE DETECTION ###########################################################

    # Based on media pipe face detection (more robust)
    results = face_detection.process(frame_rgb)

    # check if no face is detected, if so pause the video
    if results.detections == None:
        ABSENCE_COUNTER += 1
        if ABSENCE_COUNTER > ABSENCE_COUNTER_THRESH and IS_ABSENT == False:
            pyautogui.press('space')  
            IS_ABSENT = True

    else:
        ABSENCE_COUNTER = 0 
        IS_ABSENT = False

       # draw face bounding box
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            frame_height, frame_width = frame.shape[:2]
            x, y = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)
            w, h = int(bbox.width * frame_width), int(bbox.height * frame_height)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)


    cv.imshow('', frame)
cap.release()
cv.destroyAllWindows()
