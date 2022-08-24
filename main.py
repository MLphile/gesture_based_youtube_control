from models.model_architecture import model
import pandas as pd
import mediapipe as mp
import dlib
from imutils import face_utils
from utils import *
from collections import deque



# Camera resolution
width = 1028//2
height = 720//2


# Camera setting
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


# Screen size
screen_size = pyautogui.size()

# Mouse mouvement stabilization
smooth_factor = 6

plocX, plocY = 0, 0 # previous x, y locations
clocX, clocY = 0, 0 # current x, y locations


# Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75)

# Hand landmarks drawing
mp_drawing = mp.solutions.drawing_utils

# Load model
GESTURE_RECOGNIZER_PATH = 'models/model.pth'
model.load_state_dict(torch.load(GESTURE_RECOGNIZER_PATH))

# Load Label
LABEL_PATH = 'data/label.csv'
labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()

# mode normal (don't record data for training)
mode = 0  

# confidence threshold(required to send most of the commands)
conf_threshold = 0.8

# command history
history = deque([])


# Face detector mediapipe
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.75)
   
is_absent = None
absence_counter = 0
absence_counter_threshold = 10

# Frontal face + face landmarks
SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


is_sleeping = None
sleep_counter = 0
sleep_counter_thresh = 30
ear_threshold = 0.26    
while True:
    key = cv.waitKey(1)
    if key == ord('q'):
        break

    # reset mode and class id
    class_id, mode = select_mode(key, mode)

    # read camera
    has_frame, frame = cap.read()
    if not has_frame:
        break

    # horizontal flip and color conversion for mediapipe
    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Detection zone
    zone = detection_zone(frame)

    

    # HAND DETECTION
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get landmarks coordinates
            coordinates_list = calc_landmark_coordinates(frame, hand_landmarks)

            # Conversion to relative coordinates and normalized coordinates
            preprocessed = pre_process_landmark(coordinates_list)


            # Write to the csv file "keypoint.csv"(if mode == 1)
            logging_csv(class_id, mode, preprocessed)


    #         # check if middle finger mcp is inside the detection zone for command execution
    #         if cv.pointPolygonTest(zone, coordinates_list[9], False) == 1: 
                
    #             # inference
    #             # pred = predict(preprocessed, model)
    #             # gesture = labels[pred]

    #             # activate/deactivate mouse mode
    #             gesture = 'mouse_on'
                
                
    #             # keyboard

    #             # mouse
                
    #             if  gesture == 'mouse_on':
    #                 conf, pred = predict(preprocessed, model)
    #                 gesture = labels[pred]
                    
                    
    #                 if gesture == 'Forward' or gesture == 'Backward':
    #                     x, y = mouse_movement_area(coordinates_list[9], zone, screen_size)
                        
    #                     clocX = plocX + (x - plocX) / smooth_factor
    #                     clocY = plocY + (y - plocY) / smooth_factor
    #                     pyautogui.moveTo(clocX, clocY)

    #                     plocX, plocY = clocX, clocY

    #                 if conf >= conf_threshold:
    #                     history = track_history(history, gesture)
    #                     # print(history)
    #                     before_last = history[len(history) - 2]
    #                     if gesture == 'Stop' and before_last != 'Stop':
    #                         pyautogui.click()

    #                     if gesture == 'Pause' and before_last != 'Pause':
    #                         pyautogui.rightClick()



                

    #             # cv.putText(frame, f'COMMAND: {command}', (int(width*0.05), int(height*0.1)),
    #             #            cv.FONT_HERSHEY_COMPLEX, 1, (22, 69, 22), 3, cv.LINE_AA)

            

    

    # # FACE AND FACE LANDMARKS
    # # 1. Sleepness feature (based on eye aspect ratio)
    # # frame_rgb = imutils.resize(frame_rgb, width=450)
    # frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY)
    # faces = detector(frame_gray)
    # for face in faces:
        
    #     landmarks = predictor(frame_gray, face)
    #     landmarks = face_utils.shape_to_np(landmarks)


    #     leftEye = landmarks[lStart:lEnd]
    #     rightEye = landmarks[rStart:rEnd]
    #     # print(leftEye)

    #     leftEAR = eye_aspect_ratio(leftEye)
    #     rightEAR = eye_aspect_ratio(rightEye)
	# 	# average the eye aspect ratio together for both eyes
    #     ear = (leftEAR + rightEAR) / 2.0
        
    #     # check if sleeping, then pause the video
    #     if ear < ear_threshold:
    #         sleep_counter += 1
    #         # print(ear)
    #         if sleep_counter > sleep_counter_thresh and is_sleeping == False:
    #             pyautogui.press('space')
    #             print('sleeping')
    #             is_sleeping = True


    #     else:
    #         sleep_counter = 0
    #         if is_sleeping:
    #             pyautogui.press('space')
    #             print('not sleeping')
        
    #         is_sleeping = False





    #     leftEyeHull = cv.convexHull(leftEye)
    #     rightEyeHull = cv.convexHull(rightEye)
    #     cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    #     cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    #     cv.putText(frame, f'{ear:.2f}',(int(width*0.05), int(height*0.1)),
    #                        cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3, cv.LINE_AA)

    
    # # 2. Presence/absence feature based on face detection
    # results = face_detection.process(frame_rgb) 
    # # face detected
    # if results.detections:
    #     absence_counter = 0

    #     if is_absent:
    #         pyautogui.press('space')
        
    #     is_absent = False

    #    # draw bounding box
    #     for detection in results.detections:
    #         bbox = detection.location_data.relative_bounding_box
    #         frame_height, frame_width = frame.shape[:2]
    #         x, y = int(bbox.xmin * frame_width), int(bbox.ymin * frame_height)
    #         w, h = int(bbox.width * frame_width), int(bbox.height * frame_height)
    #         cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    
    # else:
    #     absence_counter += 1
    #     if absence_counter > absence_counter_threshold and is_absent == False:
            
    #         pyautogui.press('space')  
    #         is_absent = True
                    

    frame = draw_info(frame, mode, class_id)
    cv.imshow('', frame)


cap.release()
cv.destroyAllWindows()
