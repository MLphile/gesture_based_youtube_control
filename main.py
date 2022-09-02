from models.model_architecture import model
import pandas as pd
import mediapipe as mp
import dlib
from imutils import face_utils
from utils import *

# Set to nornal mode (=> no recording of data)
mode = 0

# Camera resolution
width = 1028//2
height = 720//2


# Camera setting
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

# important keypoints (wrist + tips)
to_save = [keypoint for keypoint in range(0, 21, 4)]

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


# confidence threshold(required to translate gestures into commands)
conf_threshold = 0.8

# command history
history = deque([])

# general counter (for volum up/down; forward/backward)
gen_counter = 0
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

    

    # HAND DETECTION
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get landmarks coordinates
            coordinates_list = calc_landmark_coordinates(frame, hand_landmarks)
            important_points = [coordinates_list[i] for i in to_save]


            # Conversion to relative coordinates and normalized coordinates
            preprocessed = pre_process_landmark(important_points)
            
            # compute the needed distances to add as to coordinates features
            d0 = calc_distance(coordinates_list[0], coordinates_list[5])
            pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
            distances = normalize_distances(d0, get_all_distances(pts_for_distances)) 
            

            # Write to the csv file "keypoint.csv"(if mode == 1)
            # logging_csv(class_id, mode, preprocessed)
            features = np.concatenate([preprocessed, distances])
            logging_csv(class_id, mode, features)

            # inference
            conf, pred = predict(features, model)
            gesture = labels[pred]



            # if conf >= conf_threshold:
            #     cv.putText(frame, f'{gesture} | {conf: .2f}', (int(width*0.05), int(height*0.1)),
            #         cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2, cv.LINE_AA)

                
######################### Youtube player control ###########################################################                       
                

            # check if middle finger mcp is inside the detection zone and prediction confidence is good enough
            if cv.pointPolygonTest(det_zone, coordinates_list[9], False) == 1 and conf >= conf_threshold: 

                # track command history
                history = track_history(history, gesture)
                # print(history)
                if len(history) >= 2:
                    before_last = history[len(history) - 2]
                else:
                    before_last = history[0]

            ############### Mouse ##################
                if gesture == 'Move_mouse':
                    x, y = mouse_zone_to_screen(coordinates_list[9], m_zone, screen_size)
                    
                    clocX = plocX + (x - plocX) / smooth_factor
                    clocY = plocY + (y - plocY) / smooth_factor
                    pyautogui.moveTo(clocX, clocY)

                    plocX, plocY = clocX, clocY

                if gesture == 'Right_click' and before_last != 'Right_click':
                    pyautogui.rightClick()
                    print('right')


                if gesture == 'Left_click' and before_last != 'Left_click':
                    pyautogui.click()
                    print('left')     


            ############### Main gestures ################## 
                if gesture == 'Play_Pause' and before_last != 'Play_Pause':
                    pyautogui.press('space')

                if gesture == 'Vol_up_ytb' and before_last != 'Vol_up_ytb':
                    pyautogui.press('up')

                if gesture == 'Vol_down_ytb' and before_last != 'Vol_down_ytb':
                    pyautogui.press('down')
                
                if gesture == 'Vol_up_gen':
                    pyautogui.press('volumeup')

                if gesture == 'Vol_down_gen':
                    pyautogui.press('volumedown')

                if gesture == 'Forward':
                    gen_counter += 1
                    if gen_counter % 10 == 0:
                        pyautogui.press('right')
                
                if gesture == 'Backward':
                    gen_counter += 1
                    if gen_counter % 10 == 0:
                        pyautogui.press('left')
                
                if gesture == 'Screen' and before_last != 'Screen':
                    pyautogui.press('f')

                if gesture == 'Nothing':
                    gen_counter = 0



                


                cv.putText(frame, f'{gesture} | {conf: .2f}', (int(width*0.05), int(height*0.07)),
                    cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 1, cv.LINE_AA)

            

    

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
