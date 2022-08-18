from model_architecture import model
import pandas as pd
import mediapipe as mp
from utils import *
from collections import deque

# Camera resolution
width = 1280//4
height = 720//4


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
model.load_state_dict(torch.load('model.pth'))

# Load Label
labels = pd.read_csv('data/label.csv', header=None).values.flatten().tolist()

# mode normal (don't record data for training)
mode = 0  

# confidence threshold(required to send most of the commands)
conf_threshold = 0.8

# command history
history = deque([])

while True:
    # counter += 1
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

    results = hands.process(frame_rgb)

    # if detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # get landmarks coordinates
            coordinates_list = calc_landmark_coordinates(frame, hand_landmarks)

            # Conversion to relative coordinates and normalized coordinates
            preprocessed = pre_process_landmark(coordinates_list)

            # print(coordinates_list[8])

            # check if middle finger mcp is inside the detection zone for command execution
            if cv.pointPolygonTest(zone, coordinates_list[9], False) == 1: 
                
                # inference
                # pred = predict(preprocessed, model)
                # gesture = labels[pred]

                # activate/deactivate mouse mode
                gesture = 'mouse_on'
                
                
                # keyboard

                # mouse
                
                if  gesture == 'mouse_on':
                    conf, pred = predict(preprocessed, model)
                    gesture = labels[pred]
                    
                    history = track_history(history, gesture)
                    print(history)
                    if gesture == 'Forward' or gesture == 'Backward':
                        x, y = mouse_movement_area(coordinates_list[9], zone, screen_size)
                        
                        clocX = plocX + (x - plocX) / smooth_factor
                        clocY = plocY + (y - plocY) / smooth_factor
                        pyautogui.moveTo(clocX, clocY)

                        plocX, plocY = clocX, clocY

                    d1 = calc_distance(coordinates_list[4], coordinates_list[8])
                    d2 = calc_distance(coordinates_list[4], coordinates_list[12])
                    
                    if conf >= conf_threshold:
                        before_last = history[len(history) - 2]
                        if gesture == 'Stop' and before_last != 'Stop':
                            pyautogui.click()

                        if gesture == 'Pause' and before_last != 'Pause':
                            pyautogui.rightClick()



                

                # cv.putText(frame, f'COMMAND: {command}', (int(width*0.05), int(height*0.1)),
                #            cv.FONT_HERSHEY_COMPLEX, 1, (22, 69, 22), 3, cv.LINE_AA)

            # Write to the dataset file (if mode == 1)
            logging_csv(class_id, mode, preprocessed)

        


    frame = draw_info(frame, mode, class_id)
    cv.imshow('', frame)


cap.release()
cv.destroyAllWindows()
