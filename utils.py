import csv
import cv2 as cv
import numpy as np
import torch
import pyautogui

# track command history to avoid sending same command repeatedly
# example: multiple subsequent clicks as long as the gesture is recognized
def track_history(history, command, last_n = 3):
    if len(history) < last_n:
        history.append(command)
    else:
        history.popleft()
        history.append(command)
    return history

# Running mode (normal or data logging)
def select_mode(key, mode):
    if key == ord('n'):  # normal mode
        mode = 0
    if key == ord('r'):  # record data
        mode = 1
    return mode


# ID associated to each hand gesture
def get_class_id(key):
    class_id = -1

    if 48 <= key <= 57:  # numeric keys
        class_id = key - 48
    if key == 65: # capital A
        class_id = 10
    if key == 66: # capital B
        class_id = 11 
    return class_id


# record landmarks
def logging_csv(class_id, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= class_id <= 11):
        csv_path = 'data/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([class_id, *landmark_list])


# Annotate frame
def draw_info(frame, mode, class_id):
    if mode == 1:

        cv.putText(frame, 'Logging Mode', (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

        if class_id != -1:
            cv.putText(frame, "Class ID:" + str(class_id), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                       cv.LINE_AA)

    return frame


def calc_landmark_coordinates(frame, landmarks):
    frame_height, frame_width = frame.shape[:2]

    landmark_coordinates = []

    # Keypoint
    for landmark in landmarks.landmark:
        landmark_x = int(landmark.x * frame_width)
        landmark_y = int(landmark.y * frame_height)

        landmark_coordinates.append([landmark_x, landmark_y])

    return landmark_coordinates


# preprocess coordinates
def pre_process_landmark(landmark_list):
    coordinates = np.array(landmark_list)

    # relative coordinates to wrist keypoints
    wrist_coordinates = coordinates[0]
    relatives = coordinates - wrist_coordinates

    # Convert back to 1D array
    flattened = relatives.flatten()

    # Normalize between (-1, 1)
    max_value = np.abs(flattened).max()

    normalized = flattened/max_value

    return normalized


# prediction

def predict(landmarks, model):

    model.eval()
    with torch.no_grad():
        landmarks = torch.tensor(landmarks.reshape(1, -1), dtype=torch.float)
        # confidence = torch.exp(model(landmarks))
        confidence = model(landmarks)
    # return torch.argmax(model(landmarks), dim=1).item()
    conf, pred = torch.max(confidence, dim=1)
    return conf.item(), pred.item()


# Draw detection zone
def detection_zone(frame, draw_zone = True):
    frame_height, frame_width = frame.shape[:2]
    zone_height, zone_width = frame_height // 2 , frame_width // 3
    zone = np.array([(zone_width, 0), (zone_width*2, 0), (zone_width*2, zone_height), (zone_width, zone_height)])
    
    if draw_zone:
        cv.rectangle(frame, (zone_width, 0), (zone_width*2, zone_height), (0, 255, 0), 2)

    return zone

    



# Virtual mouse
pyautogui.FAILSAFE = False
def mouse_movement_area(coordinates, detection_zone, screen_size):
    """
    map detection zone to the screen size
    """
    x, y = coordinates
    zone_width, zone_height = detection_zone[0][0], detection_zone[2][1]
    offset_x , offset_y = zone_width // 10, zone_height // 10

    screen_width, screen_height = screen_size
    new_x = np.interp(x, (zone_width + offset_x, zone_width*2 - offset_x), (0, screen_width))
    new_y = np.interp(y, (offset_y, zone_height - offset_y), (0, screen_height))
    return new_x, new_y


def mouse_move(x, y):
        pyautogui.moveTo(x, y)

def calc_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# def mouse_left_click(index_finger_tip_y, index_finger_dip_y, time_after_click = 0):
#     if (index_finger_tip_y >= index_finger_dip_y) and index_finger_dip_y > 0:
#         pyautogui.click()
#         print('left click')

# def mouse_right_click(thump_tip_x, thump_ip_x):
#     if (thump_tip_x >= thump_ip_x):
#         # pyautogui.rightClick()
#         print('right click')


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks
    A = calc_distance(eye[1], eye[5])
    B = calc_distance(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark
    C = calc_distance(eye[0], eye[3])

	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


