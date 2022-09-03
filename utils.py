import csv
import cv2 as cv
import numpy as np
import torch
from collections import deque
import pyautogui
pyautogui.FAILSAFE = False


# track command history to avoid sending same command repeatedly
# example: multiple subsequent clicks as long as the gesture is recognized
def track_history(history, item, max_n = 3):
    if len(history) < max_n:
        history.append(item)
    else:
        history.popleft()
        history.append(item)
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
    if key == 67: # capital C
        class_id = 12
    return class_id


# record landmarks
def logging_csv(class_id, mode, features):
    if mode == 0:
        pass
    if mode == 1 and (0 <= class_id <= 12):
        csv_path = 'data/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([class_id, *features])


# Annotate frame
def draw_info(frame, mode, class_id):
    if mode == 1:

        cv.putText(frame, 'Logging Mode', (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

        if class_id != -1:
            cv.putText(frame, "Class ID:" + str(class_id), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                       cv.LINE_AA)  


# Extract x, y coordinates of the landmarks
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

    # Normalize between (-1, 1), exclude wrist coordinates(always 0)
    max_value = np.abs(flattened).max()

    normalized = flattened[2:]/max_value
    
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


# Compute and draw the detection zone in which hand gestures
# are translated to commands.
# Also determine the area in which mouse movements are possible
def det_mouse_zones(frame, draw_det_zone = True, draw_mouse_zone = True, 
                    horizontal_shift = 0.05, vertical_shift = 0.10, mouse_shift = 10):
    # Detection zone
    frame_height, frame_width = frame.shape[:2]
    det_zone_height, det_zone_width = frame_height // 2 , frame_width // 3
    xd, yd = int(horizontal_shift*frame_width) , int(vertical_shift*frame_height)
    det_zone = np.array([(xd, yd), (xd + det_zone_width, yd), (xd + det_zone_width, yd + det_zone_height), 
                    (xd, yd + det_zone_height)])
    
    if draw_det_zone:
        cv.rectangle(frame, (xd, yd), (xd + det_zone_width, yd + det_zone_height), (255, 0, 255), 3)

    # Mouse zone (inside detection zone)
    m_zone = np.array([(xd + mouse_shift, yd + mouse_shift), (xd + det_zone_width - mouse_shift, yd + mouse_shift), 
                        (xd + det_zone_width - mouse_shift, yd + det_zone_height - mouse_shift), 
                        (xd + mouse_shift, yd + det_zone_height - mouse_shift)])

    if draw_mouse_zone:
        cv.rectangle(frame, (xd + mouse_shift, yd + mouse_shift), (xd + det_zone_width - mouse_shift, yd + det_zone_height - mouse_shift), 
                        (0, 255, 0), 2)
 
    return det_zone, m_zone




def mouse_zone_to_screen(coordinates, mouse_zone, screen_size):
    """
    Convert coordinates in such a way that the mouse_zone maps to 
    the screen_size
    """
    x, y = coordinates
    screen_width, screen_height = screen_size
    
    new_x = np.interp(x, (mouse_zone[0][0], mouse_zone[2][0]), (0, screen_width))
    new_y = np.interp(y, (mouse_zone[0][1], mouse_zone[2][1]), (0, screen_height))
    return new_x, new_y


def calc_distance(pt1, pt2):
    # compute euclidian distance between two points pt1 and pt2
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def get_all_distances(pts_list):

    # Compute all distances between pts in a given list (pts_list)
    pts = deque(pts_list)
    distances = deque()
    while len(pts) > 1:
        pt1 = pts.popleft()
        distances.extend( [calc_distance(pt1, pt2) for pt2 in pts] )
    
    return distances

def normalize_distances(d0, distances_list):
    # normalize distances in distances_list by d0
    return np.array(distances_list) / d0
   

def show_save_info(frame, save_icon, nb_saved , vertical_shift = 40, horintal_shift = 150):
    frame_w = frame.shape[1]
    icon_h, icon_w = save_icon.shape[:2]

    # Overlay icon image on frame
    frame[vertical_shift:vertical_shift + icon_h,
            frame_w - horintal_shift:frame_w - horintal_shift + icon_w] = save_icon

    # Show the number of links saved
    text_y = vertical_shift + icon_h//2
    text_x = frame_w - horintal_shift + icon_w + 10
    cv.putText(frame, str(nb_saved), (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 
                1 , (255, 0, 0), 2, cv.LINE_AA)



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


