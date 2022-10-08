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
    """ Keeps track of up to max_n items stored in history.

    Args:
        history (Deque: Doubly Ended Queue): A data structure where items are stored.
        item (_type_): What to keep track of.
        max_n (int, optional): Maximum number of items to store. Defaults to 3.

    Returns:
        Deque: A list (Deque) of elements to track

    """
    if len(history) < max_n:
        history.append(item)
    else:
        history.popleft()
        history.append(item)
    return history


def select_mode(key, mode):
    """ Actives either the normal mode (0 => nothing happens)
    or the recording mode (1 => saving data)

    Args:
        key (int): An integer value triggered by pressing 'n' (for normal mode) or 'r' (for recording mode)
        mode (int): The current mode

    Returns:
        int: The activated mode
    """
    if key == ord('n'):
        mode = 0
    if key == ord('r'):
        mode = 1
    return mode



def get_class_id(key):
    """ Maps pressed keys on keyboard to a class label that will
    associated to a given gesture.

    Args:
        key (int): A key on the keyboard (currently numeric keys, capital A, B and C)

    Returns:
        int: A class id/label
    """
    class_id = -1

    if 48 <= key <= 57:  # numeric keys
        class_id = key - 48
        
    if 65 <=  key <=  90: # alpha keys (capital letters)
        class_id = key - 55 
    return class_id



def logging_csv(class_id, mode, features, file_path):
    """ Records the gesture label together with features representing that gesture in a csv file.

    Args:
        class_id (int): The label corresponding to a given gesture
        mode (int): Activate the recording mode (1)
        features (Array): An array of numbers that maps to the gesture.
    """
    if mode == 0:
        pass
    if mode == 1 and (0 <= class_id <= 12):
        with open(file_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([class_id, *features])



def draw_info(frame, mode, class_id):
    """Shows info about whether the logging mode is activated
    and which class id has been triggered by pressing on the keyboard
    """
    if mode == 1:

        cv.putText(frame, 'Logging Mode', (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)

        if class_id != -1:
            cv.putText(frame, "Class ID:" + str(class_id), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                       cv.LINE_AA)  



def calc_landmark_coordinates(frame, landmarks):
    """ Converts relative landmark coordinates to actual ones.
    Returns a list of those coordinates (x, y).
    """
    frame_height, frame_width = frame.shape[:2]

    landmark_coordinates = []

    
    for landmark in landmarks.landmark:
        landmark_x = int(landmark.x * frame_width)
        landmark_y = int(landmark.y * frame_height)

        landmark_coordinates.append((landmark_x, landmark_y))

    return landmark_coordinates



def pre_process_landmark(landmark_list):
    """ Preprocesses landmark coordinates through the following steps:
    1. Computes the relative locations all coordinates to the wrist
    2. Flattens the 2D array containing the coordinates into 1D
    3. Normalizes the coordinates with regard to the max value (absolute value)
        Remove wrist coordinates.

    Args:
        landmark_list (List of tuples): List containing the coordinates

    Returns:
        Array: 1D array of coordinates
    """
    coordinates = np.array(landmark_list)

    # relative coordinates to wrist (think of it as:
    # where are the points with reference to the wrist?)
    wrist_coordinates = coordinates[0]
    relatives = coordinates - wrist_coordinates

    # Convert to 1D array
    flattened = relatives.flatten()

    # Normalize between (-1, 1) and exclude wrist coordinates(always 0)
    max_value = np.abs(flattened).max()
    normalized = flattened[2:]/max_value
    
    return normalized


 
def predict(features, model):
    """ Predicts the detected hand gesture and outputs both gesture label and 
    the corresponding probability.

    Args:
        features (1D Array): Values from which to make prediction
        model (Pytorch MLP model): Model for making prediction

    Returns:
        tuple: (probability, prediction)
    """

    model.eval()
    with torch.no_grad():
        features = torch.tensor(features.reshape(1, -1), dtype=torch.float)
        confidence = model(features)
    conf, pred = torch.max(confidence, dim=1)
    return conf.item(), pred.item()



def det_mouse_zones(frame, draw_det_zone = True, draw_mouse_zone = True, 
                    horizontal_shift = 0.05, vertical_shift = 0.10, mouse_shift = 10):
    """ Determines the area (det_zone) where detected hand gestures are mapped to player functionalities.
    Also computes the area (mouse_zone) on the frame that will represent the computer screen. This is the zone
    in which the mouse is moved; it's located inside the det_zone.   

    Args:
        frame (numpy array): Image from captured webcam video
        draw_det_zone (bool, optional): Whether to draw the det_zone on the frame. Defaults to True.
        draw_mouse_zone (bool, optional): Whether to draw the mouse zone. Defaults to True.
        horizontal_shift (float, optional): Controls where the top left x-coordinate of the det_zone is located (proportional to frame width). Defaults to 0.05.
        vertical_shift (float, optional): Controls where the top left y-coordinate of the det_zone is located (proportional to frame height). Defaults to 0.10.
        mouse_shift (int, optional): Controls by how much pixels to shift the det-zone corners, to compute the mouse zone. Defaults to 10.

    Returns:
        tuple: both det_zone and mouse_zone
    """
    # Detection zone
    frame_height, frame_width = frame.shape[:2]
    det_zone_height, det_zone_width = frame_height // 2 , frame_width // 3
    xd, yd = int(horizontal_shift*frame_width) , int(vertical_shift*frame_height)
    det_zone = np.array([(xd, yd), (xd + det_zone_width, yd), (xd + det_zone_width, yd + det_zone_height), 
                    (xd, yd + det_zone_height)])
    
    if draw_det_zone:
        cv.rectangle(frame, (xd, yd), (xd + det_zone_width, yd + det_zone_height), (0, 0, 255), 3)

    # Mouse zone (inside detection zone)
    mouse_zone = np.array([(xd + mouse_shift, yd + mouse_shift), (xd + det_zone_width - mouse_shift, yd + mouse_shift), 
                        (xd + det_zone_width - mouse_shift, yd + det_zone_height - mouse_shift), 
                        (xd + mouse_shift, yd + det_zone_height - mouse_shift)])

    if draw_mouse_zone:
        cv.rectangle(frame, (xd + mouse_shift, yd + mouse_shift), (xd + det_zone_width - mouse_shift, yd + det_zone_height - mouse_shift), 
                        (255, 255, 255), 2)
 
    return det_zone, mouse_zone




def mouse_zone_to_screen(coordinates, mouse_zone):
    """
    Converts coordinates in such a way that the mouse_zone maps to 
    the screen size
    """
    screen_size = pyautogui.size()
    x, y = coordinates
    screen_width, screen_height = screen_size
    
    new_x = np.interp(x, (mouse_zone[0][0], mouse_zone[2][0]), (0, screen_width))
    new_y = np.interp(y, (mouse_zone[0][1], mouse_zone[2][1]), (0, screen_height))
    return new_x, new_y


def calc_distance(pt1, pt2):
    """
    Computes and returns the euclidian distance between two points pt1(x1, y1) and pt2(x2, y2)
    """
    return np.linalg.norm(np.array(pt1) - np.array(pt2))


def get_all_distances(pts_list):
    """
    Computes and returns distances between all the points in a given list.
    Points are tuple (or array-like) of x and y coordinates.
    """
    pts = deque(pts_list)
    distances = deque()
    while len(pts) > 1:
        pt1 = pts.popleft()
        distances.extend( [calc_distance(pt1, pt2) for pt2 in pts] )
    return distances

def normalize_distances(d0, distances_list):
    """
    Works out normalized distances and returns an array of those.
    """
    return np.array(distances_list) / d0



def eye_aspect_ratio(eye):
    """
    Computes and return the eye aspect ratio (ear)
    """
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks
    A = calc_distance(eye[1], eye[5])
    B = calc_distance(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmarks
    C = calc_distance(eye[0], eye[3])

	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear
