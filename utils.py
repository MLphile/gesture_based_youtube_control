import csv
import cv2 as cv
import numpy as np
import torch
import pyautogui


# Running mode (normal or data logging)
def select_mode(key, mode):
    class_id = -1
    if 48 <= key <= 57:  # class_id
        class_id = key - 48
    if key == ord('n'):  # normal mode
        mode = 0
    if key == ord('r'):  # record data
        mode = 1
    return class_id, mode


# record landmarks
def logging_csv(class_id, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= class_id <= 9):
        csv_path = 'data/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([class_id, *landmark_list])
    return


# Annotate frame
def draw_info(frame, mode, class_id):
    if mode == 1:

        cv.putText(frame, 'Logging Mode', (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

        if class_id != -1:
            cv.putText(frame, "NUM:" + str(class_id), (10, 110),
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

    # return normalized
    return normalized


# prediction

def predict(landmarks, model):

    model.eval()
    with torch.no_grad():
        landmarks = torch.tensor(landmarks.reshape(1, -1), dtype=torch.float)
    return torch.argmax(model(landmarks), dim=1).item()




# Virtual mouse
pyautogui.FAILSAFE = False

def frame_to_screen(coordinates, frame_size, screen_size):
    x, y = coordinates
    frame_height, frame_width = frame_size
    screen_width, screen_height = screen_size
    x = x * screen_width / frame_width
    y = y * screen_height / frame_height
    
    return x, y


def move_mouse(x, y, mouse_mode = False):
    if mouse_mode == True:
        pyautogui.moveTo(x, y)

