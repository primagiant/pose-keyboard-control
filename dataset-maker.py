import copy
import csv
import math

import cv2
import mediapipe as mp
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    cap = cv2.VideoCapture(0)
    pose_detector = mp.solutions.pose.Pose()

    mode = 0

    while True:

        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = pose_detector.process(image)
        image.flags.writeable = True

        if results.pose_landmarks is not None:
            landmark_list = get_landmarks(debug_image, results.pose_landmarks)
            logging_csv(number, mode, landmark_list)

        debug_image = draw_info(debug_image, mode, number)
        cv2.imshow('Dataset Maker', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/dataset.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_info(image, mode, number):
    mode_string = ['Normal', 'Pengumpulan Dataset']
    if mode == 0:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
    if mode == 1:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                        cv2.LINE_AA)
    return image


def get_bounding_box(landmarks):
    if landmarks.size == 0:
        return None
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    return x_min, y_min, x_max, y_max


def draw_bounding_box(image, bbox):
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        h, w, _ = image.shape
        cv2.rectangle(image, (int(x_min * w), int(y_min * h)), (int(x_max * w), int(y_max * h)), (0, 255, 0), 2)


def normalize_bounding_box(bbox, image_shape):
    h, w, _ = image_shape
    x_min, y_min, x_max, y_max = bbox
    return x_min / w, y_min / h, x_max / w, y_max / h


def get_landmarks(image, poses):
    landmark_list = []

    landmarks = poses.landmark
    frame_height, frame_width, _ = image.shape

    # get x_min, y_min, x_max, y_max
    bbox = get_bounding_box(np.array([[landmark.x, landmark.y] for landmark in landmarks]))
    draw_bounding_box(image, bbox)

    xmin = int(bbox[0] * frame_width)
    ymin = int(bbox[1] * frame_height)

    # iterasi setiap landmark pada tangan
    for idh, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        landmark_list.extend([x - xmin, y - ymin])

        # Draw circles on specific landmarks
        # if idh in [11, 12, 13, 14, 15, 16]:  # IDs for specific landmarks
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

    return landmark_list


if __name__ == '__main__':
    main()
