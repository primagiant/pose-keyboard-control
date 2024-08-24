import csv
import math
import os

import cv2
import mediapipe as mp
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    pose_detector = mp.solutions.pose.Pose()
    d_foldername = './dataset'
    for root, dirs, files in os.walk(d_foldername):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), d_foldername)
            pose = root.split("\\")[1]
            image = cv2.imread(os.path.join(d_foldername, relative_path))
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(image)
            if results.pose_landmarks is not None:
                landmark_list = get_landmarks(image, results.pose_landmarks)
                csv_path = 'model/dataset_angle_new.csv'
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([pose, *landmark_list])
        print("a block of file have been transform into dataset")
    print("Completed")


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/dataset_angle_new.csv'
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
    landmark_pose_list = []

    landmarks = poses.landmark
    frame_height, frame_width, _ = image.shape

    # get x_min, y_min, x_max, y_max
    bbox = get_bounding_box(np.array([[landmark.x, landmark.y] for landmark in landmarks]))
    draw_bounding_box(image, bbox)

    # iterasi setiap landmark pada pose
    for idh, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        # Draw circles on specific landmarks
        if idh in [11, 12, 13, 14, 15, 16, 23, 24]:  # IDs for specific landmarks
            landmark_list.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

    landmark_pose_list.append(calc_angle(landmark_list[4], landmark_list[2], landmark_list[0]))  # right elbow angle
    landmark_pose_list.append(calc_angle(landmark_list[2], landmark_list[0], landmark_list[6]))  # right shoulder angle

    landmark_pose_list.append(calc_angle(landmark_list[5], landmark_list[3], landmark_list[1]))  # left elbow angle
    landmark_pose_list.append(calc_angle(landmark_list[3], landmark_list[1], landmark_list[7]))  # left shoulder angle

    # Display
    # cv2.putText(image, "REA:" + str(landmark_pose_list[0]), (10, frame_height - 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
    # cv2.putText(image, "RSA:" + str(landmark_pose_list[1]), (10, frame_height - 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
    # cv2.putText(image, "LEA:" + str(landmark_pose_list[2]), (10, frame_height - 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
    # cv2.putText(image, "LSA:" + str(landmark_pose_list[3]), (10, frame_height - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)

    return landmark_pose_list


def calc_angle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return int(angle)


if __name__ == '__main__':
    main()
