import copy
import math
import time

import cv2
import mediapipe as mp

from model.pose_classifier_with_angle import PoseClassifierWithAngle
from utils.keyboard_controls import PressKey, ReleaseKey

class_name = ['l-down', 'l-left', 'l-right', 'l-up', 'left', 'overhead', 'right', 'stand-up', 't-pose']

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    cap = cv2.VideoCapture(0)
    pose_detector = mp.solutions.pose.Pose()

    mode = 0

    pose_classifier = PoseClassifierWithAngle()

    while True:

        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

        mode = select_mode(key, mode)

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
            pose_cls_id = pose_classifier(landmark_list)

            if mode == 0:
                if pose_cls_id == 8: # START
                    mode = 1

            if mode == 1:
                if pose_cls_id == 0: # S
                    PressKey(0x1F)
                    ReleaseKey(0x1F)
                    time.sleep(0.3)

                elif pose_cls_id == 1: # A
                    PressKey(0x1E)
                    ReleaseKey(0x1E)
                    time.sleep(0.3)

                elif pose_cls_id == 2: # D
                    PressKey(0x20)
                    ReleaseKey(0x20)
                    time.sleep(0.3)

                elif pose_cls_id == 3: # W
                    PressKey(0x11)
                    ReleaseKey(0x11)
                    time.sleep(0.3)

                # if pose_cls_id == 4: # A
                #     PressKey(0x1E)
                #     ReleaseKey(0x1E)
                #     time.sleep(0.2)

                # if pose_cls_id == 5: # W
                #     PressKey(0x11)
                #     ReleaseKey(0x11)
                #     time.sleep(0.2)

                # if pose_cls_id == 6: # D
                #     PressKey(0x20)
                #     ReleaseKey(0x20)
                #     time.sleep(0.2)

                if pose_cls_id == 7: # STOP
                    mode = 0


        debug_image = draw_info(debug_image, mode)
        cv2.imshow('Dataset Maker', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()


def get_landmarks(image, poses):
    landmark_list = []
    landmark_pose_list = []

    landmarks = poses.landmark
    frame_height, frame_width, _ = image.shape

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

    return landmark_pose_list


def calc_angle(landmark1, landmark2, landmark3):
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return int(angle)


def draw_info(image, mode):
    mode_string = ['Virtual Keybind Not Running', 'Virtual Keybind Running']
    if mode == 0:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
    if mode == 1:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
    return image


def select_mode(key, mode):
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return mode


if __name__ == '__main__':
    main()
