import json
from json import JSONDecodeError

import cv2
import matplotlib.colors as mcolors
import numpy as np
from enum import Enum


class Status(Enum):
    NONE = 'NONE'
    OK = 'OK'
    NO_ACTION_OBSERVED = 'NO_ACTION_OBSERVED'
    NO_SKELETON = 'NO_SKELETON'
    WRONG_LABEL = 'WRONG_LABEL'
    NO_SKELETON_WRONG_LABEL = 'NO_SKELETON + WRONG_LABEL'
    SKIP = 'SKIP'


class OpenPoseLayout:
    def __init__(self, name, center, map, pairs):
        self.name = name
        self.pose_pairs = pairs
        self.center = center
        self.len = len(map)
        self.pose_map = {}
        for k, v in map.items():
            self.pose_map[k] = v
            self.pose_map[v] = k

    def __len__(self):
        return self.len

    def joint(self, i):
        return self.pose_map[i]

    def pairs(self):
        return self.pose_pairs


def draw_skeleton(image, pose, score, skeleton_layout, pid=None, color=None, join_emphasize=None, bbox=True, epsilon=1e-3):
    if color is None:
        if pid is not None:
            color = tuple(reversed(COLORS[pid % len(COLORS)]['value']))
            # color = tuple([int(x) * 255 for x in format(pid % 8, '03b')])
        else:
            color = (0, 0, 255)
    for (v1, v2) in skeleton_layout.pose_pairs:
        if score[v1] > epsilon and score[v2] > epsilon:
            cv2.line(image, pose[v1], pose[v2], color, thickness=3)
    for i, (x, y) in enumerate(pose):
        if score[i] > epsilon:
            joint_size = join_emphasize[i] if join_emphasize else 3
            cv2.circle(image, (x, y), joint_size, (0, 0, 255), thickness=2)
    pose = np.array(pose).T
    if bbox is not None:
        bbox = bounding_box(pose, score)
        bbox = (bbox[0]['min'], bbox[1]['min']), (bbox[0]['max'], bbox[1]['max'])
        cv2.rectangle(image, bbox[0], bbox[1], (255, 255, 255), thickness=1)
    if pid is not None:
        x = pose[0][score > EPSILON]
        y = pose[1][score > EPSILON]
        x_center = x.mean() * 0.975
        y_center = y.min() * 0.9
        cv2.putText(image, str(pid), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)


def draw_json_skeletons(frame, skeletons, resolution, skeleton_layout, normalized=False):
    width, height = resolution
    for i, s in enumerate(skeletons):
        pid = s['person_id'] if 'person_id' in s.keys() else None
        x = (np.array(s['pose'][0::2]) * (width if normalized else 1)).astype(int)
        y = (np.array(s['pose'][1::2]) * (height if normalized else 1)).astype(int)
        c = np.array(s['score'])
        draw_skeleton(frame, [t for t in zip(x, y)], c, skeleton_layout, pid=pid)


def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    resolution = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return resolution, fps, frame_count


def bounding_box(pose, score):
    x, y = pose[0][score > EPSILON], pose[1][score > EPSILON]
    if not any(x):
        x = np.array([0])
    if not any(y):
        y = np.array([0])
    box = {
        0: {
            'min': np.min(x),
            'max': np.max(x),
        },
        1: {
            'min': np.min(y),
            'max': np.max(y)
        }
    }
    return box


def read_json(file):
    try:
        with open(file, 'r') as j:
            return json.loads(j.read())
    except (OSError, UnicodeDecodeError, JSONDecodeError) as e:
        print(f'Error while reading {file}: {e}')
        raise e


EPSILON = 1e-4

REAL_DATA_MOVEMENTS = ['Hand flapping', 'Tapping', 'Other', 'Clapping', 'Fingers', 'Body rocking',
                       'Tremor', 'Spinning in circle', 'Toe walking', 'Back and forth', 'Head movement',
                       'Playing with object', 'Jumping in place', 'NoAction']

BODY_25_LAYOUT = OpenPoseLayout(
    'BODY_25',
    1,
    {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "MidHip",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "LBigToe",
        20: "LSmallToe",
        21: "LHeel",
        22: "RBigToe",
        23: "RSmallToe",
        24: "RHeel",
        25: "Background"
    }, [(0, 1), (1, 8),
        (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11), (11, 22), (11, 24), (22, 23),
        (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20),
        (0, 15), (15, 17), (0, 16), (16, 18)]
)

COLORS = [{'name': k.split(':')[1], 'value': tuple(int(v.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))} for k, v in mcolors.TABLEAU_COLORS.items()]
