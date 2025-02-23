import os
from enum import Enum
from time import sleep
from os import path as osp

import cv2
import numpy as np
import seaborn as sns
from taltools.io.files import read_json, write_json

from validator.constants import RESOURCES_ROOT


class Resolution(Enum):
    AUTO = 'AUTO'
    MANUAL = 'MANUAL'


class VideoPlayer:
    def __init__(self, cfg_path, cfg=None):
        self.cfg_path = cfg_path
        self.cfg = read_json(cfg_path)
        self.resolution_method = Resolution.AUTO

        self.speed = self.cfg['speed']
        self.resolution = self.cfg['resolution']

        self.set_speed(self.speed)
        self.set_resolution(*self.resolution)

        self.window_name = 'Video Player'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, 0, 0)

    def update_cfg(self):
        self.cfg['speed'] = self.speed
        self.cfg['resolution'] = self.resolution
        write_json(self.cfg, self.cfg_path)

    def set_speed(self, i):
        try:
            i = float(i)
            if i < 0.5:
                return
            self.speed = i
        except ValueError:
            pass

    def set_resolution(self, width, height):
        self.resolution = int(width), int(height)
        self.resolution_method = Resolution.MANUAL if np.sum(self.resolution) > 0 else Resolution.AUTO

    def get_resolution(self, resolution):
        return self.resolution if self.resolution_method == Resolution.MANUAL else resolution

    def reset(self):
        return

    def play(self, video_name, frames, done=None, counter_text=None):
        fps = 30
        org_resolution = frames.shape[1:3]
        i = 0
        delay = int((1000 / fps / self.speed))
        while True:
            frame = frames[i]
            resolution = self.get_resolution(org_resolution)
            if resolution != org_resolution:
                frame = cv2.resize(frame, resolution)
            if counter_text:
                self.add_text(frame, counter_text, 0.05, 0.05, 1, (100, 30, 255), 2)
            if i < 10:
                self.add_text(frame, 'Reset', 0.5, 0.5, 3, (0, 0, 255), 5)
            self.add_text(frame, f'{i}/{len(frames)}', 0.05, 0.95, 1, (100, 30, 255), 2)
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(delay)
            i += 1
            if i >= len(frames):
                i = 0
            if done and done(): # or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        cv2.destroyAllWindows()

    def add_text(self, frame, text, loc_x, loc_y, fontsize, color, thickness):
        cv2.putText(frame, text,
                    (int(frame.shape[1] * loc_x), int(frame.shape[0] * loc_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontsize,
                    color,
                    thickness)

    def gen_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            i += 1
        cap.release()
        return np.array(frames)

class AssessmentVideoPlayer(VideoPlayer):
    def gen_video(self, video_paths):
        rows, cols = (2, 3) if len(video_paths > 4) else (2, 2)
        all_vids = []
        for video_path in video_paths:
            frames = []
            cap = cv2.VideoCapture(video_path)
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                i += 1
            cap.release()
            all_vids.append(frames)
        combined_frames = []
        F = min([len(vid) for vid in all_vids])
        for frame in range(F):
            current_frames = [vid[frame] for vid in all_vids]
            grid_rows = [
                np.hstack(current_frames[i * cols:(i + 1) * cols]) for i in range(rows)
            ]
            combined_frame = np.vstack(grid_rows)
            combined_frames.append(combined_frame)

        return np.array(combined_frames)




# class SkeletonPlayer(VideoPlayer):
#     def gen_video(self, video_path, skeleton):
#         cap = cv2.VideoCapture(video_path)
#         fps, width, height, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if type(skeleton) == str:
#             skeleton = read_pkl(skeleton)
#         M, T = skeleton['keypoint'].shape[:2]
#         pids = skeleton['person_ids'].astype(int)
#         pids[pids >= len(COLORS)] = -1
#         skeleton['landmarks'] = skeleton['keypoint'].astype(int)
#         skeleton['landmarks_scores'] = skeleton['keypoint_score']
#         # skeleton['person_colors'] = np.array([[self.cmap[pids[i, t]] for t in range(T)] for i in range(M)]) * 255
#         skeleton['person_colors'] = np.array([[COLORS[pids[i, t]]['value'] for t in range(T)] for i in range(M)])
#         painter = GlobalPainter(GraphPainter(COCO_LAYOUT, tracking=True, limbs=True))
#
#         frames = []
#         i = 0
#         while i < frame_count:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = painter(frame, skeleton, i)
#             if i < 10:
#                 self.add_text(frame, 'Reset', 0.5, 0.5, 3, (0, 0, 255), 5)
#             self.add_text(frame, f'{i}/{frame_count}', 0.05, 0.95, 1, (100, 30, 255), 2)
#             frames.append(frame)
#             i += 1
#         cap.release()
#         return frames
