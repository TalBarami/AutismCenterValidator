import os
from enum import Enum
from time import sleep
from os import path as osp

import cv2
import numpy as np
import seaborn as sns
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.base_painters import GlobalPainter
from skeleton_tools.skeleton_visualization.paint_components.frame_painters.local_painters import GraphPainter
from skeleton_tools.utils.constants import COLORS

from skeleton_tools.utils.tools import write_json, read_json, read_pkl

from validator.constants import RESOURCES_ROOT


class Resolution(Enum):
    AUTO = 'AUTO'
    MANUAL = 'MANUAL'


class VideoPlayer:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.cfg = read_json(cfg_path)
        self.resolution_method = Resolution.AUTO
        self.cmap = sns.color_palette('bright', 25)

        self.speed = self.cfg['speed']
        self.resolution = self.cfg['resolution']

        self.set_speed(self.speed)
        self.set_resolution(*self.resolution)

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
            cv2.imshow(video_name, frame)
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

class YOLOPlayer(VideoPlayer):
    def gen_video(self, video_path, tracking):
        cap = cv2.VideoCapture(video_path)
        if type(tracking) == str:
            tracking = read_pkl(tracking)
        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotator = Annotator(frame)
            boxes = [b for b in tracking['data'][i]['boxes'] if b.cls == 0]
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                pid = int(box.id[0]) if box.id is not None else -1
                color = COLORS[pid % len(COLORS)]['value']
                color = (color[2], color[1], color[0])
                annotator.box_label(b, f'{tracking["names"][int(c)]} {pid}', color=color)
            frame = annotator.result()
            frames.append(frame)
            i += 1
        cap.release()
        return np.array(frames)



class SkeletonPlayer(VideoPlayer):
    def gen_video(self, video_path, skeleton):
        cap = cv2.VideoCapture(video_path)
        fps, width, height, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if type(skeleton) == str:
            skeleton = read_pkl(skeleton)
        M, T = skeleton['keypoint'].shape[:2]
        pids = skeleton['person_ids'].astype(int)
        pids[pids >= len(COLORS)] = -1
        skeleton['landmarks'] = skeleton['keypoint'].astype(int)
        skeleton['landmarks_scores'] = skeleton['keypoint_score']
        # skeleton['person_colors'] = np.array([[self.cmap[pids[i, t]] for t in range(T)] for i in range(M)]) * 255
        skeleton['person_colors'] = np.array([[COLORS[pids[i, t]]['value'] for t in range(T)] for i in range(M)])
        painter = GlobalPainter(GraphPainter(COCO_LAYOUT, tracking=True, limbs=True))

        frames = []
        i = 0
        while i < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            frame = painter(frame, skeleton, i)
            if i < 10:
                self.add_text(frame, 'Reset', 0.5, 0.5, 3, (0, 0, 255), 5)
            self.add_text(frame, f'{i}/{frame_count}', 0.05, 0.95, 1, (100, 30, 255), 2)
            frames.append(frame)
            i += 1
        cap.release()
        return frames

if __name__ == '__main__':
    player = VideoPlayer(osp.join(RESOURCES_ROOT, 'config.json'))
    # root = r'Z:\Users\TalBarami\ChildDetect'
    root = r'Z:\Users\TalBarami\ChildDetect\data2'
    f = '704767285_Cognitive_Control_300522_0848_1_7000_7500'
    v = osp.join(root, f'{f}.mp4')
    t = osp.join(root, f'{f}.pkl')
    cap = cv2.VideoCapture(v)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    tracking = read_pkl(t)
    player.play(f, frames, tracking, done=lambda: False, counter_text='xxx')
    # data_dir = osp.join(root, 'data2')
    # draw_dir = osp.join(root, 'draw')
    # fies = [f for f in os.listdir(data_dir) if f.endswith('.mp4')]
    # for f in files:
    #     skel = f.replace('.mp4', '.pkl')
    #     if osp.exists(osp.join(data_dir, f)) and \
    #         osp.exists(osp.join(data_dir, skel)) and not \
    #         osp.exists(osp.join(draw_dir, f)):
    #         try:
    #             frames = player.gen_video(osp.join(data_dir, f), osp.join(data_dir, skel))
    #             writer = cv2.VideoWriter(osp.join(draw_dir, f), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))
    #             for frame in frames:
    #                 writer.write(frame)
    #             writer.release()
    #         except Exception as e:
    #             print(f'Error processing {f}: {e}')l

    # f = '1006723600_PLS_Clinical_090720_0909_1_51500_52000'
    # frames = player.gen_video(osp.join(root, f'{f}.mp4'), osp.join(root, f'{f}.pkl'), 'XXX')
    # player.play('f', frames, done=lambda: False)