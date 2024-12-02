import cv2
import numpy as np
from taltools.cv.images import COLORS
from taltools.io.files import read_pkl
from ultralytics.utils.plotting import Annotator

from validator.video_player import VideoPlayer


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
