from time import sleep
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import cv2
import pandas as pd
import numpy as np
import os

from os import path

from utils import draw_json_skeletons, Status, REAL_DATA_MOVEMENTS, COLORS, BODY_25_LAYOUT, read_json, get_video_properties


class Validator:

    def __init__(self, videos_path, skeletons_path, out_path, skeleton_layout):
        self.videos_path = videos_path
        self.skeletons_path = skeletons_path
        self.files = [(path.splitext(name)[0], path.join(videos_path, name), path.join(skeletons_path, f'{path.splitext(name)[0]}.json')) for name in os.listdir(videos_path) if path.isfile(path.join(skeletons_path, f'{path.splitext(name)[0]}.json'))]
        self.out_path = out_path
        self.df = pd.read_csv(out_path) if path.isfile(out_path) else pd.DataFrame(columns=['video_name', 'segment_name', 'start_time', 'end_time', 'start_frame', 'end_frame', 'status', 'action', 'child_ids', 'notes'])
        self.executor = ThreadPoolExecutor()
        self.skeleton_layout = skeleton_layout

    def choose(self, msg, lst, offset=1):
        while True:
            print(msg)
            for i, e in enumerate(lst):
                print(f'{i + offset}. {e}')
            result = input()
            # if result.startswith('-'):
            #     return False, -1  # TODO: Global commands
            result = result.split(' ')
            if all(s.isdigit() and (offset <= int(s) < len(lst) + offset) for s in result if s):
                result = [int(s) - offset for s in result]
                return [i for i in result]
            print('Error: Wrong selection.')

    def play(self, skeleton, video_path, label, done):
        cap = cv2.VideoCapture(video_path)
        width, height, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)
        i = 0
        while True:
            ret, frame = cap.read()
            if ret and i < len(skeleton):
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                draw_json_skeletons(frame, skeleton[i]['skeleton'], (width, height), self.skeleton_layout, normalized=True)
                if i < 10:
                    cv2.putText(frame, 'Reset',
                                (int(width * 4 / 10), int(height / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                3,
                                (0, 0, 255),
                                5)

                cv2.putText(frame, label,
                            (int(width * 4 / 10), int(height * 1 / 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 255, 0),
                            5)
                sleep(1 / (2 * fps))
                cv2.imshow('skeleton', frame)
                i += 1
            else:
                i = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

            if done() or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        cap.release()

    def validate(self, base_label, max_id):
        result_labels = [base_label]
        result_cids = [-1]
        result_notes = ''

        opts = [s for s in [Status.OK, Status.NO_ACTION_OBSERVED, Status.NO_SKELETON, Status.WRONG_LABEL, Status.NO_SKELETON_WRONG_LABEL, Status.SKIP]]
        ans = self.choose('Choose status:', [s.value for s in opts])
        status = opts[ans[0]]
        if status == Status.SKIP:
            result_notes = input('Save notes: ')
        if status in [Status.WRONG_LABEL, Status.NO_SKELETON_WRONG_LABEL]:
            ans = self.choose('Choose label(s):', REAL_DATA_MOVEMENTS[:-1])
            result_labels = [REAL_DATA_MOVEMENTS[i] for i in ans]
            status = Status.OK if status == Status.WRONG_LABEL else Status.NO_SKELETON_WRONG_LABEL
        if status == Status.OK:
            result_cids = self.choose('Choose child id(s):', [f'({COLORS[i % len(COLORS)]["name"]})' for i in range(max_id+1)], offset=0)

        return status, result_labels, result_cids, result_notes

    def run(self):
        for name, vpath, spath in self.files:
            _df = self.df[self.df['segment_name'] == name]
            if not _df.empty:  # and if the tagged row is not None / Skip / Whatever I decide
                continue

            skeleton = read_json(spath)['data']
            max_id = np.max([s['person_id'] for v in skeleton for s in v['skeleton']])
            split = name.split('_')
            basename = '_'.join(split[:-3])
            _, fps, _ = get_video_properties(vpath)
            base_label, start_frame, end_frame = split[-3:]
            start_frame, end_frame = int(start_frame), int(end_frame)
            start_time, end_time = int(start_frame / fps), int(end_frame / fps)

            print(f'Validating: {vpath}')
            task = self.executor.submit(lambda: self.validate(base_label, max_id))
            self.play(skeleton, vpath, base_label, task.done)
            status, labels, cids, notes = task.result()
            for label in labels:
                self.df.loc[self.df.shape[0]] = [basename, name, start_time, end_time, start_frame, end_frame, status, label, cids, notes]
                self.df.to_csv(self.out_path, index=False)


# global_cmds = {
#     '-break': True,
#     '-back': True,
#     '-speed': True,
#     '-label': True,
# }
if __name__ == '__main__':
    val = Validator('E:/Data/segmented_videos', 'E:/Data/skeletons/data', 'E:/Data/skeletons/qa.csv', BODY_25_LAYOUT)
    val.run()