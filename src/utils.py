import os
from os import path as osp

from skeleton_tools.openpose_layouts.body import COCO_LAYOUT
from skeleton_tools.skeleton_visualization.numpy_visualizer import MMPoseVisualizer
from skeleton_tools.utils.tools import read_pkl, init_directories


def create_skeleton_videos(root):
    skeletons_dir = osp.join(root, 'skeletons', 'raw')
    videos_dir = osp.join(root, 'segmented_videos')
    files = [(v, f'{osp.splitext(v)[0]}.pkl') for v in os.listdir(videos_dir) if osp.exists(osp.join(skeletons_dir, f'{osp.splitext(v)[0]}.pkl'))]
    out_dir = osp.join(root, 'skeleton_videos')
    init_directories(out_dir)

    vis = MMPoseVisualizer(COCO_LAYOUT)
    for v, s in files:
        vis.create_double_frame_video(osp.join(videos_dir, v), read_pkl(osp.join(skeletons_dir, s)), osp.join(out_dir, v))


if __name__ == '__main__':
    create_skeleton_videos(r'Z:\Users\TalBarami\JORDI_50_vids_benchmark\movement_type_qa')