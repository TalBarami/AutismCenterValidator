from pathlib import Path
from os import path as osp

from skeleton_tools.utils.tools import init_logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESOURCES_ROOT = osp.join(PROJECT_ROOT, 'resources')
logger = init_logger(log_name='annotator', log_path=osp.join(RESOURCES_ROOT, 'logs'))