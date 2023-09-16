from easydict import EasyDict
from utils.enums import TrainType
cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = ''

cfg.base_functions = [] # список lambda функций

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

cfg.train_type = TrainType.gradient_descent
cfg.epoch = 100



