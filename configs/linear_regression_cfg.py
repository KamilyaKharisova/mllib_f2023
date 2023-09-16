from easydict import EasyDict
from utils.enums import TrainType

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = '/Users/kamilyakharisova/Documents/homeworks_datasets/linear_regression_dataset.csv'

# cfg.base_functions contains callable functions to transform input features.
# E.g., for polynomial regression: [lambda x: x, lambda x: x**2]
# TODO You should populate this list with suitable functions based on the requirements.

cfg.base_functions = []

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = 100
