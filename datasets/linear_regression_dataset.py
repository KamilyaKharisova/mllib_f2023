import numpy as np
from utils.common_functions import read_dataframe_file
from easydict import EasyDict
from datasets.base_dataset import BaseDataset
from utils.enums import SetType

class LinRegDataset(BaseDataset):

    def __init__(self, cfg: EasyDict):
        super(LinRegDataset, self).__init__(cfg.train_set_percent,cfg.valid_set_percent)

        advertising_dataframe = read_dataframe_file(cfg.dataframe_path)

        # define properties
        self.inputs = np.asarray(advertising_dataframe['inputs'])
        self.targets = np.asarray(advertising_dataframe['targets'])

        # divide into sets
        self._divide_into_sets()

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if isinstance(value,np.ndarray):
            self._inputs = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    def __call__(self, set_type: SetType) -> dict:

        return {'inputs' : getattr(self,f'inputs_{set_type.name}'),
                'targets': getattr(self,f'targets_{set_type.name}'),}

if __name__ == '__main__':
    from configs.linear_regression_cfg import cfg

    lin_reg_dataset = LinRegDataset(cfg)