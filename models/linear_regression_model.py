import numpy as np


class LinearRegression():

    def __init__(self, base_functions: list):
        self.weights = None  # TODO init weights using np.random.randn (normal distribution with mean=0 and variance=1).
        self.base_functions = base_functions


    def _pseudoinverse_matrix(self,matrix: np.ndarray) -> np.ndarray:
        """calculate pseudoinverse matrix using SVD. Not this homework """
        pass

    def _plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        #TODO build Plan matrix using list of lambda functions defined in config. Use only one loop (for base_functions).
        pass

    def _calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """calculate weights of the model using formula from the lecture. Not this homework
        """
        pass

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        # TODO calculate prediction of the model (y) using formula from the lecture.
        # np.dot, a@b
        pass

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        pass
        # # prepare data
        # plan_matrix = self.__plan_matrix(inputs)
        # pseudoinverse_plan_matrix = self.__pseudoinverse_matrix(plan_matrix)
        #
        # # train process
        # self.__calculate_weights(pseudoinverse_plan_matrix, targets)


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self._plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions