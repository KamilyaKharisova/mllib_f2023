import numpy as np
from configs.linear_regression_cfg import cfg
from utils.enums import TrainType
import sys

class LinearRegression():

    def __init__(self, base_functions: list, learning_rate: float):
        self.weights = None  # init weights using np.random.randn (normal distribution with mean=0 and variance=1).
        self.base_functions = base_functions
        self.learning_rate = learning_rate

    # Methods related to the Normal Equation

    def _pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the pseudoinverse of a matrix using SVD.

        The pseudoinverse (Φ^+) of the design matrix Φ can be computed using the formula:

        Φ^+ = V * Σ^+ * U^T

        Where:
        - U, Σ, and V are the matrices resulting from the SVD of Φ.

        The Σ^+ is computed as:

        Σ'_{i,j} =
        | 1/Σ_{i,j}, if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
        | 0, otherwise

        and then:
        Σ^+ = Σ'^T

        where:
        - ε is the machine epsilon, which can be obtained in Python using:
            ε = sys.float_info.epsilon
        - N is the number of rows in the design matrix.
        - M is the number of base functions (without φ_0(x_i)=1).

        TODO: Implement this method. You can use np.linalg.svd
        """
        pass

    def _calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """Calculate the optimal weights using the normal equation.

            The weights (w) can be computed using the formula:

            w = Φ^+ * t

            Where:
            - Φ^+ is the pseudoinverse of the design matrix and can be defined as:
                Φ^+ = (Φ^T * Φ)^(-1) * Φ^T

            - t is the target vector.

            TODO: Implement this method. Calculate  Φ^+ using _pseudoinverse_matrix function
        """
        pass

    # General methods
    def _plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """Construct the design matrix (Φ) using base functions.

            The structure of the matrix Φ is as follows:

            Φ = [ [ φ_0(x_1), φ_1(x_1), ..., φ_M(x_1) ],
                  [ φ_0(x_2), φ_1(x_2), ..., φ_M(x_2) ],
                  ...
                  [ φ_0(x_N), φ_1(x_N), ..., φ_M(x_N) ] ]

            where:
            - x_i denotes the i-th input vector.
            - φ_j(x_i) represents the j-th base function applied to the i-th input vector.
            - M is the total number of base functions (without φ_0(x_i)=1).
            - N is the total number of input vectors.

            TODO: Implement this method using one loop over the base functions.

        """
        pass

    def calculate_model_prediction(self, plan_matrix: np.ndarray) -> np.ndarray:
        """Calculate the predictions of the model.

            The prediction (y_pred) can be computed using the formula:

            y_pred = Φ * w^T

            Where:
            - Φ is the design matrix.
            - w^T is the transpose of the weight vector.

            To compute multiplication in Python using numpy, you can use:
            - `numpy.dot(a, b)`
            OR
            - `a @ b`

        TODO: Implement this method without using loop

        """
        pass

    # Methods related to Gradient Descent
    def _calculate_gradient(self, plan_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the cost function with respect to the weights.

            The gradient of the error with respect to the weights (∆w E) can be computed using the formula:

            ∆w E = (2/N) * Φ^T * (Φ * w - t)

            Where:
            - Φ is the design matrix.
            - w is the weight vector.
            - t is the vector of target values.
            - N is the number of data points.

            This formula represents the partial derivative of the mean squared error with respect to the weights.

            TODO: Implement this method using matrix operations in numpy. a.T - transpose. Do not use loops
            """
        pass

    def calculate_cost_function(self, plan_matrix, targets):
        """Calculate the cost function value for the current weights.

        The cost function E(w) represents the mean squared error and is given by:

        E(w) = (1/N) * ∑(t - Φ * w^T)^2

        Where:
        - Φ is the design matrix.
        - w is the weight vector.
        - t is the vector of target values.
        - N is the number of data points.

        TODO: Implement this method using numpy operations to compute the mean squared error. Do not use loops
        """
        pass

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Train the model using either the normal equation or gradient descent based on the configuration.
        TODO: Complete the training process.
        """
        plan_matrix = self._plan_matrix(inputs)
        if cfg.train_type.value == TrainType.normal_equation.value:
            pseudoinverse_plan_matrix = self._pseudoinverse_matrix(plan_matrix)
            # train process
            self._calculate_weights(pseudoinverse_plan_matrix, targets)
        else:
            """
            At each iteration of gradient descent, the weights are updated using the formula:
        
            w_{k+1} = w_k - γ * ∇_w E(w_k)
        
            Where:
            - w_k is the current weight vector at iteration k.
            - γ is the learning rate, determining the step size in the direction of the negative gradient.
            - ∇_w E(w_k) is the gradient of the cost function E with respect to the weights w at iteration k.
        
            This iterative process aims to find the weights that minimize the cost function E(w).
        """
            for e in cfg.epoch:
                gradient = self._calculate_gradient(plan_matrix,targets)
                # update weights w_{k+1} = w_k - γ * ∇_w E(w_k)

                if e % 10 == 0:
                    # TODO: Print the cost function's value.
                    pass

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self._plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions