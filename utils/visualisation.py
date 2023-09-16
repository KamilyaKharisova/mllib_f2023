import plotly.graph_objects as go
import numpy as np

class Visualisation():
    """
        A class for visualizing results using plotly.

        This class provides methods for creating various visual representations,
        such as plots, charts, etc., based on the plotly library.


        TODO:
            - Write a function `compare_model_predictions` in the class `Visualisation` that takes in:
                  1. x_values: Array-like data for the x-axis - our inputs .
                  2. y_values_list: A list of array-like data containing predictions from different models.
                  3. y_actual: Array-like data containing actual y-values - our targets.
                  4. title: A string to be used as the plot title.
              The function should generate a plot comparing model predictions from gradient descent and normal equation methods against actual data.


        Example:
            To create a simple line chart with additional traces and a title using plotly:

            >>> import numpy as np
            >>> x = np.arange(10)
            >>> y1 = np.sin(x)
            >>> y2 = np.cos(x)

            # Create an initial plot with the sine curve
            >>> fig = go.Figure(data=go.Scatter(x=x, y=y1, mode='lines', name='sin(x)'))

            # Add a trace for the cosine curve
            >>> fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='cos(x)'))

            # Add a title to the figure
            >>> fig.update_layout(title='Sine and Cosine Curves')

            # Display the figure
            >>> fig.show()
        """

    pass
