from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
def linear_fit_and_plot(x, y, save_fig_path=None):
    # Define the system of equations for the coefficients of the linear function y = mx + b.
    A = np.vstack([x, np.ones(len(x))]).T
    b = np.array(y)
    
    # Solve the system of equations to obtain the coefficients of the linear function.
    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Define a function that returns the value of the linear function y = mx + b.
    def fitted_func(x):
        return coeffs[0] * x + coeffs[1]
    
    # Generate points on the fitted curve for plotting.
    x_fit = np.linspace(0, max(x), 100)
    y_fit = fitted_func(x_fit)
    
    # Plot the input points and the fitted curve.
    plt.plot(x, y, 'ro')
    plt.plot(x_fit, y_fit, 'b-')
    
    # Set the axis labels and display the plot.
    plt.xlabel('circuit scale for the vd part')
    plt.ylabel('Y')
    
    
    # Save the figure if save_fig_path is specified.
    if save_fig_path is not None:
        plt.savefig(save_fig_path)

    plt.show()
    
    return fitted_func
