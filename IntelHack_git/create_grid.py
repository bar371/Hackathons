import numpy as np


def make_grid(X, Y, n_squares=100):
    """
    Partition geographic locations into a grid
    :param X: x coordinates of each data point, as a <# samples> array
    :param Y: y coordinates of each data point, as a <# samples> array
    :param n_squares: the number of squares that each side of the grid should have. The number of
            squares the grid will have is n_squares**2
    :return: a 2-by-<# samples> numpy array of the label of squares within the grid for
             each data point
             a 2-by-<n_squares+1> numpy array of the boundaries of each of the squares
    """
    fixed_X = X
    fixed_X[X > 34] = X[X > 34]
    fixed_X[Y > 34] = Y[Y > 34]
    fixed_Y = Y
    fixed_Y[X < 34] = X[X < 34]
    fixed_Y[Y < 34] = Y[Y < 34]
    U = fixed_X
    V = fixed_Y
    xmax, xmin = np.max(U), np.min(U)
    ymax, ymin = np.max(V), np.min(V)
    dX, dY = (U - np.min(U))/np.max(U - np.min(U)), (V - np.min(V))/np.max(V - np.min(V))
    labs = np.zeros((2, U.shape[0]))
    labs[0, :] = np.floor(dX * n_squares)
    labs[1, :] = np.floor(dY * n_squares)
    boundaries = np.zeros(shape=(2, n_squares+1))
    # print(boundaries.shape)
    a = np.linspace(xmin, xmax, n_squares + 1)
    # print(a.shape)
    boundaries[0, :] = np.linspace(xmin, xmax, n_squares + 1)
    boundaries[1, :] = np.linspace(ymin, ymax, n_squares + 1)
    return labs, boundaries

#
# import pandas as pd
# df = pd.read_pickle('./data/cleaned_with_xy.pkl')
# l, b = make_grid(df['x'], df['y'])
# print(l)