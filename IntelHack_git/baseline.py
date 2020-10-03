import numpy as np
import pandas as pd
from preprocess import pickle_path, create_regression_Xy
from just_for_testing import sample_random_rows
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def run_baseline(df: pd.DataFrame, sklearn_regressor=SVR, arguments: dict={'gamma':'auto'},
                 classifier=False) -> dict:
    X_dict, y_dict = create_regression_Xy(df, look_back=50)
    regressor_dict = {a: None for a in X_dict.keys()}
    for i, event in enumerate(X_dict.keys()):
        y = y_dict[event]
        X = X_dict[event]
        if classifier:
            y[y > 0] = 1
        X_train = X[:int(np.ceil(0.8*X.shape[0]))]
        y_train = y[:int(np.ceil(0.8*X.shape[0]))]
        X_test = X[int(np.ceil(0.8*X.shape[0])):]
        y_test = y[int(np.ceil(0.8*X.shape[0])):]
        if arguments is not None:
            model = sklearn_regressor(**arguments)
        else: model = sklearn_regressor()
        model.fit(X_train, y_train)
        regressor_dict[event] = (model, X_train, X_test, y_train, y_test)

    return regressor_dict


def prediction_RMSE(pred, y) -> float:
    return np.sqrt(np.mean((pred - y) ** 2))


if __name__ == '__main__':

    df = pd.read_pickle(pickle_path)
    X, y = create_regression_Xy(df, look_back=50)
    train_acc = np.zeros((2, len(X.keys())))
    test_acc = np.zeros((2, len(X.keys())))

    mean_train_acc = np.zeros(len(X.keys()))
    mean_test_acc = np.zeros(len(X.keys()))

    train_delta = np.zeros((2, len(X.keys())))
    test_delta = np.zeros((2, len(X.keys())))

    for i, event in enumerate(X.keys()):
        X_train, X_test, y_train, y_test = sample_random_rows(X[event], y[event])
        svm = SVR()
        svm.fit(X_train, y_train)
        lin = LinearRegression()
        # lin = DecisionTreeRegressor()
        lin.fit(X_train, y_train)
        train_acc[0, i] = np.sqrt(np.mean((svm.predict(X_train) - y_train) ** 2))
        test_acc[0, i] = np.sqrt(np.mean((svm.predict(X_test) - y_test) ** 2))

        train_acc[1, i] = np.sqrt(np.mean((lin.predict(X_train) - y_train) ** 2))
        test_acc[1, i] = np.sqrt(np.mean((lin.predict(X_test) - y_test) ** 2))

        mean_train_acc[i] = np.sqrt(np.mean((X_train.mean(axis=1) - y_train) ** 2))
        mean_test_acc[i] = np.sqrt(np.mean((X_test.mean(axis=1) - y_test) ** 2))

        train_delta[:, i] = np.abs(train_acc[:, i] - mean_train_acc[None, i])
        test_delta[:, i] = np.abs(test_acc[:, i] - mean_test_acc[None, i])

    plt.figure(figsize=(10, 16))
    plt.subplot(3, 1, 1)
    plt.bar(np.arange(train_acc.shape[1]), train_acc[0, :])
    plt.ylabel('SVM train RMSE')
    plt.subplot(3, 1, 2)
    plt.bar(np.arange(train_acc.shape[1]), train_acc[1, :])
    plt.ylabel('LR train RMSE')
    plt.subplot(3, 1, 3)
    plt.bar(np.arange(mean_train_acc.shape[0]), mean_train_acc)
    plt.ylabel('Mean (baseline) train RMSE')
    plt.xlabel('different events')

    plt.figure(figsize=(10, 16))
    plt.subplot(3, 1, 1)
    plt.bar(np.arange(test_acc.shape[1]), test_acc[0, :])
    plt.ylabel('SVM test RMSE')
    plt.subplot(3, 1, 2)
    plt.bar(np.arange(test_acc.shape[1]), test_acc[1, :])
    plt.ylabel('LR test RMSE')
    plt.subplot(3, 1, 3)
    plt.bar(np.arange(mean_test_acc.shape[0]), mean_test_acc)
    plt.ylabel('Mean (baseline) test RMSE')
    plt.xlabel('different events')

    plt.figure(figsize=(10, 16))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(test_acc.shape[1]), (test_acc[0, :]/mean_test_acc) - 1)
    plt.ylabel('SVM / Baseline test RMSE')
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(test_acc.shape[1]), (test_acc[1, :]/mean_test_acc) - 1)
    plt.ylabel('LR / Baseline test RMSE')
    plt.xlabel('different events')

    plt.figure(figsize=(10, 16))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(train_acc.shape[1]), (train_acc[0, :]/mean_train_acc) - 1)
    plt.ylabel('SVM / Baseline train RMSE')
    plt.subplot(2, 1, 2)
    plt.bar(np.arange(train_acc.shape[1]), (train_acc[1, :]/mean_train_acc) - 1)
    plt.ylabel('LR / Baseline train RMSE')
    plt.xlabel('different events')

    plt.show()