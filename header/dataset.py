import pandas as pd
import numpy as np

def load_data(x_train_filename, y_train_filename, x_test_filename):
    x_train, x_test = np.array(pd.read_csv(x_train_filename)), np.array(pd.read_csv(x_test_filename))
    y_train = np.array(pd.read_csv(y_train_filename)).reshape(-1)

    return x_train, y_train, x_test