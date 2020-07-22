import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_data(batch_size):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

    with h5py.File('kle50_mc10000.hdf5', 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]

    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))
    x_log_data  = np.log(x_data)
    X_train, X_test, y_train, y_test = train_test_split(x_log_data, y_data, test_size=0.2, random_state=42, shuffle=False)

    dataset_train = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_test = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader_train, data_loader_test, stats
