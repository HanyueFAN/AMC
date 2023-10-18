import torch
import h5py
from torch.utils.data import DataLoader, TensorDataset, random_split

file_path = '/Users/fhy/isep3th/project/dataprocess/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5'
# file_path = 'F:\ISEP_Learning_Document\Semester3\End-of-track Project\dataset\data_raw\GOLD_XYZ_OSC.0001_1024.hdf5'


def data_analyze():
    X_tensor, Y_tensor, Z_tensor = data_pre()
    # Inspect attributes and basic information
    print("Tensor Dataset_Z Information:")
    print("--------------------------")

    # Check the data type of the tensor dataset
    print(f"Data type: {Z_tensor.type()}")

    # Get the shape (dimensions) of the data
    print(f"Data shape: {Z_tensor.shape}")

    # Check the number of elements in the dataset
    print(f"Number of elements: {Z_tensor.numel()}")

    # Access the first few elements of the data
    print("First few data elements:")
    print(Z_tensor[:2])


def data_pre():
    # absolute path of modulation dataset

    # Open my file in read mode
    file_raw = h5py.File(file_path, 'r')

    # access the datasets within the HDF5 File
    data_raw_x = file_raw['X'][:]
    data_raw_y = file_raw['Y'][:]
    data_raw_z = file_raw['Z'][:]

    # Convert the hdf5 dataset to a PyTorch tensor
    X_tensor = torch.from_numpy(data_raw_x).float()
    Y_tensor = torch.from_numpy(data_raw_y).long()
    Z_tensor = torch.from_numpy(data_raw_z).long()

    # Return the tensors
    return X_tensor, Y_tensor, Z_tensor


def data_split_mixed():
    X_tensor, Y_tensor, Z_tensor = data_pre()
    dataset = TensorDataset(X_tensor, Y_tensor, Z_tensor)

    # Define the sizes for training, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each set
    batch_size = 64  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def data_split_separately():
    X_tensor, Y_tensor, Z_tensor = data_pre()
    # Define the proportions for splitting
    total_samples = len(X_tensor)
    train_percent = 0.6
    val_percent = 0.2
    test_percent = 0.2

    # Calculate the number of samples for each set
    num_train_samples = int(train_percent * total_samples)
    num_val_samples = int(val_percent * total_samples)
    num_test_samples = int(test_percent * total_samples)

    # Split the data
    X_train = X_tensor[:num_train_samples]
    Y_train = Y_tensor[:num_train_samples]
    Z_train = Z_tensor[:num_train_samples]

    X_val = X_tensor[num_train_samples:num_train_samples + num_val_samples]
    Y_val = Y_tensor[num_train_samples:num_train_samples + num_val_samples]
    Z_val = Z_tensor[num_train_samples:num_train_samples + num_val_samples]

    X_test = X_tensor[-num_test_samples:]
    Y_test = Y_tensor[-num_test_samples:]
    Z_test = Z_tensor[-num_test_samples:]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_train, Z_val, Z_test


if __name__ == '_main_':
    data_analyze();
    X, Y, Z = data_pre()
    # the shapes of the tensors:
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Z shape:", Z.shape)
    # analyze the data
    # data_analyze()
    train_loader, val_loader, test_loader = data_split_mixed();
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_train, Z_val, Z_test = data_split_separately()

