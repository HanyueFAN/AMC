

# 设置文件路径和数据集参数


import h5py
import numpy as np

file_path = 'F:\ISEP_Learning_Document\Semester3\End-of-track Project\dataset\data_raw\GOLD_XYZ_OSC.0001_1024.hdf5'  # 需要替换为正确的路径

# Define the new classes and SNR range
new_classes = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '32PSK', '16APSK', '32APSK', '64APSK',
    '16QAM', 'AM-SSB-WC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]
snr_range = range(2, 31)  # 2dB to 30dB

# Open the original dataset
with h5py.File(file_path, 'r') as f:
    X = f['X'][:]
    Y = f['Y'][:]
    Z = f['Z'][:]

# Filter the data based on new classes and SNR range
filtered_indices = np.isin(Y.argmax(axis=1), new_classes) & np.isin(Z.flatten(), snr_range)
X_filtered = X[filtered_indices]
Y_filtered = Y[filtered_indices]
Z_filtered = Z[filtered_indices]

# Now, create 17 smaller datasets
for i, mod_class in enumerate(new_classes):
    # Filter the dataset for the current modulation class
    class_indices = Y.argmax(axis=1) == mod_class
    X_class = X_filtered[class_indices]
    Y_class = Y_filtered[class_indices]
    Z_class = Z_filtered[class_indices]

    # Save the filtered data to a new HDF5 file
    filename = f'F:\sourceCode\AMC-2\Dataset\part_{mod_class}.h5'
    with h5py.File(filename, 'w') as fw:
        fw.create_dataset('X', data=X_class)
        fw.create_dataset('Y', data=Y_class)
        fw.create_dataset('Z', data=Z_class)
        print(f'File {filename} created with shapes: X:{X_class.shape}, Y:{Y_class.shape}, Z:{Z_class.shape}')
