import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Set the paths
raw_data_dir = '/Users/rewatiramansingh/Desktop/Projects/listening-to-pipes/data/raw/Dynamic Pressure Sensor'
processed_data_dir = '/Users/rewatiramansingh/Desktop/Projects/listening-to-pipes/data/processed'

# Create the processed data directory if it doesn't exist
os.makedirs(processed_data_dir, exist_ok=True)

# Preprocess the pressure sensor data using Kriging interpolation
def preprocess_pressure_data(file_path):
    # Load the CSV data
    df = pd.read_csv(file_path, header=None, names=['Sample', 'Value'])

    # Perform Kriging interpolation
    X = df['Sample'].values.reshape(-1, 1)
    y = df['Value'].values
    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X, y)
    X_new = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    y_pred, y_std = gpr.predict(X_new, return_std=True)

    # Create the processed data DataFrame
    processed_df = pd.DataFrame({'Sample': X_new.flatten(), 'Value': y_pred})
    return processed_df

# Create the spatial dataframe
def create_spatial_dataframe(x_coords, y_coords):
    # Create the spatial dataframe
    spatial_df = np.column_stack((x_coords, y_coords))
    return spatial_df

# Preprocess the data and create the spatial dataframes for each class
for network_type in ['Branched', 'Looped']:
    network_dir = os.path.join(raw_data_dir, network_type)
    for leak_type in os.listdir(network_dir):
        leak_dir = os.path.join(network_dir, leak_type)

        # Create the class-specific directories
        class_processed_dir = os.path.join(processed_data_dir, f"{network_type}_{leak_type}")
        os.makedirs(class_processed_dir, exist_ok=True)

        # Preprocess the pressure sensor data
        spatial_coords = []
        for file_name in os.listdir(leak_dir):
            # Load the raw data and preprocess it
            file_path = os.path.join(leak_dir, file_name)
            processed_df = preprocess_pressure_data(file_path)

            # Create the processed data file path
            processed_file_name = f"{file_name.split('.')[0]}.csv"
            processed_file_path = os.path.join(class_processed_dir, processed_file_name)

            # Save the processed data
            processed_df.to_csv(processed_file_path, index=False)
            print(f"Processed and saved: {processed_file_path}")

            # Collect the spatial coordinates
            if file_name == os.listdir(leak_dir)[0]:
                if leak_type == 'Circumferential Crack':
                    x_coords = np.array([2.45, 4.9, 4.9, 2.45, 0, 4.9, 2.45]) / 4.9
                    y_coords = np.array([4.9, 2.45, 4.9, 2.45, 0, 0, 7.35]) / 7.35
                # Add more conditions for other leak types
                spatial_coords = create_spatial_dataframe(x_coords, y_coords)

        # Save the class-specific spatial dataframe
        spatial_df_path = os.path.join(class_processed_dir, 'spatial_dataframe.csv')
        np.savetxt(spatial_df_path, spatial_coords, delimiter=',')
        print(f"Saved spatial dataframe to: {spatial_df_path}")