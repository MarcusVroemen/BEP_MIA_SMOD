import pandas as pd
import glob

file_pattern = 'path_to_directory/*.csv'
file_paths = glob.glob(file_pattern)

dataframes = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)