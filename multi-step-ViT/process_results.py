#%%
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

file_pattern = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/multi-step-ViT/output/model_val_metrics/csv/*.csv'
csv_files = glob.glob(file_pattern)

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    filename = file.split('\\')[-1].split('.')[0]
    df['Source'] = filename
    
    # Extract the number using regex pattern
    run_nr = r'VIT-(\d+)_'
    df['run_nr'] = df['Source'].str.extract(run_nr)
    df['run_nr'] = pd.to_numeric(df['run_nr'])
    
    # Extract the last 4 digits from the filename
    last_digits = filename[-4:]
    df['epochs'] = last_digits
    
    columns = ['Source', 'run_nr', 'epochs'] + [col for col in df.columns if col not in ['Source', 'run_nr', 'epochs']]
    df = df[columns]
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# %%
# Assuming you have a DataFrame called 'df' and you want to select rows with numbers 122, 144, and 145
desired_numbers = [122, 144, 145, 146]
number_labels = {122: 'gryds100', 144: 'SMOD50_10-10', 145: 'SMOD50_10-15', 146: 'SMOD50_15-15'}


# Filter the DataFrame to select the desired rows
filtered_df = df[df['run_nr'].isin(desired_numbers)]# Plotting using seaborn

colors = sns.color_palette("rocket")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot 'tre' column on the first subplot
sns.boxplot(data=filtered_df, x='run_nr', y='tre', ax=axes[0], palette=colors)
axes[0].set_title('TRE')
axes[0].set_xlabel('Augmentation')
axes[0].set_ylabel('tre (mm)')
axes[0].set_xticklabels([number_labels.get(x, str(x)) for x in filtered_df['run_nr'].unique()])

# Plot 'ssim' column on the second subplot
sns.boxplot(data=filtered_df, x='run_nr', y='ssim', ax=axes[1], palette=colors)
axes[1].set_title('SSIM')
axes[1].set_xlabel('Augmentation')
axes[1].set_ylabel('ssim')
axes[1].set_xticklabels([number_labels.get(x, str(x)) for x in filtered_df['run_nr'].unique()])

# Adjust the layout and spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

# %%
# Assuming you have a DataFrame called 'df' and you want to select rows with numbers 122, 144, and 145
desired_numbers = [250, 251, 253, 254]
number_labels = {250: 's500', 251: 's1000', 253: 's1500', 254: 's2000'}


# Filter the DataFrame to select the desired rows
filtered_df = df[df['run_nr'].isin(desired_numbers)]# Plotting using seaborn

colors = sns.color_palette("rocket")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot 'tre' column on the first subplot
sns.boxplot(data=filtered_df, x='run_nr', y='tre', ax=axes[0], palette=colors)
axes[0].set_title('TRE')
axes[0].set_xlabel('Augmentation')
axes[0].set_ylabel('tre (mm)')
axes[0].set_xticklabels([number_labels.get(x, str(x)) for x in filtered_df['run_nr'].unique()])

# Plot 'ssim' column on the second subplot
sns.boxplot(data=filtered_df, x='run_nr', y='ssim', ax=axes[1], palette=colors)
axes[1].set_title('SSIM')
axes[1].set_xlabel('Augmentation')
axes[1].set_ylabel('ssim')
axes[1].set_xticklabels([number_labels.get(x, str(x)) for x in filtered_df['run_nr'].unique()])

# Adjust the layout and spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()
# %%
