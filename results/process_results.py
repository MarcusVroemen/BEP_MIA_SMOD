#%%
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

file_pattern = 'C:/Users/20203531/OneDrive - TU Eindhoven/Y3/Q4/BEP/BEP_MIA_DIR/multi-step-ViT/output/model_val_metrics/csv/*.csv'
csv_files = glob.glob(file_pattern)

dfs_test = []
for file in csv_files:
    df_test = pd.read_csv(file)
    filename = file.split('\\')[-1].split('.')[0]
    df_test['Source'] = filename
    
    # Extract the number using regex pattern
    run_nr = r'VIT-(\d+)_'
    df_test['run_nr'] = df_test['Source'].str.extract(run_nr)
    df_test['run_nr'] = pd.to_numeric(df_test['run_nr'])
    
    # Extract the last 4 digits from the filename
    last_digits = filename[-4:]
    df_test['epochs'] = last_digits
    
    columns = ['Source', 'run_nr', 'epochs'] + [col for col in df_test.columns if col not in ['Source', 'run_nr', 'epochs']]
    df_test = df_test[columns]
    dfs_test.append(df_test)

df_test = pd.concat(dfs_test, ignore_index=True)

df_neptune = pd.read_csv("neptune_dashboard.csv")
df_neptune["run_nr"]= df_neptune["Id"].str.slice(-3).astype(int)

df_main = pd.merge(df_test, df_neptune, on='run_nr')
df_main = df_main[df_main['run_nr'].isin(df_neptune['run_nr'])]


#%%
title = "50 epochs of SMOD σ₁ and σ₂"

df_filtered = df_main[df_main["AUG"]=="SMOD"]
df_filtered = df_filtered.sort_values(['parameters/sigma1', 'parameters/sigma2'])
# df_filtered = df_filtered.

colors = sns.color_palette("crest")
colors = colors[:4] * (len(colors) // 4)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(data=df_filtered, x='run_nr', y='tre', ax=axes[0], palette=colors)
axes[0].set_title(f'TRE {title}', fontsize=15)
axes[0].set_xlabel('σ₁', fontsize=15)
axes[0].set_ylabel('TRE (mm)')
# axes[0].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6])
axes[0].set_xticks(range(len(df_filtered['parameters/sigma1'].values[5::6])))  # Set the x-tick positions
axes[0].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6], rotation=90)  # Hide the default x-tick labels

axes2 = axes[0].twiny()
axes2.set_xticks(axes[0].get_xticks())
axes2.set_xticklabels(df_filtered['parameters/sigma2'].values[5::6], rotation=90)  # Set the second set of x-tick labels
axes2.set_xlabel('σ₂', fontsize=15)

sns.boxplot(data=df_filtered, x='run_nr', y='ssim', ax=axes[1], palette=colors)
axes[1].set_title(f'SSIM {title}', fontsize=15)
axes[1].set_xlabel('σ₁', fontsize=15)
axes[1].set_ylabel('ssim')
# axes[1].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6])
axes[1].set_xticks(range(len(df_filtered['parameters/sigma1'].values[5::6])))  # Set the x-tick positions
axes[1].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6], rotation=90)  # Hide the default x-tick labels

# Create a second x-axis with desired x-tick labels
# axes[1].twiny().set_xticks(range(len(df_filtered['parameters/sigma2'].values[5::6])))
# axes[1].twiny().set_xticklabels(df_filtered['parameters/sigma2'].values[5::6], rotation=90)  # Rotate the labels if needed

axes2 = axes[1].twiny()
axes2.set_xticks(axes[1].get_xticks())
axes2.set_xticklabels(df_filtered['parameters/sigma2'].values[5::6], rotation=90)  # Set the second set of x-tick labels
axes2.set_xlabel('σ₂', fontsize=15)

plt.tight_layout()
plt.show()


#%%
title = "50 epochs of SMOD breath σ₂"
df_filtered = df_main[df_main["AUG"]=="SMOD_breath"]
df_filtered = df_filtered.sort_values(['parameters/sigma1', 'parameters/sigma2'])
# df_filtered = df_filtered.

colors = sns.color_palette("crest")
colors = colors[:4] * (len(colors) // 4)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

sns.boxplot(data=df_filtered, x='run_nr', y='tre', ax=axes[0], palette=colors)
axes[0].set_title(f'TRE {title}', fontsize=15)
axes[0].set_xlabel('σ₂', fontsize=15)
axes[0].set_ylabel('TRE (mm)')
# axes[0].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6])
axes[0].set_xticks(range(len(df_filtered['parameters/sigma2'].values[5::6])))  # Set the x-tick positions
axes[0].set_xticklabels(df_filtered['parameters/sigma2'].values[5::6], rotation=90)  # Hide the default x-tick labels

sns.boxplot(data=df_filtered, x='run_nr', y='ssim', ax=axes[1], palette=colors)
axes[1].set_title(f'SSIM {title}', fontsize=15)
axes[1].set_xlabel('σ₂', fontsize=15)
axes[1].set_ylabel('ssim')
# axes[1].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6])
axes[1].set_xticks(range(len(df_filtered['parameters/sigma2'].values[5::6])))  # Set the x-tick positions
axes[1].set_xticklabels(df_filtered['parameters/sigma2'].values[5::6], rotation=90)  # Hide the default x-tick labels


plt.tight_layout()
plt.show()


# %%
title = "150 epochs of real and gryds"
df_filtered = df_main[df_main["AUG"].isin(["none", "gryds", "gryds+real"])]
df_filtered = df_filtered.sort_values(['constant'])
# df_filtered = df_filtered.

colors = sns.color_palette("crest")
colors = colors[:4] * (len(colors) // 4)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))

sns.boxplot(data=df_filtered, x='run_nr', y='tre', ax=axes[0], palette=colors)
axes[0].set_title(f'TRE {title}', fontsize=15)
axes[0].set_xlabel('σ₂', fontsize=15)
axes[0].set_ylabel('TRE (mm)')
# axes[0].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6])
axes[0].set_xticks(range(len(df_filtered['AUG'].values[5::6])))  # Set the x-tick positions
axes[0].set_xticklabels(df_filtered['AUG'].values[5::6], rotation=90)  # Hide the default x-tick labels

sns.boxplot(data=df_filtered, x='run_nr', y='ssim', ax=axes[1], palette=colors)
axes[1].set_title(f'SSIM {title}', fontsize=15)
axes[1].set_xlabel('σ₂', fontsize=15)
axes[1].set_ylabel('ssim')
# axes[1].set_xticklabels(df_filtered['parameters/sigma1'].values[5::6])
axes[1].set_xticks(range(len(df_filtered['AUG'].values[5::6])))  # Set the x-tick positions
axes[1].set_xticklabels(df_filtered['AUG'].values[5::6], rotation=90)  # Hide the default x-tick labels


plt.tight_layout()
plt.show()

# %%
