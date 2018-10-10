import pandas as pd
import os

#Store the names of all the files into a list
all_files = list(os.listdir())

#Delete non-data files, including self-referential name
all_files.remove('merg_dataframes.py')
all_files.remove('.DS_Store')

print(all_files)

df = pd.DataFrame()

#Import all csv files into pandas and concatenate into one suuper dataframe
for filename in all_files:
	current_df = pd.read_csv(filename)
	print(current_df.shape)
	df = pd.concat([df, current_df])
print(df.shape)

#Low selling phones will have duplicates, delete those
df = df.drop_duplicates()
print(df.shape)

file_name = "iphone_df_all.csv"
df.to_csv(file_name, encoding='utf-8', index=False)
