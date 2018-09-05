import pandas as pd


# read data
df = pd.read_csv('/home/danni/Dropbox/CS/Code/gene_autoencoder/SC3expression.csv'
,header=None)
df['zero_each_row'] = df.apply( lambda s : s.value_counts().get(0,0), axis=1) # get number of zeros in each row
total_cols = len(df.columns)
df['zero_percentage'] = df['zero_each_row']/total_cols

# remove the rows meeting the requirements
df = df[df['zero_percentage'] <= 0.75]

# cols
# df.groupby('id').apply(lambda column: column.sum()/(column != 0).sum())
df.loc["Total"] = df.sum()

# export files
df.to_csv('/home/danni/Dropbox/CS/Code/gene_autoencoder/zeros_removed.csv',index=False)
print('Done!')