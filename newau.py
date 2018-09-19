import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model



df= pd.read_csv('/home/danni/Dropbox/CS/Code/gene_autoencoder/human_time.csv'
                 , header=None)
df= df.iloc[1:]  # drop first header


df['zero_each_row'] = df.apply( lambda s : s.value_counts().get(0,0), axis=1) # get number of zeros in each row
total_cols = len(df.colamns) -1
df['zero_percentage'] = df['zero_each_row']/total_cols
df = df[df['zero_percentage'] <= 0.75]

df.loc["Total"] = df.sum()

print(df)
