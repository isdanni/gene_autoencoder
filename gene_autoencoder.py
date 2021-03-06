import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from keras import Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler


# 1. pre-process data
# each row: count num of zeros & percentage
# then drop the rows that percentage >= 0.75
# each column: sum values & drop the columns values < = 100
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
df = df[df.loc['Total'] > 100]
# export files
df.to_csv('/home/danni/Dropbox/CS/Code/gene_autoencoder/zeros_removed.csv',index=False)
print('Pre-process finished!')

# now load the processed data
df = pd.read_csv('/home/danni/Dropbox/CS/Code/gene_autoencoder/zeros_removed.csv',header=None)
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['zero_percentage'], axis=1)
y_test = X_test['zero_percentage']
X_test = X_test.drop(['zero_percentage'], axis=1)
X_train = X_train.values
X_test = X_test.values

input_dim = X_train.shape[1]
encoding_dim = 20 # layer of how many connected +
# first one for encoder



# 2. Design encoder & decoder
input_layer = Input(shape=(input_dim))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='op', loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

nb_epoch = 100
batch_size= 16

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
autoencoder = load_model('model.h5')


"""
Class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])
“”“


# 3. Evaluation
# weight = tf.multiply(4, tf.cast(tf.equal(labels, 3), tf.float32)) + 1
# onehot_labels = tf.one_hot(labels, num_classes=5)
# tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels, weight=weight)

weight = MyComplicatedWeightingFunction(labels)
  weight = tf.div(weight, tf.size(weight))
  loss = tf.contrib.losses.mean_squared_error(predictions, depths, weight)
