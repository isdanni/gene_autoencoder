import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from keras import Input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.models import Model, load_model


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



# load the processed data
df = pd.read_csv('/home/danni/Dropbox/CS/Code/gene_autoencoder/zeros_removed.csv',header=None)
# split the data into training and testing data set
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['zero_percentage'], axis=1)
y_test = X_test['zero_percentage']
X_test = X_test.drop(['zero_percentage'], axis=1)
X_train = X_train.values
X_test = X_test.values

def sampling(args):
  # Function with args required for Keras Lambda function
  z_mean, z_log_var = args
  
  # Draw epsilon of the same shape from a standard normal distribution
  epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,stddev=epsilon_std)
  
  # The latent vector is non-deterministic and differentiable
  # in respect to z_mean and z_log_var
  z = z_mean + K.exp(z_log_var / 2) * epsilon
  return z
  

# add Loss Function
class VariationalLayer(Layer):
  """
  using variational_autoencoder.py from Keras
  """
  def __init__(self, **kwargs):
    self.is_placeholder = True
    super(CustomVariationalLayer, self).__init__(**kwargs)
    
  def loss(self, x_input,x_encoded):
    reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
    kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - K.exp(z_log_var_encoded), axis=-1)
    return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))
  
  def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

      

# 2. Encoder & Decoder
# set hyper parameters
original_dim = 
latent_dim = 

batc_size = 32 # as good default batch size to train neural nets
epochs = 100
learning_rate=0.0005 # how to adjust?

# Encoder
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
