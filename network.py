import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class Value(object):
    def __init__(self, sess, obs_dim, epochs=20, lr=1e-4, hdim=64, seed=0):
        
        self.sess = sess
        self.seed = seed
        self.obs_dim = obs_dim
        self.epochs = epochs
        self.lr = lr
        self.hdim = hdim
        self._build_graph()
        
    def _build_graph(self):
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
        self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            
        hid1_size = self.hdim 
        hid2_size = self.hdim 
            
        with tf.variable_scope('Critic'):    
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.01,seed=self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.01,seed=self.seed), name="h2")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                          stddev=0.01,seed=self.seed), name='output')
            self.out = tf.squeeze(out)
            
        # L2 LOSS
        self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))
            
        # OPTIMIZER
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
            
        self.init = tf.global_variables_initializer()
        self.variables = tf.global_variables()
        
    def fit(self, x, y, batch_size=32):
        num_batches = max(x.shape[0] // batch_size, 1)
        x_train, y_train = x, y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train, random_state=self.seed)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                self.sess.run([self.train_op], feed_dict=feed_dict)
        feed_dict = {self.obs_ph: x_train,
                     self.val_ph: y_train}
        loss, = self.sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def predict(self, x): # PREDICT VALUE OF THE GIVEN STATE
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(y_hat)