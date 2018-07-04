import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class Policy(object):
    def __init__(self, sess, obs_dim, act_dim, clip_range=0.2,
                 epochs=10, lr=3e-5, hdim=64, mdn_weight="sparsemax", n_mixture=4, max_std=1.0,
                 seed=0,alpha=1.0):
        self.sess = sess
        self.alpha = alpha
        self.seed = seed
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.clip_range = clip_range
        
        self.epochs = epochs
        self.lr = lr
        self.hdim = hdim
        self.mdn_weight = mdn_weight
        self.n_mixture = n_mixture
        self.max_std = max_std
        
        self._build_graph()

    def _build_graph(self):
        self._placeholders()
        self._policy_nn()
        self._logprob()
        self._kl_entropy()
        self._loss_train_op()
        self.init = tf.global_variables_initializer()
        self.variables = tf.global_variables()
            
    def _placeholders(self):
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')

        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        
        # place holder for old parameters
        self.old_std_ph = tf.placeholder(tf.float32, (None, self.act_dim, self.n_mixture), 'old_std')
        self.old_mean_ph = tf.placeholder(tf.float32, (None, self.act_dim, self.n_mixture), 'old_means')
        self.old_pi_ph = tf.placeholder(tf.float32, (None, self.n_mixture), 'old_pi')

    def _policy_nn(self):
        
        hid1_size = self.hdim
        hid2_size = self.hdim
        
        # TWO HIDDEN LAYERS
        with tf.variable_scope('Actor'):  
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h2")
            means = tf.layers.dense(out, self.act_dim*self.n_mixture,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), 
                                    name="flat_mean")
            self.mean = tf.reshape(means,shape=[-1,self.act_dim,self.n_mixture], name="mean")
            logits_std = tf.layers.dense(out, self.act_dim*self.n_mixture,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), 
                                    name="flat_logits_std")
            self.std = tf.reshape(self.max_std*tf.sigmoid(logits_std),shape=[-1,self.act_dim,self.n_mixture], name="std")
            if self.mdn_weight=="softmax":
                self.pi = tf.nn.softmax(tf.layers.dense(out, self.n_mixture,
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="pi"))
            elif self.mdn_weight=="sparsemax":
                self.pi = tf.contrib.sparsemax.sparsemax(tf.layers.dense(out, self.n_mixture,
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="pi"))

    def _logprob(self):
        # PROBABILITY WITH TRAINING PARAMETER
        y = self.act_ph 
        mu = self.mean
        sigma = self.std
        pi = self.pi
        
        quadratics = -0.5*tf.reduce_sum(tf.square((tf.tile(y[:,:,tf.newaxis],[1,1,self.n_mixture])-mu)/sigma),axis=1)
        logdet = -0.5*tf.reduce_sum(tf.log(sigma),axis=1)
        logconstant = - 0.5*self.act_dim*np.log(2.*np.pi)
        logpi = tf.log(pi + 1e-8)
        
        exponents = quadratics + logdet + logconstant + logpi
        logprobs = tf.reduce_logsumexp(exponents,axis=1)
        
        self.logp = logprobs

        old_mu_ph = self.old_mean_ph                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        old_sigma_ph = self.old_std_ph
        old_pi_ph = self.old_pi_ph
    
        quadratics = -0.5*tf.reduce_sum(tf.square((tf.tile(y[:,:,tf.newaxis],[1,1,self.n_mixture])-old_mu_ph)/old_sigma_ph),axis=1)
        logdet = -0.5*tf.reduce_sum(tf.log(old_sigma_ph),axis=1)
        logconstant = - 0.5*self.act_dim*np.log(2.*np.pi)
        logpi = tf.log(old_pi_ph + 1e-8)
        
        exponents = quadratics + logdet + logconstant + logpi
        old_logprobs = tf.reduce_logsumexp(exponents,axis=1)
        
        self.logp_old = old_logprobs
        
    def _kl_entropy(self):
        
        def energy(mu1,std1,pi1,mu2,std2,pi2):
            energy_components = []
            for i in range(self.n_mixture):
                for j in range(self.n_mixture):
                    mu1i = mu1[:,:,i] 
                    mu2j = mu2[:,:,j]
                    std1i = std1[:,:,i]
                    std2j = std2[:,:,j]
                    pi1i = pi1[:,i]
                    pi2j = pi2[:,j]
                    energy_components.append(pi1i*pi2j * tf.exp(-0.5*tf.reduce_sum(((mu1i - mu2j)/(std1i+std2j))**2+2.*tf.log(std1i+std2j)+np.log(2*np.pi),axis=1)))
            return tf.reduce_sum(tf.stack(energy_components,axis=1),axis=1) 
            
        mean, std, weight = self.mean, self.std, self.pi
        old_mean, old_std, old_weight = self.old_mean_ph, self.old_std_ph, self.old_pi_ph

#         weight = weight/tf.reduce_sum(weight,axis=1,keep_dims=True)
#         old_weight = old_weight/tf.reduce_sum(old_weight,axis=1,keep_dims=True)
        
        if self.mdn_weight=="softmax":
            self.entropy = tf.reduce_sum(self.pi*(-tf.log(self.pi) + 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                                                                        tf.reduce_sum(tf.log(std),axis=1))),axis=1)
            self.entropy = tf.reduce_mean(self.entropy)
        elif self.mdn_weight=="sparsemax":
            self.entropy = tf.reduce_mean(0.5*(1-energy(mean, std, weight,mean, std, weight)))
            
        log_det_cov_old = tf.reduce_sum(tf.log(old_std),axis=1)
        log_det_cov_new = tf.reduce_sum(tf.log(std),axis=1)
        tr_old_new = tf.reduce_sum(old_std/std,axis=1)

        kl = tf.reduce_sum(old_weight*tf.log((old_weight+1e-8)/(weight+1e-8)) + 0.5 * old_weight*(log_det_cov_new - log_det_cov_old + tr_old_new + tf.reduce_sum(tf.square((mean - old_mean)/std),axis=1) - self.act_dim),axis=1)
        self.kl = tf.reduce_mean(kl)
        
    def _loss_train_op(self):
        
        # Proximal Policy Optimization CLIPPED LOSS FUNCTION
#         ratio = tf.exp(self.logp - self.logp_old) 
#         clipped_ratio = tf.clip_by_value(ratio,clip_value_min=1-self.clip_range,clip_value_max=1+self.clip_range) 
#         self.loss = -tf.reduce_mean(tf.minimum(self.advantages_ph*ratio,self.advantages_ph*clipped_ratio))
                
        def energy(mu1,std1,pi1,mu2,std2,pi2):
            energy_components = []
            for i in range(self.n_mixture):
                for j in range(self.n_mixture):
                    mu1i = mu1[:,:,i] 
                    mu2j = mu2[:,:,j]
                    std1i = std1[:,:,i]
                    std2j = std2[:,:,j]
                    pi1i = pi1[:,i]
                    pi2j = pi2[:,j]
                    energy_components.append(pi1i*pi2j * tf.exp(-0.5*tf.reduce_sum(((mu1i - mu2j)/(std1i+std2j))**2+2.*tf.log(std1i+std2j)+np.log(2*np.pi),axis=1)))
            return tf.reduce_sum(tf.stack(energy_components,axis=1),axis=1) 
            
        mean, std, weight = self.mean, self.std, self.pi
        
        alpha = self.alpha
        self.error = tf.maximum(self.advantages_ph + alpha*(0.5 + 0.5*energy(mean, std, weight, mean, std, weight)), 
                                0)- alpha*tf.exp(self.logp)
        self.loss = tf.reduce_mean(tf.square(self.error))
        # OPTIMIZER 
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def sample(self, obs): # SAMPLE FROM POLICY
        feed_dict = {self.obs_ph: obs}
        pi, mu, sigma = self.sess.run([self.pi, self.mean, self.std],feed_dict=feed_dict)
        pi = (pi+1e-8)/np.sum(pi+1e-8,axis=1,keepdims=True)
        sigma = sigma
        n_points = np.shape(obs)[0]
        
        _y_sampled = np.zeros([n_points,self.act_dim])
        for i in range(n_points):
            k = np.random.choice(self.n_mixture,p=pi[i,:])
            _y_sampled[i,:] = mu[i,:,k] + np.random.randn(1,self.act_dim)*sigma[i,:,k]
        return _y_sampled
        
    def control(self, obs): # COMPUTE MEAN
        feed_dict = {self.obs_ph: obs}
        pi, mu, sigma = self.sess.run([self.pi, self.mean, self.std],feed_dict=feed_dict)
        pi = (pi+1e-8)/np.sum(pi+1e-8,axis=1,keepdims=True)
        sigma = sigma
        n_points = np.shape(obs)[0]
        
        _y_sampled = np.zeros([n_points,self.act_dim])
        for i in range(n_points):
            k = np.argmax(pi[i,:])
            _y_sampled[i,:] = mu[i,:,k] + np.random.randn(1,self.act_dim)*sigma[i,:,k]
        return _y_sampled        
    
    def update(self, observes, actions, advantages, batch_size = 128): # TRAIN POLICY
        
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches
        
        old_means_np, old_std_np, old_pi_np = self.sess.run([self.mean, self.std, self.pi],{self.obs_ph: observes}) # COMPUTE OLD PARAMTER
        for e in range(self.epochs):
            observes, actions, advantages, old_means_np, old_std_np = shuffle(observes, actions, advantages, old_means_np, old_std_np, random_state=self.seed)
            for j in range(num_batches): 
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: observes[start:end,:],
                     self.act_ph: actions[start:end,:],
                     self.advantages_ph: advantages[start:end],
                     self.old_std_ph: old_std_np[start:end,:,:],
                     self.old_mean_ph: old_means_np[start:end,:,:],
                     self.old_pi_ph: old_pi_np[start:end,:],
                     self.lr_ph: self.lr}        
                self.sess.run(self.train_op, feed_dict)
            
        feed_dict = {self.obs_ph: observes,
                 self.act_ph: actions,
                 self.advantages_ph: advantages,
                 self.old_std_ph: old_std_np,
                 self.old_mean_ph: old_means_np,
                 self.old_pi_ph: old_pi_np,
                 self.lr_ph: self.lr}             
        loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
        return loss, kl, entropy