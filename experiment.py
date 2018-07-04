from collections import deque
from multigoal import MultiGoalEnv
import policy
import network
import numpy as np
import scipy.signal
import pickle
import tensorflow as tf
from sklearn.utils import shuffle

def discount(x, gamma=0.99): # compute discount
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def add_value(trajectories, val_func): # Add value estimation for each trajectories
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values

def add_gae(trajectories, gamma=0.99, lam=0.98): # generalized advantage estimation (for training stability)
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        values = trajectory['values']
        
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        
        trajectory['advantages'] = advantages
        trajectory['returns'] = values+advantages

def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])

    # Normalization of advantages 
    # In baselines, which is a github repo including implementation of PPO by OpenAI, 
    # all policy gradient methods use advantage normalization trick as belows.
    # The insight under this trick is that it tries to move policy parameter towards locally maximum point.
    # Sometimes, this trick doesnot work.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, returns
    
class Experiments:
    def __init__(self, seed=0, env_name = 'MultiGoal', mdn_hidden_spec=None, 
                v_epochs=50, v_hdim=32, v_lr=1e-3,
                p_epochs=30, p_hdim=32, p_lr=3e-4, clip_range=0.2, alpha=40.0,
                batch_size = 128, episode_size = 100, nupdates = 150,
                gamma=0.99, max_step = 1000):
        # Fix the numpy random seed
        seed = 0
        np.random.seed(seed)

        # Set session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        env = MultiGoalEnv()

        # Get environment information
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Set network spec
        if mdn_hidden_spec is None:
            mdn_hidden_spec = [
                {'dim': 32,'activation': tf.nn.tanh},
                {'dim': 32,'activation': tf.nn.tanh}
            ]

        # Initialize Tensorflow Graph
        #tf.reset_default_graph()
        
        # Gen value network
        value_func = network.Value(sess, obs_dim, epochs=v_epochs, hdim=v_hdim, lr=v_lr, seed=seed)

        # Initialize tf variable and old variable of value network
        tf.set_random_seed(seed)
        
        # Gen policy function
        policy_func = policy.Policy(sess, obs_dim, act_dim, epochs=p_epochs, hdim=p_hdim, lr=p_lr, clip_range=clip_range, seed=seed, alpha=alpha)
        
        sess.run(tf.global_variables_initializer())
        
        # Store All Variable to Class
        self.seed=seed
        self.env_name=env_name
        self.mdn_hidden_spec=mdn_hidden_spec
        
        self.v_epochs = v_epochs
        self.v_hdim = v_hdim
        self.v_lr = v_lr
        
        self.p_epochs = p_epochs
        self.p_hdim = p_hdim
        self.p_lr = p_lr
        self.clip_range = clip_range
        self.alpha = alpha
                
        self.batch_size = batch_size
        self.episode_size = episode_size
        self.nupdates = nupdates
        self.gamma = gamma
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env = env
        self.value_func = value_func
        self.policy_func = policy_func
        self.sess = sess
        self.config = config
        
    def run(self, display_period=10):
        file_name = "results/policy_" + str(self.p_epochs) + "_" + str(self.p_hdim) + "_" + str(self.p_lr) + "_" + str(self.alpha)
        file_name = file_name + "_value_" + str(self.v_epochs) + "_" + str(self.v_hdim) + "_" + str(self.v_lr)
        file_name = file_name + "_train_" + str(self.episode_size) + "_" + str(self.batch_size) + "_" + str(self.nupdates)
        file_name = file_name + "_gamma_" + str(self.gamma)
        f = open(file_name + "_log.txt", 'w')
        for update in range(self.nupdates+1):
            trajectories = self.run_policy(self.env, self.policy_func, episodes=self.episode_size)

            add_value(trajectories, self.value_func)
            add_gae(trajectories, gamma=self.gamma)
            observes, actions, advantages, returns = build_train_set(trajectories)

            pol_loss, pol_kl, pol_entropy = self.policy_func.update(observes, actions, advantages, batch_size=self.batch_size)  
            vf_loss = self.value_func.fit(observes, returns, batch_size=self.batch_size)

            mean_ret = np.mean([np.sum(t['rewards']) for t in trajectories])
            num_goal = self.env.print_goal_reached()
            
            f.write('[{}/{}] Mean Ret : {:.3f}, Value Loss : {:.3f}, Policy loss : {:.5f}, Policy KL : {:.5f}, Policy Entropy : {:.3f}, Goal Reached : {} ***'.format(update, self.nupdates, mean_ret, vf_loss, pol_loss, pol_kl, pol_entropy, num_goal))
            if (update%5) == 0:
                print('[{}/{}] Mean Ret : {:.3f}, Value Loss : {:.3f}, Policy loss : {:.5f}, Policy KL : {:.5f}, Policy Entropy : {:.3f}, Goal Reached : {} ***'.format(update, self.nupdates, mean_ret, vf_loss, pol_loss, pol_kl, pol_entropy, num_goal))
            if mean_ret > -300:
                break
        data = []
        data_list = {}
        traj_num = 0
        for update in range(0, 10):
            trajectories = self.run_policy(self.env, self.policy_func, episodes=100)
            mean_ret = np.mean([np.sum(t['rewards']) for t in trajectories])
            for trajectory in trajectories:
                init_pos = [{'pos': trajectory['observes'][0]}]
                init_pos = init_pos + trajectory['infos']
                trajectory['infos'] = init_pos
                data_list.update({'traj' + str(traj_num): np.stack([info['pos'] for info in trajectory['infos']])})
                data.append(np.stack([info['pos'] for info in trajectory['infos']]))
                traj_num = traj_num + 1

        with open(file_name + ".txt", "wb") as fp:   #Pickling
            pickle.dump(data, fp)
        f.close()
        for i in range(0, 100):
            self.env.render(data[i])
        self.env.pause() 

    def run_episode(self, env, policy, animate=False, evaluation=False): # Run policy and collect (state, action, reward) pairs
        obs = env.reset()
        observes, actions, rewards, infos = [], [], [], []
        done = False
        for i in range(0, 30):
            if done: break
        #while not done:
            #if animate:
            #    env.render()
            obs = obs.astype(np.float32).reshape((1, -1))
            observes.append(obs)
            if evaluation:
                action = policy.control(obs).reshape((1, -1)).astype(np.float32)
            else:
                action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)
            infos.append(info) 
        return (np.concatenate(observes), np.concatenate(actions), np.array(rewards, dtype=np.float32), infos)

    def run_policy(self, env, policy, episodes, evaluation=False): # collect trajectories. if 'evaluation' is ture, then only mean value of policy distribution is used without sampling.
        total_steps = 0
        trajectories = []
        for e in range(episodes):
            observes, actions, rewards, infos = self.run_episode(env, policy, evaluation=evaluation)
            total_steps += observes.shape[0]
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards,
                          'infos': infos}
            #env.render(trajectory)
            trajectories.append(trajectory)
        return trajectories

    def close(self):
        self.sess.close()
