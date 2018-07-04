import experiment
import tensorflow as tf
for i in range(0, 10):
    for j in range(0, 10):
        exp = experiment.Experiments(seed=0, env_name='MultiGoal', mdn_hidden_spec=None, 
                        v_epochs=50, v_hdim=32, v_lr=1e-3,
                        p_epochs=30, p_hdim=32, p_lr=3e-4, clip_range=0.2, alpha=10.0*i,
                        batch_size=128, episode_size=100, nupdates=20*j,
                        gamma=0.99, max_step=1000)
        exp.run()
        exp.close()