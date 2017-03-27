"""
Implementation of DDPG - Deep Deterministic Policy Gradient Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 and MountainCarContinuous-v0 OpenAI gym task
"""

import numpy as np
import datetime
import gym
from gym.wrappers import Monitor
import tensorflow as tf
from tqdm import tqdm

from src.agent.ddpg_agent import DDPGAgent
from src.network.ddpg_network import CriticNetwork, ActorNetwork
from src.replaybuffer import ReplayBuffer
from src.explorationnoise import OrnsteinUhlenbeckProcess, GreedyPolicy

flags = tf.app.flags

# ================================
#    UTILITY PARAMETERS
# ================================
# Gym environment name
#'Pendulum-v0''MountainCarContinuous-v0'
flags.DEFINE_string('env_name', 'Pendulum-v0', 'environment name in gym.')
flags.DEFINE_boolean('env_render', False, 'whether render environment (display).')
flags.DEFINE_boolean('env_monitor', True, 'whether use gym monitor.')
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
RANDOM_SEED = 1234


# ================================
#    TRAINING PARAMETERS
# ================================
flags.DEFINE_integer('mini_batch', 64, 'mini batch size for training.')
# Learning rates actor and critic
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
# Maximum number of episodes
MAX_EPISODES = 100000
# Maximum number of steps per episode
MAX_STEPS_EPISODE = 50000
# warmup steps.
WARMUP_STEPS = 10000
# Exploration duration
EXPLORATION_EPISODES = 10000
# Discount factor
GAMMA = 0.99
# Soft target update parameter
TAU = 0.001
# Size of replay buffer
BUFFER_SIZE = 1000000
# Exploration noise variables Ornstein-Uhlenbeck variables
OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3
# Explorationnoise for greedy policy
MIN_EPSILON = 0.1
MAX_EPSILON = 1

#================
# parameters for evaluate.
#================
# evaluate periods
EVAL_PERIODS = 100
# evaluate episodes
EVAL_EPISODES = 10


FLAGS = flags.FLAGS

# Directory for storing gym results
MONITOR_DIR = './results/{}/{}/gym_ddpg'.format(FLAGS.env_name, DATETIME)
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/{}/{}/tf_ddpg'.format(FLAGS.env_name, DATETIME)


# ================================
#    MAIN
# ================================
def main(_):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        env = gym.make(FLAGS.env_name)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        if FLAGS.env_monitor:
            if not FLAGS.env_render:
                env = Monitor(env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = Monitor(env, MONITOR_DIR, force=True)

        state_dim = env.observation_space.shape
        try: 
            action_dim = env.action_space.shape[0]
            action_bound = env.action_space.high
            # Ensure action bound is symmetric
            assert(np.all(env.action_space.high == -env.action_space.low))
            action_type = 'Continuous'
        except:
            action_dim = env.action_space.n
            action_bound = None
            action_type = 'Discrete'

        print(action_type)
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU, action_type)

        critic = CriticNetwork(sess, state_dim, action_dim, action_bound,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), action_type)

        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        if action_type == 'Continuous':
            noise = OrnsteinUhlenbeckProcess(OU_THETA, mu=OU_MU, sigma=OU_SIGMA, n_steps_annealing=EXPLORATION_EPISODES)
        else:
            noise = GreedyPolicy(action_dim, EXPLORATION_EPISODES, MIN_EPSILON, MAX_EPSILON)


        agent = DDPGAgent(sess, action_type, actor, critic, GAMMA, env, replay_buffer, noise=noise, exploration_episodes=EXPLORATION_EPISODES,\
                max_episodes=MAX_EPISODES, max_steps_episode=MAX_STEPS_EPISODE, warmup_steps=WARMUP_STEPS,\
                mini_batch=FLAGS.mini_batch, eval_episodes=EVAL_EPISODES, eval_periods=EVAL_PERIODS, \
                env_render=FLAGS.env_render, summary_dir=SUMMARY_DIR)

        agent.train()

if __name__ == '__main__':
    tf.app.run()

