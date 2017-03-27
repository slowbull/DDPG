import tensorflow as tf
import gym
from tqdm import tqdm
import numpy as np
from src.agent.agent import BaseAgent

class DDPGAgent(BaseAgent):
    def __init__(self, sess, action_type, actor, critic, gamma, env, replay_buffer, noise=None, exploration_episodes=10000, max_episodes=10000, max_steps_episode=10000,\
            warmup_steps=5000, mini_batch=32, eval_episodes=10, eval_periods=100, env_render=False, summary_dir=None):
        """
        Deep Deterministic Policy Gradient Agent.
        Args:
            actor: actor network.
            critic: critic network.
            gamma: discount factor.
        """
        super(DDPGAgent, self).__init__(sess, env, replay_buffer, noise=noise, exploration_episodes=exploration_episodes, max_episodes=max_episodes, max_steps_episode=max_steps_episode,\
                warmup_steps=warmup_steps, mini_batch=mini_batch, eval_episodes=eval_episodes, eval_periods=eval_periods, env_render=env_render, summary_dir=summary_dir)

        self.action_type = action_type
        self.actor = actor
        self.critic = critic
        self.gamma = gamma


    def train(self):
        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()

        for cur_episode in tqdm(xrange(self.max_episodes)):

            # evaluate here. 
            if cur_episode % self.eval_periods == 0:
                self.evaluate(cur_episode)

            state = self.env.reset()

            episode_reward = 0
            episode_ave_max_q = 0

            for cur_step in xrange(self.max_steps_episode):

                if self.env_render:
                    self.env.render()

                # Add exploratory noise according to Ornstein-Uhlenbeck process to action
                if self.replay_buffer.size() < self.warmup_steps:
                    action = self.env.action_space.sample()
                else: 
                    if self.action_type == 'Continuous':
                        if cur_episode < self.exploration_episodes and self.noise is not None:
                            action = np.clip(self.actor.predict(np.expand_dims(state, 0))[0] + self.noise.generate(cur_episode), -1, 1) 
                        else: 
                            action = self.actor.predict(np.expand_dims(state, 0))[0] 
                    else:
                        action = self.noise.generate(self.actor.predict(np.expand_dims(state, 0))[0,0], cur_episode)

                next_state, reward, terminal, info = self.env.step(action)

                self.replay_buffer.add(state, action, reward, terminal, next_state)

                # Keep adding experience to the memory until there are at least minibatch size samples
                if self.replay_buffer.size() > self.warmup_steps:
                    state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = \
                        self.replay_buffer.sample_batch(self.mini_batch)

                    # Calculate targets
                    target_q = self.critic.predict_target(next_state_batch, self.actor.predict_target(next_state_batch))

                    y_i = np.reshape(reward_batch, (self.mini_batch, 1)) + (1 \
                            - np.reshape(terminal_batch, (self.mini_batch, 1)).astype(float))\
                            * self.gamma * np.reshape(target_q, (self.mini_batch, 1))

                    # Update the critic given the targets
                    if self.action_type == 'Discrete':
                        action_batch = np.reshape(action_batch, [self.mini_batch, 1])
                    predicted_q_value, _ = self.critic.train(state_batch, action_batch, y_i)

                    episode_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    if self.action_type == 'Continuous':
                        a_outs = self.actor.predict(state_batch)
                        a_grads = self.critic.action_gradients(state_batch, a_outs)
                        self.actor.train(state_batch, a_grads[0])
                    else:
                        a_outs = self.actor.predict(state_batch)
                        a_grads = self.critic.action_gradients(state_batch, a_outs)
                        self.actor.train(state_batch, a_grads[0])


                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                state = next_state
                episode_reward += reward

                if terminal or cur_step == self.max_steps_episode-1:
                    train_episode_summary = tf.Summary() 
                    train_episode_summary.value.add(simple_value=episode_reward, tag="train/episode_reward")
                    train_episode_summary.value.add(simple_value=episode_ave_max_q/float(cur_step), tag="train/episode_ave_max_q")
                    self.writer.add_summary(train_episode_summary, cur_episode)
                    self.writer.flush()

                    print 'Reward: %.2i' % int(episode_reward), ' | Episode', cur_episode, \
                          '| Qmax: %.4f' % (episode_ave_max_q / float(cur_step))

                    break


    def evaluate(self, cur_episode):
        # evaluate here. 
        total_episode_reward = 0 
        for eval_i in xrange(self.eval_episodes):
            state = self.env.reset() 
            terminal = False
            while not terminal:
                if self.action_type == 'Continuous':
                    action = self.actor.predict(np.expand_dims(state, 0))[0]
                else:
                    action = self.actor.predict(np.expand_dims(state, 0))[0,0]
                state, reward, terminal, info = self.env.step(action)
                total_episode_reward += reward
        ave_episode_reward = total_episode_reward / float(self.eval_episodes)
        print("\nAverage reward {}\n".format(ave_episode_reward))
        # Add ave reward to Tensorboard
        eval_episode_summary = tf.Summary()
        eval_episode_summary.value.add(simple_value=ave_episode_reward, tag="eval/reward")
        self.writer.add_summary(eval_episode_summary, cur_episode)

