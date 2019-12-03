# Clase para cálculo de media y varianza de una secuencia
from time import time
import pandas as pd
import gym
import numpy as np
import moviepy.editor as mpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import keras.backend as K
from tensorboardX import SummaryWriter
import sklearn
import sklearn.preprocessing

def format_as_pandas(time_step, obs, preds, actions, rewards, disc_sum_rews, ep_returns, decimals = 3):
    df = pd.DataFrame({'step': time_step.reshape(-1)})
    df['observation'] = [np.array(r*10**decimals, dtype=int)/(10**decimals) for r in obs]
    df['policy_distribution']=[np.array(r*10**decimals, dtype=int)/(10**decimals) for r in preds]
    df['sampled_action'] = [np.array(r, dtype=int) for r in actions]

    df['rewards']=rewards
    df['discounted_sum_rewards']=np.array(disc_sum_rews*10**decimals, dtype=int)/(10**decimals)
    df['episode_return']=np.array(ep_returns*10**decimals, dtype=int)/(10**decimals)
    return df

class BaseAgent:
    def __init__(self, ENV, logdir_root='logs', n_experience_episodes=1, gamma=0.999, epochs=1, lr=0.001, hidden_layer_neurons=128, EPISODES=2000, eval_period=50, algorithm='REINFORCE', noise=1.0):
        self.noise = noise
        self.last_eval = 0
        self.best_return = -np.inf
        self.eval_period = 50
        self.writer = None
        self.epsilon = 1e-12
        self.logdir_root = logdir_root
        self.EPISODES = EPISODES
        self.n_experience_episodes = n_experience_episodes
        self.episode = 0
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.logdir = self.get_log_name(ENV, algorithm, logdir_root)
        self.env = gym.make(ENV)
        
        if type(self.env.action_space) != gym.spaces.box.Box:
            self.nA = self.env.action_space.n
        else:
            print('Warning: El espacio de acción es continuo')
            self.nA = self.env.action_space.shape[0]
            self.logdir = self.logdir + '_' +  str(self.noise)
            
        if type(self.env.observation_space) == gym.spaces.box.Box:
            self.nS = self.env.observation_space.shape[0]
        else:
            print('Warning: El espacio de observación no es continuo')
        self.model = self.get_policy_model(lr=lr, hidden_layer_neurons=hidden_layer_neurons, input_shape=[self.nS] ,output_shape=self.nA)
        
        
        state_space_samples = np.array(
            [self.env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(state_space_samples)
        
        self.reset_env()
        
    def get_policy_model(self, lr=0.001, hidden_layer_neurons = 128, input_shape=[4], output_shape=2):
        pass
        
    def get_log_name(self,ENV, algorithm, logdir_root):
        name = logdir_root + '/'
        name += ENV + '/' + algorithm + '/'
        name += str(self.n_experience_episodes) + '_'
        name += str(self.epochs) + '_'
        name += str(self.gamma) + '_'
        name += str(self.lr) + '_'  + str(int(time()))
        return name
    
    def reset_env(self):
        # Se suma uno a la cantidad de episodios
        self.episode += 1
        # Se observa el primer estado
        self.observation = self.env.reset()
        # Se resetea la lista con los rewards
        self.reward = []
        
    def get_experience_episodes(self, return_ts=False):
        # Antes de llamar esta función hay que asegurarse de que el env esta reseteado
        observations = []
        actions = []
        predictions = []
        rewards = []
        discounted_rewards = []
        episodes_returns = []
        episodes_lenghts = []
        time_steps = []
        exp_episodes = 0
        ts_count = 0
        # Juega n_experience_episodes episodios
        while exp_episodes < self.n_experience_episodes:
            # Obtengo acción
            action, action_one_hot, prediction = self.get_action(eval=False)
            
            # Ejecuto acción
            observation, reward, done, info = self.env.step(action)
            
            # Guardo reward obtenido por acción
            self.reward.append(reward)

            # Notar que se guarda la observación anterior
            observations.append(self.observation)
            
            actions.append(action_one_hot)
            predictions.append(prediction.flatten())
            rewards.append(reward)
            self.observation = observation
            ts_count+=1
            time_steps.append(ts_count)
            if done:
                exp_episodes += 1
                discounted_reward = self.get_discounted_rewards(self.reward)
                discounted_rewards = np.hstack([discounted_rewards, discounted_reward])
                ep_len = len(discounted_reward)
                episodes_lenghts.append(ep_len)
                episodes_returns = episodes_returns + [discounted_reward[0]]*ep_len
                self.last_observation = self.observation
                self.reset_env()
                ts_count = 0
        if return_ts:
            return np.array(observations), np.array(actions), np.array(predictions), np.array(discounted_rewards), np.array(rewards), np.array(episodes_returns), np.array(episodes_lenghts), self.last_observation, np.array(time_steps).reshape(-1, 1)
        else:
            return np.array(observations), np.array(actions), np.array(predictions), np.array(discounted_rewards), np.array(rewards), np.array(episodes_returns), np.array(episodes_lenghts), self.last_observation
        
    
    def log_data(self, episode, loss, ep_len_mean, entropy, rv, nomalized_loss, deltaT, ep_return, critic_loss=None):
        if self.writer is None:
            self.writer = SummaryWriter(self.logdir)
            print(f"correr en linea de comando: tensorboard --logdir {self.logdir_root}/")
            
        print(f'\rEpisode: {episode}', end='')
        self.writer.add_scalar('loss', loss, episode)
        self.writer.add_scalar('episode_len', ep_len_mean, episode)
        self.writer.add_scalar('entropy', entropy, episode)
        self.writer.add_scalar('running_var', rv, episode)
        self.writer.add_scalar('episode_return', ep_return, episode)
        if nomalized_loss is not None:
            self.writer.add_scalar('normalized_loss', nomalized_loss, episode)
        self.writer.add_scalar('time', deltaT, episode)
        if critic_loss is not None:
            self.writer.add_scalar('critic_loss', critic_loss, episode)
        if self.episode - self.last_eval >= self.eval_period:
            obs, actions, preds, disc_sum_rews, rewards, ep_returns, ep_len = self.get_eval_episode()
            if self.best_return <= ep_returns[-1]:
                self.model.save(self.logdir + '.hdf5')
                print()
                print(f'Model on episode {self.episode - 1} improved from {self.best_return} to {ep_returns[-1]}. Saved!')
                self.best_return = ep_returns[-1]
            self.writer.add_scalar('eval_episode_steps', len(obs), self.episode)
            self.writer.add_scalar('eval_episode_return', ep_returns[-1], episode)
            self.last_eval = self.episode
            
    def get_eval_episode(self, gif_name=None, fps=50):
        frames=[]
        self.reset_env()
        observations = []
        actions = []
        predictions = []
        rewards = []
        discounted_rewards = []
        episodes_returns = []
        episodes_lenghts = []
        exp_episodes = 0
        if gif_name is not None:
            frames.append(self.env.render(mode = 'rgb_array'))
        while True:
            # Juega episodios hasta juntar un tamaño de buffer mínimo
            action, action_one_hot, prediction = self.get_action(eval=True)
            
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            # Notar que se guarda la observación anterior
            observations.append(self.observation)
            actions.append(action_one_hot)
            predictions.append(prediction.flatten())
            rewards.append(reward)
            self.observation = observation
            if gif_name is not None:
                frames.append(self.env.render(mode = 'rgb_array'))
            if done:
                exp_episodes += 1
                discounted_reward = self.get_discounted_rewards(self.reward)
                discounted_rewards = np.hstack([discounted_rewards, discounted_reward])
                ep_len = len(discounted_reward)
                episodes_lenghts.append(ep_len)
                episodes_returns = episodes_returns + [discounted_reward[0]]*ep_len
                self.reset_env()
                if gif_name is not None:
                    clip = mpy.ImageSequenceClip(frames, fps=fps)
                    clip.write_gif(gif_name, fps=fps, verbose=False, logger=None)
                return np.array(observations), np.array(actions), np.array(predictions), np.array(discounted_rewards), np.array(rewards), np.array(episodes_returns), np.array(episodes_lenghts)

class RunningVariance:
    # Keeps a running estimate of variance

    def __init__(self):
        self.m_k = None
        self.s_k = None
        self.k = None

    def add(self, x):
        if not self.m_k:
            self.m_k = x
            self.s_k = 0
            self.k = 0
        else:
            old_mk = self.m_k
            self.k += 1
            self.m_k += (x - self.m_k) / self.k
            self.s_k += (x - old_mk) * (x - self.m_k)

    def get_variance(self, epsilon=1e-12):
        return self.s_k / (self.k - 1 + epsilon) + epsilon
    
    def get_mean(self):
        return self.m_k