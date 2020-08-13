import gym
import tensorflow as tf
import numpy as np
from collections import deque
import random
from tf21_RL.PuckWorld.puckworld import PuckWorldEnv
import time
import os
import pandas as pd
# %%
class DQN():
    def __init__(self, env):
        # hyper params and constants
        self.epsilon = 0.99
        self.update_freq = 80  # 模型更新频率
        self.step = 0
        self.weight_backup = './DQN_Agent_Lunar.hdf5'
        self.learning_rate = 0.001
        self.exploration_rate = 1.0
        self.gamma = 0.95
        self.env = env
        self.replay_size = 100000  # 训练集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.exploration_min = 0.01
        self.expolration_decay = 0.990
        self.update_target_freq = 80
        self.LReluLAlpha = 0.01  # keras default is 0.3
        self.TrainEvery = 100  # steps beetween every training
        self.debug = True
        self.MemorySize = 10000

        # env variables
        self.action_space = env.action_space.shape
        self.action_space_size = env.action_space.n
        self.state_space = env.observation_space.shape
        self.state_space_size = env.observation_space.shape[0]
        self.memory = deque(maxlen=self.MemorySize)

        # create the model and the target model
        self.model = self.GenModelDuelingDQN()  # alternativly: use simple DQN: self.GenModelDQN()
        self.target_model = self.GenModelDuelingDQN()  # alternativly: use simple DQN: self.GenModelDQN()
        self.today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        self.no = '-2'
        self.path = "./objects/DDQN" + self.today + self.no + "/"
        self.actionPath = self.path + "action.csv"
        self.rewardPath = self.path + "reward.csv"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            print(self.path + "has existed")
        self.action_df = pd.read_csv(self.actionPath) if os.path.isfile(self.actionPath) else pd.DataFrame(
            columns=['step_counter', 'action'])

        self.reward_df = pd.read_csv(self.rewardPath) if os.path.isfile(self.rewardPath) else pd.DataFrame(
            columns=['episodes', 'reward'])
    def GenModelDuelingDQN(self):
        # define input
        input_node = tf.keras.Input(shape=self.state_space)

        # If there should be a pre-prcessing NN (like a conv2d-stack) here is the place.
        input_layer = input_node

        # define state value function
        state_value = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=self.LReluLAlpha))(
            input_layer)
        state_value = tf.keras.layers.Dense(1, activation='linear')(state_value)
        # state value and action value need to have the same shape for merging (adding)
        state_value = tf.keras.layers.Lambda(
            lambda s: tf.keras.backend.expand_dims(s[:, 0], axis=-1),
            output_shape=(self.action_space_size,))(state_value)

        # define acion advantage
        action_advantage = tf.keras.layers.Dense(40, activation=tf.keras.layers.LeakyReLU(alpha=self.LReluLAlpha))(
            input_layer)
        # action_advantage = tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=self.LReluLAlpha))(
        #     action_advantage)
        # action_advantage = tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=self.LReluLAlpha))(
        #     action_advantage)
        action_advantage = tf.keras.layers.Dense(self.action_space_size, activation='linear')(action_advantage)
        # See Dueling_DQN Paper
        action_advantage = tf.keras.layers.Lambda(
            lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),
            output_shape=(self.action_space_size,))(action_advantage)

        # merge by adding
        Q = tf.keras.layers.add([state_value, action_advantage])

        # define model
        model = tf.keras.Model(inputs=input_node, outputs=Q)

        # compile
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')

        # print a model summary
        if self.debug:
            print(model.summary())
            # time.sleep(1000)

        # return
        return model

    def GenModelDQN(self):
        # create sequential fully connected NN
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(40, input_shape=self.state_space,
                                  activation=tf.keras.layers.LeakyReLU(alpha=self.LReluLAlpha)),
            tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=self.LReluLAlpha)),
            tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU(alpha=self.LReluLAlpha)),
            tf.keras.layers.Dense(self.action_space_size, activation='linear')
        ])

        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')

        # print a model summary
        if self.debug:
            print(model.summary())

        # Return
        return model

    def save_model(self):
        self.model.save_weights(self.weight_backup)

    # def act(self, state, override_random=False):
    #     # update exploration_rate:
    #     if self.exploration_rate > self.exploration_min:
    #         self.exploration_rate *= self.expolration_decay
    #     # Get either random action:
    #     if np.random.rand() <= self.exploration_rate and not override_random:
    #         ret = np.random.randint(self.action_space_size)
    #         if ret >= self.action_space_size: print(f"random {ret}")
    #         return ret
    #     # Or the best guess according to the Neural Net representation of Q(s,a):
    #     act_values = self.model.predict(state)
    #     ret = np.argmax(act_values[0])
    #     if ret >= self.action_space_size: print(f"nn: {ret} act_values: {act_values}")
    #     return ret
    def act(self, s):
        """预测动作"""
        # 刚开始时，加一点随机成分，产生更多的状态
        # if np.random.uniform() < epsilon - self.step * 0.0002:
        #     return
        # return np.argmax(self.model.predict(np.array([s]))[0])

        print("选行为")
        if np.random.uniform() < self.epsilon:
            # print('000000000',self.env.action_space.sample())
            action = self.env.action_space.sample()
            print("随机选取行为{}，epsilon{}".format(action, self.epsilon))
            # return action
        else:
            # 使用预测值    返回，回报最大到最大的那个
            print("预测行为的状态{}".format(s))
            act_values = self.model.predict(s)
            print("所有的行为{}".format(act_values))
            action = np.argmax(act_values[0])
            print("最大行为{}，epsilon{}".format(action, self.epsilon))

        return action
    # def remember(self, state, action, reward, next_state, done, t):
    #     self.memory.append((state, action, reward, next_state, done))
    #     # After some time interval update the target model to be same with model
    #     if t % self.update_target_freq == 0:
    #         self.target_model.set_weights(self.model.get_weights())
    def remember(self, s, a, next_s, reward):
        """历史记录，position >= 0.4时给额外的reward，快速收敛"""
        # if next_s[0] >= 0.4:
        #     reward += 1
        self.replay_queue.append((s, a, next_s, reward))
    def replay_batch(self, sample_batch_size):
        # dont start learning before queue has enough samples
        if len(self.memory) < sample_batch_size:
            return

        # create emty bachtes
        X_batch = np.empty((0, self.state_space_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_space_size), dtype=np.float64)

        # get random samples from memory
        num_samples = min(sample_batch_size, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        # work though the samples and generate the batch
        for state, action, reward, next_state, done in replay_samples:
            Q = self.model.predict(state)
            Q_NewState = self.target_model.predict(next_state)

            # Calc the target. This is "the core" of the Q learning algorithm
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(Q_NewState)

            # This Y is the previous Q but with the action updated to the new target value
            Y = Q.copy()
            Y[action] = target
            Y_batch = np.append(Y_batch, np.array([Y]))

            # State is the input (=X)
            X_batch = np.append(X_batch, np.array([state.copy()]))

            # add terminal state if this sample is an episode end
            if done:
                X_batch = np.append(X_batch, np.array([next_state.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[reward] * 1]), axis=0)

        # Finally, fit the model with the generated mini_batch
        self.model.fit(X_batch, Y_batch, batch_size=num_samples, epochs=1, verbose=0)

        # Aftermath: Adapt the exploration rate after each learning
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.expolration_decay

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:  # dont start sampling before queue has enough samples
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            # print(f"{state.T} {state.shape} {next_state.T} {next_state.shape}")
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.expolration_decay

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        # 每 update_freq 步，将 model 的权重赋值给 target_model
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)

        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])
        s_batch = np.squeeze(s_batch)
        next_s_batch = np.squeeze(next_s_batch)
        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        # 传入网络进行训练
        self.model.fit(s_batch, Q, verbose=0)
        if self.epsilon<=0.005:
            self.epsilon = 0.005
        else:
            self.epsilon-=0.00003
# %%
# ----------------------MAIN---------------------
# if __name__ == "__main__":
# ENV_NAME = 'LunarLander-v2'
# EPISODES = 1000
# SAMPLE_BATCH_SIZE = 64
# env = PuckWorldEnv()
# # create environment and agent
# # env = gym.make(ENV_NAME)
# agent = DQN_Agent(env)
#
# # init running variables
# results_lst = []
# index = 0
#
# # main loop
# for episode in range(EPISODES):
#     state = env._reset()
#     state = np.reshape(state, [1,
#                                agent.state_space_size])  # don't understand why but otherwise tf.keras does sometimes not like the shape
#     done = False
#     total_reward = 0
#     while not done:
#         action = agent.act(state)  # get action, play epsion greedy
#         if action > agent.action_space_size - 1 or action < 0:
#             print(f"WARNING: ACITON OUT OF BOUND: {action}")
#         next_state, reward, done, _ = env._step(action)  # use action to get S'R from env
#         next_state = np.reshape(next_state, [1,
#                                              agent.state_space_size])  # don't understand why but otherwise tf.keras does sometimes not like the shape
#         agent.remember(state, action, reward, next_state, done, index)  # add tuple to memory buffer
#         state = next_state
#         total_reward += reward
#         # Train from time to time when enough new experiences are in the agent.memory:
#         if index > agent.MemorySize and index % agent.TrainEvery == 0:
#             agent.replay(SAMPLE_BATCH_SIZE)
#             # render sometimes against bordom
#         # if episode % 50 == 0:
#         env._render()
#         index += 1
#
#     results_lst.append(total_reward)
#     if episode % 50 == 0:  # show current score.
#         mean_score = np.mean(results_lst[-50:])
#         print(f"Episode: {episode}  Score This Episode: {total_reward}  Mean Score last 50 episodes: {mean_score}")
#         if mean_score > np.mean(results_lst[-100:-50]):
#             # agent.save_model() #has a bug.
#             pass
#
# agent.save_model()



env = PuckWorldEnv()
episodes = 2000  # 训练1000次
score_list = []  # 记录所有分数
agent = DQN(env)
step = 0
import time
for i in range(episodes):
    s = env._reset()
    score = 0
    while True:
        print(s)
        env._render()
        s = np.reshape(s, [1, 6])
        # time.sleep(1000)
        a = agent.act(s)
        step+=1
        agent.action_df.loc[len(agent.action_df)] = step, a

        next_s, reward, done, _ = env._step(a)

        next_s = np.reshape(next_s, [1, 6])
        agent.remember(s, a, next_s, reward)
        if i>10:
            agent.train()
        score += reward
        s = next_s
        if done:
            score_list.append(score)
            agent.reward_df.loc[len(agent.reward_df)] = i, score
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
        if i!=0 and i%50==0:
            agent.action_df.to_csv(agent.actionPath, index=False)
            agent.reward_df.to_csv(agent.rewardPath, index=False)
    # 最后10次的平均分大于 -160 时，停止并保存模型
    if i!=0 and i%500==0:
        agent.save_model()
env.close()

import matplotlib.pyplot as plt

plt.plot(score_list, color='green')
plt.show()