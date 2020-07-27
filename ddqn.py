# dqn.py
# https://geektutu.com
from collections import deque
import random
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers
from DDQN.puckworld import PuckWorldEnv
import tensorflow as tf
import os
import pandas as pd
import time
class DQN(object):
    def __init__(self,env):
        self.step = 0
        self.update_freq = 80  # 模型更新频率
        self.replay_size = 100000  # 训练集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.model.summary()
        self.target_model = self.create_model()
        self.env = env
        self.epsilon=0.99
        self.today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        self.no = '-2'
        self.path = "./objects/DDQN" + self.today + self.no + "/"
        self.actionPath = self.path + "action.csv"
        self.rewardPath = self.path + "reward.csv"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            print(self.path+"has existed")
        self.action_df = pd.read_csv(self.actionPath) if os.path.isfile(self.actionPath) else pd.DataFrame(
            columns=['step_counter', 'action'])

        self.reward_df = pd.read_csv(self.rewardPath) if os.path.isfile(self.rewardPath) else pd.DataFrame(
            columns=['episodes', 'reward'])

    def create_model(self):
        """创建一个隐藏层为100的神经网络"""
        STATE_DIM, ACTION_DIM = 6, 5
        model = models.Sequential([
            layers.Dense(64, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(128, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        # model.compile(loss='mean_squared_error',
        #               optimizer=optimizers.Adam(0.001))

        model.compile(loss=tf.losses.mean_squared_error,
                      optimizer=optimizers.Adam(0.001))
        return model

    def act(self, s):
        """预测动作"""
        # 刚开始时，加一点随机成分，产生更多的状态
        # if np.random.uniform() < epsilon - self.step * 0.0002:
        #     return
        # return np.argmax(self.model.predict(np.array([s]))[0])

        print("选行为")
        if np.random.uniform() < self.epsilon:
            # print('000000000',self.env.action_space.sample())
            action =self.env.action_space.sample()
            print("随机选取行为{}，epsilon{}".format(action, self.epsilon))
            # return action
        else:
            # 使用预测值    返回，回报最大到最大的那个
            print("预测行为的状态{}".format(s))
            act_values = self.model.predict(s)
            print("所有的行为{}".format(act_values))
            action  = np.argmax(act_values[0])
            print("最大行为{}，epsilon{}".format(action, self.epsilon))

        return action

    def save_model(self, file_path='ddqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        """历史记录，position >= 0.4时给额外的reward，快速收敛"""
        # if next_s[0] >= 0.4:
        #     reward += 1
        self.replay_queue.append((s, a, next_s, reward))

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


# env = gym.make('MountainCar-v0')
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