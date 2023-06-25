import gym
import numpy as np
import random
import keras
import cv2
import math
from collections import deque
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

#discount rate of Q(s,a)
DISCOUNT_RATE = 0.99
#extract last 3 frams in games in order to consider enemy's bullets early.
NUM_FRAMES = 3

class DQN(object):
    #deep learning
    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        #build Q network used to calculate Q value
        self.Q_network = Sequential()
        self.Q_network.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES))) #input image:84*84
        self.Q_network.add(Activation('relu'))
        self.Q_network.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.Q_network.add(Activation('relu'))
        self.Q_network.add(Convolution2D(64, 3, 3))
        self.Q_network.add(Activation('relu'))
        self.Q_network.add(Flatten())
        self.Q_network.add(Dense(512))
        self.Q_network.add(Activation('relu'))
        self.Q_network.add(Dense(6))
        self.Q_network.compile(loss='mse', optimizer=Adam(lr=0.00001))

        #build target network used to select action
        self.target_network = Sequential()
        self.target_network.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.target_network.add(Activation('relu'))
        self.target_network.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_network.add(Activation('relu'))
        self.target_network.add(Convolution2D(64, 3, 3))
        self.target_network.add(Activation('relu'))
        self.target_network.add(Flatten())
        self.target_network.add(Dense(512))
        self.target_network.add(Activation('relu'))
        self.target_network.add(Dense(6))
        self.target_network.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_network.set_weights(self.Q_network.get_weights())

        print("Successfully constructed networks.")

    def epsilon_greedy(self, data, epsilon):
        #epsilon_greedy algorithm used to keep balance of exploration and exploritation
        q_actions = self.Q_network.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, 6)
        return opt_policy, q_actions[0, opt_policy]

    def mellomax(self,actions,size):
        #temperature parameter omega=2 in this case, explore the value of omega in Space Invader is a good future work
        omiga=2
        sum_actions=0
        for i in range(size):
            sum_actions=sum_actions+math.exp(actions[i])
        return math.log(1/size*sum_actions)/omiga


    def train(self, state, action, reward, over, next_state, observation_num):
        #train Q-network and target network
        batch_size = state.shape[0]
        size2=batch_size
        targets = np.zeros((batch_size, 6))

        for i in range(batch_size):
            targets[i] = self.Q_network.predict(state[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            fut_action = self.target_network.predict(next_state[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            targets[i, action[i]] = reward[i]
            if over[i] == False:
                targets[i,action[i]]+=DECAY_RATE*self.mellomax(fut_action[0],len(fut_action[0]))
                #use below commnet to set the max opertor of Double DQN
                #targets[i, action[i]] += DISCOUNT_RATE * np.max(fut_action)



        loss = self.Q_network.train_on_batch(state, targets)
        #it is used to print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print("We had a loss equal to ", loss)

class Experience_replay:
    #Experience_replay is the special characteristic of DQN model, it is used to decrease correlation of samples 

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        #Add an experience to the buffer"""
        # s is current state, a is action,r is reward, d is whether it is the end, and s2 is next state
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        #when the size of buffer equal to threshold, samples part of elements equal to batch_size from buffer
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch, a_batch, r_batch, d_batch, s2_batch = list(map(np.array, list(zip(*batch))))

        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
