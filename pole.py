# Learn the Cart-Pole problem through basic Neural Network
# The input to the Network would be pole position
# output will be 0 or 1 (i.e. left or right)

import gym
import universe
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from collections import Counter
from statistics import median, mean

learning_rate = 0.001
e = gym.make("CartPole-v0")
e.reset()
steps = 500
score_min = 60
start_games = 20000


# Learning the model by getting the training data by running the model for many times
# This data can be feed to neural net in order to find the weights and classify 
def learn():
    train = []
    scores = []
    eligible_scores = []

    for _ in range(start_games):
        score = 0
        game_mem = []
        prev_observationervation = []

        for _ in range(steps):
            action = random.randrange(0,2)
            observation, reward, done, info = e.step(action)
            
            if len(prev_observationervation) > 0 :
                game_mem.append([prev_observationervation, action])
            prev_observationervation = observation
            score+=reward
            if done: break


        if score >= score_min:
            eligible_scores.append(score)
            for i in game_mem:

                if i[1] == 1:
                    output = [0,1]
                elif i[1] == 0:
                    output = [1,0]
                    
                train.append([i[0], output])

        e.reset()

        scores.append(score)
    
    return train

def neural_network(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.6)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.6)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.6)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.6)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.6)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(train, model=False):

    X = np.array([i[0] for i in train]).reshape(-1,len(train[0][0]),1)
    y = [i[1] for i in train]

    if not model:
        model = neural_network(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500)
    return model

train = learn()

model = train_model(train)

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_mem = []
    prev_observation = []
    e.reset()
    for _ in range(steps):
        e.render()

        if len(prev_observation)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_observation.reshape(-1,len(prev_observation),1))[0])

        choices.append(action)
                
        new_observation, reward, done, info = e.step(action)
        prev_observation = new_observation
        game_mem.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print(score_min)


    
