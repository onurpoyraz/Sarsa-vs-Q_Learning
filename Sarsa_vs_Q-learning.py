import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import patches
import pandas as pd

def chooseNextAction(Q, x, y, is_sarsa = True):
    epsilon = 0.1
    if np.random.random() < epsilon and is_sarsa:
        action = np.random.choice(range(4))
        value = Q[x, y, action]
    else:
        max_value = -np.inf
        for action in range(4):
            value = Q[x,y,action]
            if max_value <= value:
                max_value = value
                max_action = action
        action = max_action
    return action

def chooseNextState(x, y, current_action):
    x_new, y_new = x, y
    if current_action == 0:
        x_new = x + 1
    elif current_action == 1:
        x_new = x - 1
    elif current_action == 2:
        y_new = y + 1
    elif current_action == 3:
        y_new = y - 1
    x_new = max(0, x_new)
    x_new = min(3, x_new)
    y_new = max(0, y_new)
    y_new = min(11, y_new)
    return x_new, y_new

def updateValueQ(Q, x, y, current_action, is_sarsa=False):
    alpha = 0.1
    gamma = 1
    x_new, y_new = chooseNextState(x, y, current_action)
    if x_new == 0 and y_new == 11:
        reward = 0
    elif x_new == 0 and 11 > y_new > 0:
        reward = -500
    else:
        reward = -1
    next_action = chooseNextAction(Q, x_new, y_new, is_sarsa)
    Q[x, y, current_action] = Q[x, y, current_action] + alpha * float((reward + gamma * Q[x_new, y_new, next_action] - Q[x, y, current_action]))
    if x_new == 0 and 11 > y_new > 0:
        return 0, 0, reward
    else:
        return x_new, y_new, reward

def algorithm(is_sarsa=False):
    q_actions_of_last_episode = [(0,0)]
    q_actions_per_episode = []
    q_rewards_per_episode = []
    sarsa_actions_of_last_episode = [(0,0)]
    sarsa_actions_per_episode = []
    sarsa_rewards_per_episode = []
    Q = np.zeros((4, 12, 4))
    for n in range(1000):
        x = 0
        y = 0
        actions = 0
        rewards = 0
        while not(x == 0 and y == 11):
            current_action = chooseNextAction(Q, x, y)
            x_new, y_new, reward = updateValueQ(Q, x, y, current_action, is_sarsa=is_sarsa)
            x = x_new
            y = y_new
            actions = actions + 1
            rewards = rewards + reward
            if n == 999:
                sarsa_actions_of_last_episode.append((x_new, y_new)) if is_sarsa else q_actions_of_last_episode.append((x_new, y_new))
            if x == 0 and y == 0:
                if is_sarsa:
                    sarsa_actions_of_last_episode = [(0, 0)]
                else:
                    q_actions_of_last_episode = [(0, 0)]
        if is_sarsa:
            sarsa_actions_per_episode.append(actions)
            sarsa_rewards_per_episode.append(rewards)
        else:
            q_actions_per_episode.append(actions)
            q_rewards_per_episode.append(rewards)
    if is_sarsa:
        return sarsa_actions_of_last_episode, sarsa_actions_per_episode, sarsa_rewards_per_episode
    else:
        return q_actions_of_last_episode, q_actions_per_episode, q_rewards_per_episode

def plot(q_steps, q_actions, q_rewards, sarsa_steps, sarsa_actions, sarsa_rewards):
    fig1 = plt.figure()
    previous_x = 0
    previous_y = 0
    for position in q_steps:
        x, y = position[1], position[0]
        plt.arrow(previous_x, previous_y, x - previous_x, y - previous_y, head_width = 0.1, head_length = 0.2, color = 'red')
        plt.plot(x, y, 'ro', markersize=1)
        previous_x = x
        previous_y = y

    previous_x = 0
    previous_y = 0
    for position in sarsa_steps:
        x, y = position[1], position[0]
        plt.arrow(previous_x, previous_y, x - previous_x, y - previous_y, head_width = 0.1, head_length = 0.2, color = 'blue')
        plt.plot(x, y, 'bo', markersize=1)
        previous_x = x
        previous_y = y

    plt.plot(0, 0, 'mo', markersize=15)
    plt.plot(11, 0, 'go', markersize=15)
    axes = plt.gca()
    axes.set_xticks(range(0, 12))
    axes.set_yticks(range(0, 4))
    axes.set_title('Paths of Algoritms')
    red_patch = patches.Patch(color = 'red', label = 'Q Learning')
    blue_patch = patches.Patch(color = 'blue', label = 'Sarsa Learning')
    plt.legend(handles=[red_patch, blue_patch])
    plt.grid()
    plt.savefig('paths_of_algoritms.png')
    #plt.show(fig1)

    labels = ['Q learning','Sarsa Learning']

    fig2 = plt.figure(figsize=(10, 6))
    smoothing = 100
    rewards_smoothed = pd.Series(q_rewards).rolling(smoothing, min_periods=smoothing).mean()
    plt.plot(rewards_smoothed, 'r')
    rewards_smoothed = pd.Series(sarsa_rewards).rolling(smoothing, min_periods=smoothing).mean()
    plt.plot(rewards_smoothed, 'b')
    plt.xlabel("Number of Episode")
    plt.ylabel("Reward Per Episode")
    plt.title("Reward Per Episode Over Time (Smoothed)")
    plt.legend(labels)
    plt.savefig('episode_reward.png')
    #plt.show(fig2)

    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(q_actions, 'r')
    plt.plot(sarsa_actions, 'b')
    plt.xlabel("Number of Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend(labels)
    plt.savefig('episode_length.png')
    #plt.show(fig3)

q_steps, q_actions, q_rewards = algorithm()
print "Q-Learning Steps"
for step in q_steps:
    print step

sarsa_steps, sarsa_actions, sarsa_rewards = algorithm(is_sarsa = True)
print "Sarsa-Learning Steps"
for step in sarsa_steps:
    print step

plot(q_steps, q_actions, q_rewards, sarsa_steps, sarsa_actions, sarsa_rewards)
