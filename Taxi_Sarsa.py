import gym
import numpy as np
import matplotlib.pyplot as plt


class Sarsa:
    '''Sarsa算法'''
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.05):
        self.gamma = gamma # 折扣因子
        self.alpha = alpha # 学习率
        self.epsilon = epsilon # epsilon-贪婪策略中参数
        self.n_action = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    # 决策 epsilon-greedy
    def decide(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.q[state])
        return action

    # 更新策略
    def update(self, state, action, reward, next_s, done, next_a):
        td_error = reward + self.gamma*self.q[next_s,next_a]*(1-done) - self.q[state, action]
        self.q[state, action] += self.alpha * td_error

def play(env, agent, train=False):
    '''play一回合(episode)'''
    episode_reward = 0
    observation, info= env.reset()
    action = agent.decide(observation)
    while True:
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Reward shaping
        width = 5
        height = 5
        locations = {0:[0,0],1:[0,4],2:[4,0],3:[4,3]} # 每个地点对应的坐标
        x, y, p, d = env.unwrapped.decode(next_observation)
        if p != 4: # 乘客不在Taxi上
            # 计算对应的哈密顿距离
            length = np.double(np.abs(x-locations[p][0]) + np.abs(y-locations[p][1]))
            F = 0.1*(1-length/(width+height))
        else: # 乘客在车上,length=0
            F = 0.1
        reward += F

        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.update(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward


if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make("Taxi-v3")#, render_mode="human")
    env.reset()
    agent = Sarsa(env)
    episodes = 4000
    episode_rewards = []
    for episode in range(episodes):
        episode_reward = play(env, agent, train=True)
        episode_rewards.append(episode_reward)
    # env.close()
    fig,  ax = plt.subplots()
    ax.plot(episode_rewards)
    ax.set_title('Taxi by Saras with Reward shaping')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    # fig.savefig('return of Taxi by Saras with Reward shaping.svg')
    
    # 测试
    agent.epsilon = 0 # 确定性策略
    episode_rewards = []
    # env = gym.make("Taxi-v3", render_mode="human")
    for i in range(10):
        episode_reward = play(env, agent)
        episode_rewards.append(episode_reward)
        print("测试第{:d}次奖励是{:.4f}".format(i, episode_reward))
    
    print("平均奖励：{:.4f}".format(np.mean(episode_rewards)))
    print("q={}".format(agent.q))
    
    plt.show()

    env.close()
