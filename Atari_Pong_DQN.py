import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import collections
import random

class ReplayBuffer:
    '''经验回放池'''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 队列,先进先出

    def add(self, state, action, reward, next_state, done): # 将数据加入buffer
        self.buffer.append( (state, action, reward, next_state, done) )

    def sample(self, batch_size): # 从buffer中采样数据，数量为batch_size
        # 随机采样,transitions为列表 [(),(),..]，每个元素是一个采样()
        transitions = random.sample(self.buffer, batch_size)
        # *transitions将其解开为单个的一个个元素 (),(),.. ,再zip把每个元素对应位置打包到每个类别
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self): # 目前buffer中数据的数量
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    '''Q网络'''
    def __init__(self, dim_state, n_hidden1, n_hidden2, n_action):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim_state, n_hidden1)
        self.fc2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = torch.nn.Linear(n_hidden2, n_action)

    def forward(self, x):
        h1 = torch.nn.functional.relu(self.fc1(x))
        h2 = torch.nn.functional.relu(self.fc2(h1))
        return self.fc3(h2)
    
class DQN:
    '''DQN算法'''
    def __init__(self, dim_state, n_hidden1, n_hidden2, n_action, learning_rate, 
                 gamma, epsilon, target_update, device):
        self.dim_state = dim_state
        self.n_action = n_action
        # Q网络
        self.q_net = Qnet(self.dim_state, n_hidden1, n_hidden2, self.n_action).to(device)
        # 目标网络
        self.target_q_net = Qnet(self.dim_state, n_hidden1, n_hidden2, self.n_action).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict()) # 初始时其参数相同
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # epsilon-greedy
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 计数器，记录更新次数
        self.device = device

    # 决策 epsilon-greedy
    def decide(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            state = torch.tensor([state], dtype=torch.float).view(-1,self.dim_state).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    # 更新策略
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).view(-1,self.dim_state).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                              dtype=torch.float).view(-1,self.dim_state).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1,1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions) # q值
        # 下个状态的最大q值
        max_next_q_value = self.target_q_net(next_states).max(1)[0].view(-1,1)
        q_targets = rewards + self.gamma*max_next_q_value*(1-dones) # TD目标
        # 均方误差损失函数
        dqn_loss = torch.mean(torch.nn.functional.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward() # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # 更新目标网络
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
    
def play(env, agent, train=False):
    '''play一回合'''
    episode_reward = 0
    observation, info= env.reset()
    observation = preprocess(observation)
    while True:
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_observation = preprocess(next_observation)
        done = terminated or truncated
        episode_reward += reward
        if done:
            break
        observation = next_observation
    return episode_reward


def preprocess(image):
    '''图像预处理'''
    image = image[35:195] # 裁剪
    image = image[::2, ::2]  # 下采样，缩放2倍
    image = image.reshape(-1, 1)
    return image


if __name__ == "__main__":
    lr = 2e-4
    episodes = 3000
    n_hidden1 = 256
    n_hidden2 = 64
    gamma = 0.99
    epsilon = 0.05
    target_update = 10
    buffer_size = 100000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    env = gym.make("ALE/Pong-v5", obs_type='grayscale')#, render_mode="human")
    state, _ = env.reset(seed=0)
    state = preprocess(state)
    dim_state = len(state)
    n_action = env.action_space.n

    replay_buffer = ReplayBuffer(buffer_size)

    agent = DQN(dim_state, n_hidden1, n_hidden2, n_action, lr, gamma, 
                epsilon, target_update, device)
    
    episode_rewards = []
    # 训练
    for episode in range(episodes):
        episode_reward = 0
        state, info= env.reset()
        state = preprocess(state)
        done = False
        while not done:
            action = agent.decide(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess(next_state)
            done = terminated or truncated
            
            replay_buffer.add(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            # 当buffer数据的数量达到一定值后，进行Q网络训练
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)
            
        episode_rewards.append(episode_reward)
        print("{:d}episode is over, and the episode_reward is {:.4f}".format(episode, episode_reward))
    # env.close()
    fig,  ax = plt.subplots()
    ax.plot(episode_rewards)
    ax.set_title('Pong by DQN')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    fig.savefig('return of Pong by DQN.svg')

    # 保存神经网络模型
    torch.save(agent.q_net.state_dict(), 'Pong_Qnet.pt')
    '''
    再次打开，可以先创建一个Qnet类，然后实例化，再载入参数，eg:
    class Qnet:
            ...
    myQnet = Qnet()
    myQnet.load_state_dict(torch.load('Pong_Qnet.pt'))
    '''
    
    # 测试
    agent.epsilon = 0 # 确定性策略
    episode_rewards = []
    # env = gym.make("ALE/Pong-v5", render_mode="human")
    for i in range(10):
        episode_reward = play(env, agent)
        episode_rewards.append(episode_reward)
        print("测试第{:d}次奖励是{:.4f}".format(i, episode_reward))
    
    print("平均奖励：{:.4f}".format(np.mean(episode_rewards)))
    
    plt.show()

    env.close()


