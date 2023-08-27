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
    '''2层隐藏层的Q网络'''
    def __init__(self, dim_state, n_hidden, n_action):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim_state, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden)
        self.fc3 = torch.nn.Linear(n_hidden, n_action)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN:
    '''DQN算法'''
    def __init__(self, dim_state, n_hidden, n_action, learning_rate, 
                 gamma, epsilon, target_update, device):
        self.dim_state = dim_state
        self.n_action = n_action
        # Q网络
        self.q_net = Qnet(self.dim_state, n_hidden, self.n_action).to(device)
        # 目标网络
        self.target_q_net = Qnet(self.dim_state, n_hidden, self.n_action).to(device)
        # self.target_q_net.load_state_dict(self.q_net.state_dict()) # 初始时其参数相同
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
    while True:
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        if done:
            break
        observation = next_observation
    return episode_reward
    

if __name__ == "__main__":
    lr = 0.01
    episodes = 300
    n_hidden = 128
    gamma = 0.90
    epsilon = 0.05
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    env = gym.make("Taxi-v3")#, render_mode="human")
    # env = gym.make('CartPole-v0')
    env.reset(seed=0)

    replay_buffer = ReplayBuffer(buffer_size)
    dim_state = 1
    # dim_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    agent = DQN(dim_state, n_hidden, n_action, lr, gamma, 
                epsilon, target_update, device)
    
    episode_rewards = []
    # 训练
    for episode in range(episodes):
        episode_reward = 0
        state, info= env.reset()
        done = False
        while not done:
            action = agent.decide(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # # Reward shaping
            # width = 5
            # height = 5
            # locations = {0:[0,0],1:[0,4],2:[4,0],3:[4,3]} # 每个地点对应的坐标
            # x, y, p, d = env.unwrapped.decode(next_state)
            # if p != 4: # 乘客不在Taxi上
            #     # 计算对应的哈密顿距离
            #     length = np.double(np.abs(x-locations[p][0]) + np.abs(y-locations[p][1]))
            #     F = 0.1*(1-length/(width+height))
            # else: # 乘客在车上,length=0
            #     F = 0.1
            # reward += F
            
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
    # env.close()
    fig,  ax = plt.subplots()
    ax.plot(episode_rewards)
    ax.set_title('Taxi by DQN')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    # fig.savefig('return of Taxi by DQN with Reward shaping.svg')
    
    # 测试
    agent.epsilon = 0 # 确定性策略
    episode_rewards = []
    # env = gym.make("Taxi-v3", render_mode="human")
    for i in range(10):
        episode_reward = play(env, agent)
        episode_rewards.append(episode_reward)
        print("测试第{:d}次奖励是{:.4f}".format(i, episode_reward))
    
    print("平均奖励：{:.4f}".format(np.mean(episode_rewards)))
    
    plt.show()

    env.close()
