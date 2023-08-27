# Atari 环境介绍

## Description

​	每一回合(episode)开始时，Taxi会随机在一个方格启动，乘客也会随机待在一个方格内，Taxi要开往乘客的位置，然后pick up乘客，再带着乘客去目的地，最后drop off乘客。当乘客被drop off，这个episode就结束了。

## Actions

action space是离散的数字，可通过`full_action_space=True or False`来设定是否要完整的动作空间。

## Observations

​	observation通常是RGM图像或者灰度图像。

## Stochasticity

​	为了避免模型只是简单记住操作，ALE并不总是模拟传递给环境的操作，还有较小概率执行以前的操作。

​	Gym还有随机跳帧（stochastic frame skipping）。通常，agent与Atari环境的交互时，Atari环境会返回游戏的每一帧图像作为observation，agent需要为这个observation选择一个action，再让Atari环境去执行这个action。为了减少运算量，使用跳帧技术，即agent选择一个action后，会执行n帧，然后再选择下一个action，也就是每过n帧才会选择一次action。

## Pong游戏

|      Import       | gym.make("ALE/Pong-v5") |
| :---------------: | :---------------------: |
|   Action Space    |      Discrete(18)       |
| Observation Space |      (210, 160, 3)      |
| Observation High  |           255           |
|  Observation Low  |            0            |

### Description

​	Pong是Atari中一个仿真打球的游戏：玩家和电脑每人拿一个板子，接对方弹来的球，如果没接住，对方得一分，先得到21分的获胜。

### Actions

​	如果设定`full_action_space=False`，则简化版的action space如下：

| Num  |  Action   |
| :--: | :-------: |
|  0   |   NOOP    |
|  1   |   FIRE    |
|  2   |   RIGHT   |
|  3   |   LEFT    |
|  4   | RIGHTFIRE |
|  5   | LEFTFIRE  |

### Observations

​	默认返回RGB图像。

### Common Arguments

​	`gym.make`可加如下参数

- **mode**: `int`. Game mode.
- **difficulty**: `int`. Difficulty of the game, Together with `mode`, this determines the “flavor” of the game.
- **obs_type**: `str`. This argument determines what observations are returned by the environment. Its values are:
  - ram: The 128 Bytes of RAM are returned
  - rgb: An RGB rendering of the game is returned
  - grayscale: A grayscale rendering is returned
- **frameskip**: `int` or a `tuple` of two `int`s. This argument controls stochastic frame skipping, as described in the section on stochasticity.
- **repeat_action_probability**: `float`.
- **full_action_space**: `bool`. If set to `True`, the action space consists of all legal actions on the console. Otherwise, the action space will be reduced to a subset.
- **render_mode**: `str`. Specifies the rendering mode. Its values are:
  - human: We’ll interactively display the screen and enable game sounds. This will lock emulation to the ROMs specified FPS
  - rgb_array: we’ll return the `rgb` key in step metadata with the current environment RGB frame.



# DQN(Deep Q Network)

## DQN算法思想

​	DQN不再记录复杂的q值表格，而是用一个参数化的函数 $q_w$ 来拟合其。由于神经网络具有强大的表达能力，因此我们可以用一个神经网络来表示函数q。其目标是使 $q_w$ 与TD目标 $r+\gamma \max_{a'\in \mathcal{A}}q_w(s',a')$ 接近，即最小化函数：
$$
J=\frac{1}{2N}\sum_{i=1}^N[ r_i+\gamma \max_{a'}q_w(s_i',a')-q_w(s_i,a_i) ]^2
$$

### 经验回放（experience replay）

​	DQN为off-policy，为了更好地将 Q-learning 和深度神经网络结合，DQN 算法采用了**经验回放**（experience replay）方法，具体做法是维护一个**回放缓冲区**，将每次从环境中采样得到的四元组数据（状态，动作，奖励，下一状态）存储到回放缓冲区中，训练 Q 网络的时候再从回放缓冲区中随机采样若干数据来进行训练。其优点为：

- 使样本满足独立假设。在 MDP 中交互采样得到的数据本身不满足独立假设，因为这一时刻的状态和上一时刻的状态有关。非独立同分布的数据对训练神经网络有很大的影响，会使神经网络拟合到最近训练的数据上。采用经验回放可以打破样本之间的相关性，让其满足独立假设。
- 提高样本效率。每一个样本可以被使用多次，十分适合深度神经网络的梯度学习。

###　目标网络（target network）

​	由于网络参数 $w$ 既在函数 $q_w$ 中，又在TD目标中，这样求梯度可能会很麻烦，而且在更新网络参数 $w$ 的同时目标也在不断地改变，这非常容易造成神经网络训练的不稳定性。因此采用**目标网络方法**，即使用两个神经网络：

- main network：用来计算原来的 $q_w$ 函数，并且使用正常的梯度下降算法来进行更新。
- target network：用于计算原来的TD目标 $r+\gamma \max_{a'\in \mathcal{A}}q_{w_T}(s',a')$ ，其中，$w_T$ 表示目标网络中的参数。为了让更新目标更稳定，目标网络并不会每一步都更新。训练网络 $q_w(s,a)$ 在训练的每一步都会正常更新，目标网络的参数每隔C步才会和训练网络进行一次同步，即 $w_T\leftarrow w$。

## 算法实现

	###  经验回放池

​	用`collections`库里面的`deque`作为经验池，其先进先出，数据满时，新数据进来，最先进来的数据出去。

```python
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
```

### 神经网络

​	定义Q网络，其有两个隐藏层，使用relu激活函数。

```python
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
```

### DQN模型

​	定义DQN类，其有`decide`和`update`成员函数，分别用来决策和更新。

```python
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
```

### play函数

​	使用agent进行一轮游戏。

```python
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
```

### 图像预处理

​	设置observation是灰度图像，并对其进行裁剪和采样，略去无关部分，缩小数据量，便于加快训练速度，最后返回值是向量。

```python
def preprocess(image):
    '''图像预处理'''
    image = image[35:195] # 裁剪
    image = image[::2, ::2]  # 下采样，缩放2倍
    image = image.reshape(-1, 1)
    return image
```

### 主函数

​	设置超参数，建立环境和模型，进行训练，训练完成后将模型进行保存，以便后续使用。

```python
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
```

## 模型评估

​	设置episode=3000，训练过程的return如下图所示：

![return of Pong by DQN](https://github.com/Huex123/Reinforcement-Learning-project1/blob/main/return%20of%20Pong%20by%20DQN.svg)

观察到模型并未收敛，可能代码编写还有问题，或者训练次数不够。
