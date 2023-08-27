# Taxi-v3 环境介绍

|      Import       | `gym.make("Taxi-v3")` |
| :---------------: | :-------------------: |
| Observation Space |    Discrete（500）    |
|   Action Space    |     Discrete（6）     |

## Description

​	每一回合(episode)开始时，Taxi会随机在一个方格启动，乘客也会随机待在一个方格内，Taxi要开往乘客的位置，然后pick up乘客，再带着乘客去目的地，最后drop off乘客。当乘客被drop off，这个episode就结束了。

## Actions

6个离散的确定性动作：

- 0：move south
- 1：move north
- 2：move east
- 3：move west
- 4：pick up passenger
- 5：drop off passenger

## Observations

​	一共有500个离散状态，因为有25个Taxi位置、5个乘客位置（包括乘客在Taxi上）和4个目的地位置。因为有可能乘客和目的地在同一位置，那么这个episode直接结束，所以减去100个状态；当一个episode成功结束时，乘客和Taxi都在目的地，这有4个状态，所以实际上一个episode只能达到404个状态。

​	每一个state space都用tuple表示：`(taxi_row, taxi_col, passenger_location, destination)`。状态值Observation是对应state的编码，一个整数，对应的tuple表示可通过解码"decode"获得，eg：`x, y, p, d = env.unwrapped.decode(observation)`。

Passenger locations：

- 0：R(ed)
- 1：G(reen)
- 2：Y(ellow)
- 3：B(lue)
- 4：in taxi

Destinations：

- 0：R(ed)
- 1：G(reen)
- 2：Y(ellow)
- 3：B(lue)

## Rewards

+ Taxi每走一步得到 -1 除非它触发了其他奖励
+ Taxi完成任务 +20
+ Taxi非法执行"pick up"和"drop off" -10

# Sarsa

## Sarsa算法思想

​	Sarsa是一个基于时序差分方法的model-free的on-policy、online强化学习算法。由于是on-policy，所以无法进行Experience Replay。其先用时序差分算法来估计action values：
$$
\begin{aligned}
q_{t+1}(s_t,a_t) &=q_t(s_t,a_t)+\alpha_t(s_t,a_t)[r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})-q_t(s_t,a_t)],\\
q_{t+1}(s,a) &=q_t(s,a),\quad for\,\ all \,\ (s,a)\neq (s_t,a_t)
\end{aligned}
$$
​	然后用贪婪算法来选取在某个状态下action value最大的那个动作，以此来更新策略：
$$
\pi(a|s)=\begin{cases}
\epsilon/\vert\mathcal{A}\vert+1-\epsilon,\quad if\,\ a=\arg\max_{a'}{q(s,a')} \\
\epsilon/\vert\mathcal{A}\vert, \quad otherwise \\
\end{cases}
$$

## 算法实现

### Sarsa类

​	首先定义Saras模型，其有两个方法：`decide(self, state)`和`update(self, state, action, reward, next_s, done, next_a)`，分别用来得到策略、策略更新。

​	之所以使用$\epsilon-greedy$，是因为如果在策略提升中一直根据贪婪算法得到一个确定性策略(s, a)，可能会导致某些状态动作对永远没有在序列中出现，以至于无法对其动作价值进行估计，进而无法保证策略提升后的策略比之前的好。在测试时，为了使算法每一步都是最佳策略，可将$\epsilon$置为0。

```python
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
```

### play函数

​	定义`play(env, agent, train=False)`函数，其作用是用`agent`在环境`env`中进行一轮的游戏，返回整个episode的return。如果是训练模式，则过程中进行策略更新，否则仅仅只是进行游戏。

​	训练模式中，先得到观测observation，再用其生成action，再通过`env.step(action)`函数与环境进行一次交互，得到下一步的observation和相应的reward，再用这个observation生成next_action，然后利用其带入Sarsa算法的迭代公式进行更新，其更新是对于整个episode$(s_0, a_0, r_1, s_1, a_1, \dots, s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1},\dots )$中的每个$( s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$，其中，$r_{t+1},s_{t+1}$是环境env产生的，$a_{t+1}$是策略$\pi_t(s_{t+1})$产生的。

```python
def play(env, agent, train=False):
    '''play一回合(episode)'''
    episode_reward = 0
    observation, info= env.reset()
    action = agent.decide(observation)
    while True:
        next_observation, reward, terminated, truncated, info = env.step(action) # 得到下一步状态
        done = terminated or truncated
        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            # policy update
            agent.update(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward
```

### 主函数

​	主函数中进行模型的训练和测试。记录每次训练的return加入到episode_rewards列表中。当进行测试时，将策略改成确定性策略，即将$\epsilon$置为0。

```python
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
    ax.set_title('Taxi by Saras')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    # fig.savefig('return of Taxi by Saras.svg')
    
    # 测试
    agent.epsilon = 0 # 确定性策略
    episode_rewards = []
    # env = gym.make("Taxi-v3", render_mode="human") # 若测试时想要可视化游戏界面
    for i in range(100):
        episode_reward = play(env, agent)
        episode_rewards.append(episode_reward)
        print("测试第{:d}次奖励是{:.4f}".format(i, episode_reward))
    
    print("平均奖励：{:.4f}".format(np.mean(episode_rewards)))
    print("q={}".format(agent.q))
    
    plt.show()

    env.close()
```

## 模型评估

### 结果分析

​	设置模型参数$\gamma=0.9, \alpha=0.1, \epsilon=0.05$，迭代4000次（episode=4000），得到训练过程中的return如下图所示：

![return of Taxi by Saras](https://github.com/Huex123/Reinforcement-Learning-project1/blob/main/return%20of%20Taxi%20by%20Saras.svg)

​	由图可以看出，agent已经收敛。从`Taxi_Sarsa.mp4`视频中也可看出，每个episode中，Taxi都可以稳定将乘客送到目的地。

### Reward Shaping

​	Reward Shaping是在强化学习训练的过程中适当地多给予模型一些即时的、小的 reward 来帮助模型能够更快、更好地拟合。可在Sarsa原迭代公式基础上再增加一个reward F，则变成以下公式：
$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha_t(s_t,a_t)[r_{t+1}+F+\gamma q_t(s_{t+1},a_{t+1})-q_t(s_t,a_t)]
$$
​	本题中F可通过Taxi和乘客之间的哈密顿距离$\abs{x_t-x_p}+\abs{y_t-y_p}$来衡量，其中$x_t、y_t$是Taxi的横纵左边，$x_p、y_p$是乘客的横纵坐标，则得到：
$$
F=0.1\times(1-\frac{ \abs{x_t-x_p}+\abs{y_t-y_p} }{width+height})
$$

​	修改`play`函数，得到使用了reward shaping的Sarsa算法：

```python
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
            F = 0
        reward += F

        episode_reward += reward
        next_action = agent.decide(next_observation)
        if train:
            agent.update(observation, action, reward, next_observation, done, next_action)
        if done:
            break
        observation, action = next_observation, next_action
    return episode_reward
```

​	训练得到return如下图所示：

![return of Taxi by Saras with Reward shaping](https://github.com/Huex123/Reinforcement-Learning-project1/blob/main/return%20of%20Taxi%20by%20Saras%20with%20Reward%20shaping.svg)

对比无Reward shaping的图像可以看到，训练前期，其收敛速度加快。



# Q-Learning

## Q-Learning算法思想

​	Q-Learning是一个基于时序差分方法的model-free的强化学习算法。其与Sarsa算法类似，其先用时序差分算法来估计action values：
$$
\begin{aligned}
q_{t+1}(s_t,a_t) &=q_t(s_t,a_t)+\alpha_t(s_t,a_t)[r_{t+1}+\gamma \max_{a\in \mathcal{A}(s_{t+1})} q_t(s_{t+1},a)-q_t(s_t,a_t)],\\
q_{t+1}(s,a) &=q_t(s,a),\quad for\,\ all \,\ (s,a)\neq (s_t,a_t)
\end{aligned}
$$
它的更新公式使用的是$(s_t,a_t,r_{t+1},s_{t+1})$，故是off-policy算法。但本题不使用其off-policy特性。

​	然后用贪婪算法来选取在某个状态下action value最大的那个动作，以此来更新策略：
$$
\pi(a|s)=\begin{cases}
\epsilon/\vert\mathcal{A}\vert+1-\epsilon,\quad if\,\ a=\arg\max_{a'}{q(s,a')} \\
\epsilon/\vert\mathcal{A}\vert, \quad otherwise \\
\end{cases}
$$

## 算法实现

其只需在Sarsa算法的基础上修改更新公式即可，其余稍作修改。

### QLearning类

```python
class QLearning:
    '''Q-Learning算法'''
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
    def update(self, state, action, reward, next_s, done):
        td_error = reward + self.gamma*self.q[next_s].max()*(1-done) - self.q[state, action]
        self.q[state, action] += self.alpha * td_error
```

### play函数

​	训练模式中，先得到观测observation，再用其生成action，再通过`env.step(action)`函数与环境进行一次交互，得到下一步的observation以及相应的reward，然后利用其带入Q-Learning算法的迭代公式进行更新，其更新是对于整个episode$(s_0, a_0, r_1, s_1, a_1, \dots, s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1},\dots )$中的每个$( s_t, a_t, r_{t+1}, s_{t+1})$，其中，$a_{t}$是策略$\pi_t(s_{t})$产生的，$r_{t+1},s_{t+1}$是环境env产生的。

```python
def play(env, agent, train=False):
    '''play一回合(episode)'''
    episode_reward = 0
    observation, info= env.reset()
    while True:
        action = agent.decide(observation)
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
        if train:
            agent.update(observation, action, reward, next_observation, done)
        if done:
            break
        observation = next_observation
    return episode_reward
```

## 模型评估

### 结果分析

​	设置模型参数$\gamma=0.9, \alpha=0.1, \epsilon=0.05$，迭代4000次（episode=4000），在不使用Reward Shaping时，得到训练过程中的return如下图所示：

![return of Taxi by Q-Learning](https://github.com/Huex123/Reinforcement-Learning-project1/blob/main/return%20of%20Taxi%20by%20Q-Learning.svg)

​	由图可以看出，agent已经收敛。从`Taxi_QLearning.mp4`视频中也可看出，每个episode中，Taxi都可以稳定将乘客送到目的地。

### Reward Shaping

​	使用Reward Shaping后，训练过程得到的return如下图所示：

![return of Taxi by Q-Learning with Reward shaping](https://github.com/Huex123/Reinforcement-Learning-project1/blob/main/return%20of%20Taxi%20by%20Q-Learning%20with%20Reward%20shaping.svg)

对比前面无Reward shaping的图像可以看到，训练前期，其收敛速度稍有加快。
