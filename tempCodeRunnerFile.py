# Reward shaping
            width = 5
            height = 5
            locations = {0:[0,0],1:[0,4],2:[4,0],3:[4,3]} # 每个地点对应的坐标
            x, y, p, d = env.unwrapped.decode(next_state)
            if p != 4: # 乘客不在Taxi上
                # 计算对应的哈密顿距离
                length = np.double(np.abs(x-locations[p][0]) + np.abs(y-locations[p][1]))
                F = 0.1*(1-length/(width+height))
            else: # 乘客在车上,length=0
                F = 0.1
            reward += F