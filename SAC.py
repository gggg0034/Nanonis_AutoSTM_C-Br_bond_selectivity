import multiprocessing
import os
import random
import time
from collections import deque

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal

from env import Env  # 导入自定义环境
from utils import weights_init_

torch.autograd.set_detect_anomaly(True)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return torch.cat([mean, log_std], dim=-1)
    
class CriticNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc2(x))
        x = self.q_value(x)
        return x




class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# Replay Buffer 
class ReplayBuffer:
    def __init__(self, max_size, env):
        self.buffer = deque(maxlen=max_size)
        self.env = env
        self.now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    
    def add(self, state, action, reward, next_state, done, info):
        self.buffer.append((state, action, reward, next_state, done, info))
    
    def sample(self, batch_size):
        # get the batch data from new data
        success_ratio = 0.3
        latest_ratio = 0.2
        random_ratio = 0.5
        num_success = round(success_ratio * batch_size)
        num_latest = round(latest_ratio * batch_size)
        num_random = round(random_ratio * batch_size)

        # initial the sampled data
        sampled_success_experiences = []
        success_indices = [idx for idx, experience in enumerate(self.buffer) 
                   if experience[-1].get("reaction") == "success"]
        actual_success = min(num_success, len(success_indices))
        if actual_success > 0:
            sampled_success = random.sample(success_indices, actual_success)
            sampled_success_experiences = [self.buffer[idx] for idx in sampled_success]
        else:
            sampled_success_experiences = []
            actual_success = 0
        sampled_latest_experiences = []
        if num_latest > 0:
            sampled_latest_experiences = list(self.buffer)[-num_latest:]
        remaining = batch_size - actual_success - len(sampled_latest_experiences)
        if remaining > 0:
            exclude_indices = set()
            try:
                exclude_indices.update(sampled_success)
            except:
                pass
            exclude_indices.update(range(len(self.buffer) - num_latest, len(self.buffer)))
            
            available_indices = list(set(range(len(self.buffer))) - exclude_indices)
            
            if len(available_indices) < remaining:
                sampled_random_experiences = random.choices(self.buffer, k=remaining)
            else:
                sampled_random = random.sample(available_indices, remaining)
                sampled_random_experiences = [self.buffer[idx] for idx in sampled_random]
        else:
            sampled_random_experiences = []
        
        batch = sampled_success_experiences + sampled_latest_experiences + sampled_random_experiences

        if len(batch) < batch_size:
            additional = batch_size - len(batch)
            additional_samples = random.choices(self.buffer, k=additional)
            batch += additional_samples


        batch = random.sample(self.buffer, batch_size)
        random.shuffle(batch)
        states, actions, rewards, next_states, dones, info = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def size(self):
        return len(self.buffer)

    def plot_train(self, hyperparameters):
        # plot the action and plot the reward
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(2, 2, height_ratios=[5, 3], width_ratios=[5, 5])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        action_bound = [self.env.action_space.low, self.env.action_space.high]
        ax1.set_title("x, y Distribution")
        ax1.set_xlim(-50,50)
        ax1.set_ylim(-50,50)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("v, a Distribution")
        ax2.set_xlim(action_bound[0][2],action_bound[1][2])
        ax2.set_ylim(action_bound[0][3],action_bound[1][3])
        ax2.set_xlabel("v")
        ax2.set_ylabel("a")
        data_x, data_y, colors_xy = [], [], []
        data_v, data_a, colors_va = [], [], []
        reward_list = []
        for state, action, reward, next_state, done, info in self.buffer:
            if self.env.polar_space:
                l, theta, v, a = action
                x = l * np.cos(np.radians(theta))
                y = l * np.sin(np.radians(theta))
            else:
                x, y, v, a = action
            if info["reaction"] == "success":
                color = (0, 1, 0, 0.5)  # 半透明绿色
            elif info["reaction"] == "failure":
                color = (1, 0, 0, 0.5)  # 半透明红色
            else:  # 无变化
                color = (0.5, 0.5, 0.5, 0.5)  # 半透明灰色
            data_x.append(x)
            data_y.append(y)
            colors_xy.append(color)
            data_v.append(v)
            data_a.append(a)
            colors_va.append(color)
            reward_list.append(reward)
        ax1.scatter(data_x, data_y, c=colors_xy, alpha=0.5)
        ax2.scatter(data_v, data_a, c=colors_va, alpha=0.5)
        ax3.plot(reward_list, color='blue')
        plt.tight_layout()

        if not os.path.exists("./train_result_plot"):
            os.makedirs("./train_result_plot")
        # add the hyperparameters to the figure name
        str_hyperparameters = '_'.join([f"{key}={value}" for key, value in hyperparameters.items()])
        plt.savefig(f"train_result_plot/action_{self.now}_{str_hyperparameters}.png")
    
    # save the buffer data as a txt file
    def save(self):
        # now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        with open(f"train_result_plot/buffer_data_{self.now}.txt", "w") as f:
            for state, action, reward, next_state, done, info in self.buffer:
                f.write(f"state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {done}, info: {info}\n")
        print("Save the buffer data successfully!")
        


# SAC Agent
class SACAgent:
    def __init__(self, env: Env
                 , buffer_size=10000, 
                 batch_size=64, 
                 gamma=0.01, 
                 tau=0.005, 
                 alpha=0.2, 
                 lr=0.0003):
        
        self.hyperparameters = {
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "alpha": alpha,
            "lr": lr
        }
        self.env = env
        self.state_dim = env.observation_space.shape[0]                         # input dimension
        self.action_dim = env.action_space.shape[0]                             # output dimension
        self.action_bound = [env.action_space.low, env.action_space.high]    

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        self.target_entropy = -torch.tensor([self.action_dim], dtype=torch.float32, device=self.device)

        self.replay_buffer = ReplayBuffer(buffer_size,env)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.epsilon = 1e-6

        low, high = self.action_bound

        self.action_scale = torch.tensor((high - low) / 2, dtype=torch.float32, device=self.device)
        self.action_bias = torch.tensor((high + low) / 2, dtype=torch.float32, device=self.device)
        # self.action_scale = (high - low) / 2
        # self.action_bias = (high + low) / 2

        # Networks
        # self.policy_net = MLP(self.state_dim, self.action_dim * 2).to(self.device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(self.device)
        self.q_net1 = CriticNet(self.state_dim + self.action_dim, 1).to(self.device)
        self.q_net2 = CriticNet(self.state_dim + self.action_dim, 1).to(self.device)
        self.target_q_net1 = CriticNet(self.state_dim + self.action_dim, 1).to(self.device)
        self.target_q_net2 = CriticNet(self.state_dim + self.action_dim, 1).to(self.device)
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=self.device, dtype=torch.float32)
        self.alpha = self.log_alpha.detach().exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=lr, weight_decay=1e-4)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=lr, weight_decay=1e-4)

        # Copy weights to target networks
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

    def one_hot_encode(self, state):
        """将分子状态（0, 1, 2）转换为独热编码，并保留反应时间"""
        molecule_state = int(state) + 1  # 分子状态
        # reaction_time = state[1]  # 反应时间
        one_hot = np.zeros(1)
        one_hot[0] = molecule_state  # 对分子状态进行独热编码
        return np.array(one_hot)

    def remap_action(self, action):
        """将 [-1, 1] 范围的动作重新映射到实际动作空间"""
        return self.action_bias + self.action_scale * action

    def select_action(self, states, deterministic=False, detach_action=True):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float().to(self.device)
        elif isinstance(states, torch.Tensor):
            states = states.to(self.device)
        mean, log_std = torch.chunk(self.policy_net(states), 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            pre_tanh_action = mean
        else:
            pre_tanh_action = dist.rsample()
        action = torch.tanh(pre_tanh_action)  # 将动作限制在 [-1, 1]

        log_prob = dist.log_prob(pre_tanh_action) - torch.log(self.action_scale * (1 - action.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        remapped_action = self.action_bias + self.action_scale * action

        if detach_action:
            return remapped_action.detach().cpu().numpy(), log_prob.detach()
        return remapped_action, log_prob

    def train(self, num_steps, render=False, visualize=False):
        state = self.env.reset()
        state = self.one_hot_encode(state)  # 转换状态为独热编码并添加反应时间
        
        # plot the action
        if visualize:
            queue = multiprocessing.Queue(50)
            plot_process = multiprocessing.Process(target=self.plot_actions, args=(queue,self.env))
            plot_process.start()
        
        for step in range(num_steps):
            action,_ = self.select_action(state, detach_action=True)
            next_state, reward, done, info = self.env.step(action)
            print(f"Reward: {reward}")
            next_state = self.one_hot_encode(next_state)  # 转换为独热编码并添加反应时间
            self.replay_buffer.add(state, action, reward, next_state, done, info)

            if render:
                self.env.render()
                self.env.clock.tick(60)

            if visualize:
                
                if queue.full():  # 如果队列满了，则等待一段时间
                    queue.get()
                
                queue.put((state.copy(), action.copy(), reward, next_state.copy(), done, info.copy()))

            if self.replay_buffer.size() > self.batch_size:
                self.update()

            if step % 1000 == 0:
                self.save_model(step)

            if done:
                state = self.one_hot_encode(self.env.reset())  # 重置并编码
            else:
                state = next_state
            
        if visualize:
            queue.put(None)
            plot_process.join()
        
        self.replay_buffer.plot_train(self.hyperparameters)
        self.replay_buffer.save()



    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        #reverse the action to the range of [-1,1]
        actions = (actions - self.action_bias) / self.action_scale
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q-Loss
        with torch.no_grad():
            next_actions, next_state_log_pi = self.select_action(next_states, detach_action=True)
            next_actions = torch.tensor(next_actions, dtype=torch.float32, device=self.device)
            next_actions = (next_actions - self.action_bias) / self.action_scale

            q_target1 = self.target_q_net1(torch.cat([next_states, next_actions], dim=-1))
            q_target2 = self.target_q_net2(torch.cat([next_states, next_actions], dim=-1))
            q_min = torch.min(q_target1, q_target2) - self.alpha * next_state_log_pi 
            q_target = rewards + (1 - dones) * self.gamma * q_min

        q1 = self.q_net1(torch.cat([states, actions], dim=-1))
        q2 = self.q_net2(torch.cat([states, actions], dim=-1))
        q_loss1 = ((q1 - q_target) ** 2).mean()
        q_loss2 = ((q2 - q_target) ** 2).mean()
        q_loss = q_loss1 + q_loss2

        # Policy-loss
        new_actions, log_prob = self.select_action(states, detach_action=False)  # 保持与计算图的连接
        new_actions = (new_actions - self.action_bias) / self.action_scale
        # new_actions 已经是张量，与计算图相连
        q1_new = self.q_net1(torch.cat([states, new_actions], dim=-1))
        q2_new = self.q_net2(torch.cat([states, new_actions], dim=-1))
        min_q_pi = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - min_q_pi).mean()
        # policy_loss = (self.alpha* log_prob - states  ).mean()

        # Alpha-loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        # Update critic
        self.q_optimizer1.zero_grad()
        self.q_optimizer2.zero_grad()        
        q_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), 2, norm_type=2.0)    # gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), 2, norm_type=2.0)

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 2, norm_type=2.0)      # gradient clipping
        # Update alpha    
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.log_alpha.parameters(), 2, norm_type=2.0)      # gradient clipping

        # Update all networks
        self.q_optimizer1.step()
        self.q_optimizer2.step()
        self.policy_optimizer.step()   
        self.alpha_optim.step()        

        self.alpha = self.log_alpha.detach().exp()

        # Soft update target networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, step):
        # get the current date and time
        now = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        if not os.path.exists(f"models/{now}"):
            os.makedirs(f"models/{now}")
        torch.save(self.policy_net.state_dict(), f"models/{now}/policy_net_{step}.pth")
        torch.save(self.q_net1.state_dict(), f"models/{now}/q_net1_{step}.pth")
        torch.save(self.q_net2.state_dict(), f"models/{now}/q_net2_{step}.pth")

    @classmethod
    def plot_actions(self,queue,env):
        """绘图进程：消费队列中的数据并更新图表"""
        # 初始化绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title("x, y Distribution")
        action_bound = [env.action_space.low, env.action_space.high]
        # set the range of x and y, according to the action_bound
        ax1.set_xlim(action_bound[0][0],action_bound[1][0])
        ax1.set_ylim(action_bound[0][1],action_bound[1][1])
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax2.set_title("v, a Distribution")
        # set the range of v and a, according to the action_bound
        ax2.set_xlim(action_bound[0][2],action_bound[1][2])
        ax2.set_ylim(action_bound[0][3],action_bound[1][3])
        ax2.set_xlabel("v")
        ax2.set_ylabel("a")

        data_x, data_y, colors_xy = [], [], []
        data_v, data_a, colors_va = [], [], []
        # pop data_x data_y colors_xy data_v data_a colors_va if the length is more than 50

        while True:
            item = queue.get()
            if item is None:  # 终止信号
                break

            state, action, reward, next_state, done, info = item
            if env.polar_space:
                l, theta, v, a = action # theta in 0-360
                x = l * np.cos(np.radians(theta))
                y = l * np.sin(np.radians(theta))
            else:
                x, y, v, a = action

            # 根据反应结果设置颜色
            if info["reaction"] == "success":
                color = (0, 1, 0, 0.5)  # 半透明绿色
            elif info["reaction"] == "failure":
                color = (1, 0, 0, 0.5)  # 半透明红色
            else:  # 无变化
                color = (0.5, 0.5, 0.5, 0.5)  # 半透明灰色

            # 更新数据
            data_x.append(x)
            data_y.append(y)
            colors_xy.append(color)
            data_v.append(v)
            data_a.append(a)
            colors_va.append(color)
            # pop data_x data_y colors_xy data_v data_a colors_va if the length is more than 50
            if len(data_x) > 50:
                data_x.pop(0)
                data_y.pop(0)
                colors_xy.pop(0)
                data_v.pop(0)
                data_a.pop(0)
                colors_va.pop(0)
            # 更新图表
            # clear the ax1 and ax2
            ax1.clear()
            ax2.clear()
            ax1.set_xlim(-50,50)
            ax1.set_ylim(-50,50)
            ax2.set_xlim(action_bound[0][2],action_bound[1][2])
            ax2.set_ylim(action_bound[0][3],action_bound[1][3])
            ax1.scatter(data_x, data_y, c=colors_xy, alpha=0.5)
            ax2.scatter(data_v, data_a, c=colors_va, alpha=0.5)
            plt.pause(0.01)  # 实时刷新


        # plt.ioff()
        # plt.show()

# Train the agent
if __name__ == "__main__":
    # polar_space_list = [0]
    # batch_size_list = [32,64,128]
    # gamma_list = [0.01,0.5,0.99]
    # tau_list = [0.005,0.01,0.1]
    # alpha_list = [0.1,0.2,0.99]
    # # lr_list = [0.0001,0.0003,0.001]

    # for polar_space in polar_space_list:
    #     for batch_size in batch_size_list:
    #         for gamma in gamma_list:
    #             for tau in tau_list:
    #                 for alpha in alpha_list:
    #                     # for lr in lr_list:
    #                     for i in range(5):
    #                         env = Env(polar_space=polar_space)
    #                         agent = SACAgent(env,buffer_size=10000,batch_size=batch_size,gamma=gamma,tau=tau,alpha=alpha,lr=0.0003)
    #                         agent.train(1000,visualize=False)
    
    env = Env(polar_space=False)
    agent = SACAgent(env)
    agent.train(1000,visualize=True)
