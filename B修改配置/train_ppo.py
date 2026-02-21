import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pickle
import json
import os
import matplotlib.pyplot as plt

# ==================== SimplePOMDP 类定义 ====================
class SimplePOMDP:
    """简化的POMDP类（与build_pomdp.py中相同）"""
    def __init__(self, transition_model, observation_model, reward_model):
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        
        self.n_states = transition_model.shape[0]
        self.n_actions = transition_model.shape[1]
        self.n_observations = observation_model.shape[1]
        
        # 初始信念状态（均匀分布）
        self.belief = np.ones(self.n_states) / self.n_states
    
    def update_belief(self, action, observation):
        """更新信念状态（贝叶斯更新）"""
        new_belief = np.zeros(self.n_states)
        
        for s_next in range(self.n_states):
            # P(s' | o, a, b) ∝ P(o | s') * Σ_s P(s' | s, a) * b(s)
            prob_o_given_s = self.observation_model[s_next, observation]
            sum_term = 0
            
            for s in range(self.n_states):
                sum_term += self.transition_model[s, action, s_next] * self.belief[s]
            
            new_belief[s_next] = prob_o_given_s * sum_term
        
        # 归一化
        if new_belief.sum() > 0:
            new_belief /= new_belief.sum()
        else:
            new_belief = np.ones(self.n_states) / self.n_states
        
        self.belief = new_belief
        return self.belief
    
    def get_action_probabilities(self, belief=None):
        """获取每个动作的概率（简单启发式）"""
        if belief is None:
            belief = self.belief
        
        # 计算每个动作的期望奖励
        expected_rewards = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            for s in range(self.n_states):
                expected_rewards[a] += belief[s] * self.reward_model[s, a]
        
        # 使用softmax转换为概率
        exp_rewards = np.exp(expected_rewards - np.max(expected_rewards))
        action_probs = exp_rewards / exp_rewards.sum()
        
        return action_probs
    
    def reset_belief(self):
        """重置信念状态"""
        self.belief = np.ones(self.n_states) / self.n_states
# ==================== SimplePOMDP 类定义结束 ====================

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

class PPOActorCritic(nn.Module):
    """PPO Actor-Critic网络"""
    def __init__(self, n_states, n_actions, hidden_dim=64):
        super(PPOActorCritic, self).__init__()
        
        # 共享的特征提取层
        self.shared = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 策略网络 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PPOTrainer:
    """PPO训练器"""
    def __init__(self, pomdp, n_states, n_actions):
        self.pomdp = pomdp
        self.n_states = n_states
        self.n_actions = n_actions
        
        # 初始化PPO网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOActorCritic(n_states, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # PPO超参数
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.update_epochs = 10
        self.batch_size = 64
        
        # 训练记录
        self.rewards_history = []
        self.episode_lengths = []
    
    def select_action(self, belief):
        """根据当前信念选择动作"""
        belief_tensor = torch.FloatTensor(belief).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(belief_tensor.unsqueeze(0))
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, state_value
    
    def compute_returns(self, rewards, values, dones, next_value):
        """计算GAE回报"""
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
            next_value = values[t]
        
        return returns
    
    def update_policy(self, states, actions, log_probs_old, returns, advantages):
        """更新策略网络"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        for _ in range(self.update_epochs):
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            log_probs_new = dist.log_prob(actions)
            
            # 概率比率
            ratios = torch.exp(log_probs_new - log_probs_old)
            
            # 裁剪目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # 策略损失
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
    
    def train(self, num_episodes=100, max_steps=30):
        """训练PPO代理"""
        print(f"🚀 开始PPO训练，共{num_episodes}回合")
        print(f"   设备: {self.device}")
        print(f"   状态数: {self.n_states}, 动作数: {self.n_actions}")
        
        for episode in range(num_episodes):
            # 重置环境
            self.pomdp.reset_belief()
            episode_reward = 0
            episode_steps = 0
            
            # 存储轨迹
            states = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            dones = []
            
            for step in range(max_steps):
                # 获取当前信念
                current_belief = self.pomdp.belief
                
                # 选择动作
                action, log_prob, value = self.select_action(current_belief)
                
                # 模拟环境步骤（使用POMDP的奖励模型）
                state_idx = np.argmax(current_belief) if current_belief.max() > 0.5 else 0
                reward = self.pomdp.reward_model[state_idx, action] + np.random.normal(0, 0.1)
                
                # 随机生成观测（简化）
                observation = np.random.randint(0, self.pomdp.n_observations)
                
                # 更新信念
                self.pomdp.update_belief(action, observation)
                
                # 存储轨迹数据
                states.append(current_belief)
                actions.append(action)
                log_probs.append(log_prob.item())
                values.append(value.item())
                rewards.append(reward)
                dones.append(0 if step < max_steps-1 else 1)
                
                episode_reward += reward
                episode_steps += 1
            
            # 计算最后一个状态的value
            final_belief = self.pomdp.belief
            final_belief_tensor = torch.FloatTensor(final_belief).to(self.device)
            with torch.no_grad():
                _, final_value = self.policy(final_belief_tensor.unsqueeze(0))
            
            # 计算回报和优势
            returns = self.compute_returns(rewards, values, dones, final_value.item())
            advantages = np.array(returns) - np.array(values)
            
            # 更新策略
            self.update_policy(states, actions, log_probs, returns, advantages)
            
            # 记录训练进度
            self.rewards_history.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # 打印进度
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.rewards_history[-20:])
                print(f"回合 {episode+1}/{num_episodes} | "
                      f"平均奖励: {avg_reward:.3f} | "
                      f"回合步数: {episode_steps}")
        
        print("✅ PPO训练完成!")
        return self.policy
    
    def save_model(self, path):
        """保存训练好的模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'rewards_history': self.rewards_history,
            'episode_lengths': self.episode_lengths
        }, path)
        print(f"✅ PPO模型已保存到: {path}")
    
    def plot_training(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 奖励曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('PPO Training Reward History')
        plt.grid(True)
        
        # 滑动平均奖励
        window_size = 20
        if len(self.rewards_history) >= window_size:
            moving_avg = np.convolve(self.rewards_history, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.rewards_history)), moving_avg, 'r-', linewidth=2, label=f'{window_size}-ep moving avg')
            plt.legend()
        
        # 回合长度
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Length History')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/li/csle/ppo_training_curves.png', dpi=100)
        print("✅ 训练曲线图已保存")
        plt.close()  # 不显示图形，只保存文件

def main():
    print("🎯 Day 5: PPO训练开始")
    
    # 加载POMDP模型 - 现在使用新的加载方式
    model_file = "/home/li/csle/estimated_models.pkl"
    pomdp_file = "/home/li/csle/pomdp_object.pkl"
    
    if not os.path.exists(model_file):
        print(f"❌ 模型文件不存在: {model_file}")
        print("请先完成Day 3任务")
        return
    
    if not os.path.exists(pomdp_file):
        print(f"❌ POMDP文件不存在: {pomdp_file}")
        print("尝试重新构建POMDP...")
        # 可以在这里重新构建，但为了简单，我们直接加载原始模型
        pass
    
    try:
        # 方法1：尝试加载pickle文件
        if os.path.exists(pomdp_file):
            try:
                with open(pomdp_file, 'rb') as f:
                    pomdp = pickle.load(f)
                print("✅ 成功加载POMDP对象")
            except:
                print("⚠️  pickle加载失败，重新构建POMDP...")
                # 加载原始模型并重新构建POMDP
                with open(model_file, 'rb') as f:
                    models = pickle.load(f)
                
                pomdp = SimplePOMDP(
                    models['transition_model'],
                    models['observation_model'],
                    models['reward_model']
                )
                print("✅ 重新构建POMDP成功")
        else:
            # 直接加载原始模型并构建POMDP
            with open(model_file, 'rb') as f:
                models = pickle.load(f)
            
            pomdp = SimplePOMDP(
                models['transition_model'],
                models['observation_model'],
                models['reward_model']
            )
            print("✅ 从原始模型构建POMDP成功")
        
        print(f"   - 状态数: {pomdp.n_states}")
        print(f"   - 动作数: {pomdp.n_actions}")
        
        # 创建训练器
        trainer = PPOTrainer(pomdp, pomdp.n_states, pomdp.n_actions)
        
        # 开始训练（简化为80回合，加快速度）
        print("\n开始训练PPO策略...")
        policy = trainer.train(num_episodes=80, max_steps=25)
        
        # 保存训练好的模型
        trainer.save_model('/home/li/csle/ppo_policy.pth')
        
        # 绘制训练曲线
        trainer.plot_training()
        
        # 测试训练后的策略
        print("\n🧪 测试训练后的策略:")
        pomdp.reset_belief()
        test_belief = pomdp.belief
        
        # 使用训练好的策略选择动作
        belief_tensor = torch.FloatTensor(test_belief).to(trainer.device)
        with torch.no_grad():
            action_probs, _ = trainer.policy(belief_tensor.unsqueeze(0))
        
        print(f"初始信念: {test_belief}")
        action_probs_np = action_probs.squeeze().cpu().numpy()
        print(f"动作概率: {action_probs_np}")
        print(f"推荐动作: {np.argmax(action_probs_np)}")
        
        # 生成给C的更新文件
        output_for_C = {
            'trained_policy': {
                'file': '/home/li/csle/ppo_policy.pth',
                'format': 'PyTorch state_dict',
                'description': 'PPO训练后的策略网络'
            },
            'training_results': {
                'final_avg_reward': float(np.mean(trainer.rewards_history[-20:])),
                'total_episodes': len(trainer.rewards_history),
                'avg_episode_length': float(np.mean(trainer.episode_lengths)),
                'training_curves': '/home/li/csle/ppo_training_curves.png'
            },
            'model_architecture': {
                'n_states': pomdp.n_states,
                'n_actions': pomdp.n_actions,
                'hidden_dim': 64
            },
            'test_results': {
                'initial_belief': test_belief.tolist(),
                'action_probabilities': action_probs_np.tolist(),
                'recommended_action': int(np.argmax(action_probs_np))
            },
            'day': 5,
            'task': 'PPO训练完成'
        }
        
        output_file = '/home/li/csle/ppo_training_results.json'
        with open(output_file, 'w') as f:
            json.dump(output_for_C, f, indent=2)
        
        print(f"\n✅ PPO训练完成，产出文件:")
        print(f"   1. 策略模型: /home/li/csle/ppo_policy.pth")
        print(f"   2. 训练曲线: /home/li/csle/ppo_training_curves.png")
        print(f"   3. 训练结果: {output_file}")
        print(f"   4. 最后20回合平均奖励: {np.mean(trainer.rewards_history[-20:]):.3f}")
        
        print("\n🎯 Day 5 任务完成！准备进入Day 6: 策略调优")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
