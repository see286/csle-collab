import numpy as np
import pickle
import json
import os

class SimplePOMDP:
    """简化的POMDP类"""
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

def main():
    # 加载估计的模型
    model_file = "/home/li/csle/estimated_models.pkl"
    
    if not os.path.exists(model_file):
        print(f"❌ 模型文件不存在: {model_file}")
        print("请先运行: python3 ~/csle/system_identification_demo.py")
        return
    
    try:
        with open(model_file, 'rb') as f:
            models = pickle.load(f)
        
        print("✅ 成功加载估计的模型")
        
        # 创建POMDP实例
        pomdp = SimplePOMDP(
            models['transition_model'],
            models['observation_model'],
            models['reward_model']
        )
        
        print(f"✅ POMDP创建成功")
        print(f"   - 状态数: {pomdp.n_states}")
        print(f"   - 动作数: {pomdp.n_actions}")
        print(f"   - 观测数: {pomdp.n_observations}")
        
        # 测试信念更新
        print("\n🧪 测试信念更新:")
        print(f"初始信念: {pomdp.belief}")
        
        # 模拟一个步骤
        test_action = 0
        test_observation = 0
        new_belief = pomdp.update_belief(test_action, test_observation)
        print(f"执行动作 {test_action}，观测到 {test_observation}")
        print(f"更新后信念: {new_belief}")
        
        # 获取动作概率
        action_probs = pomdp.get_action_probabilities()
        print(f"\n动作概率分布: {action_probs}")
        
        # 保存POMDP对象
        pomdp_file = "/home/li/csle/pomdp_object.pkl"
        with open(pomdp_file, 'wb') as f:
            pickle.dump(pomdp, f)
        
        print(f"\n✅ POMDP对象已保存到: {pomdp_file}")
        
        # 保存给C使用的文件
        output_for_C = {
            'policy': {
                'type': 'heuristic_policy',
                'description': '基于信念的启发式策略'
            },
            'action_probabilities': action_probs.tolist(),
            'belief_state': new_belief.tolist(),
            'n_states': pomdp.n_states,
            'n_actions': pomdp.n_actions,
            'model_info': '基于CSLE轨迹数据的POMDP模型'
        }
        
        output_file = '/home/li/csle/output_for_C.json'
        with open(output_file, 'w') as f:
            json.dump(output_for_C, f, indent=2)
        
        print(f"✅ 已生成给角色C的输出文件: {output_file}")
        
        # 显示给C的产出摘要
        print("\n📦 Day 4 完成产出:")
        print(f"   1. POMDP模型文件: {pomdp_file}")
        print(f"   2. JSON配置文件: {output_file}")
        print(f"   3. 动作概率: {action_probs}")
        
        print("\n🎯 Day 4 任务完成！准备进入Day 5: PPO训练")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
