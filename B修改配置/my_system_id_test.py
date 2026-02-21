# 文件名：~/csle/my_system_id_test.py
import pandas as pd
import numpy as np
import sys
import os

# 添加csle库路径
sys.path.append('/home/li/csle/simulation-system/venv/lib/python3.10/site-packages')
sys.path.append('/home/li/csle/simulation-system/libs/csle-system-identification/src')

# 尝试导入系统辨识模块
try:
    from csle_system_identification.emulator import Emulator
    from csle_system_identification.empirical.empirical_algorithm import EmpiricalAlgorithm
    print("✅ 成功导入 csle_system_identification 模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确认以下路径存在：")
    print("1. /home/li/csle/simulation-system/venv/lib/python3.10/site-packages/csle_system_identification")
    print("2. /home/li/csle/simulation-system/libs/csle-system-identification")
    sys.exit(1)

# 读取仿真轨迹数据
trace_file = "/home/li/csle/traces_output_csv/emulation_trace_1.csv"
if os.path.exists(trace_file):
    print(f"✅ 找到轨迹文件: {trace_file}")
    
    # 读取CSV文件
    df = pd.read_csv(trace_file)
    print(f"✅ 成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
    print("数据列名:", df.columns.tolist())
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(df.head())
    
    # 检查必要列是否存在
    required_cols = ['step', 'attacker_action', 'defender_action', 'reward']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  缺少列: {missing_cols}")
    else:
        print("✅ 所有必要列都存在")
        
else:
    print(f"❌ 轨迹文件不存在: {trace_file}")
