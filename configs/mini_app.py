from flask import Flask, jsonify, request,send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
# 增加这个：让后端直接把 build 文件夹里的网页发给浏览器
# --- 增加这些路由来堵住 404 漏洞 ---

@app.route('/version', methods=['GET'])
def get_version():
    return jsonify({"version": "1.1.0", "name": "CSLE-Bridge-Li"})

@app.route('/config/registration-allowed', methods=['GET'])
def reg_allowed():
    return jsonify(True)

@app.route('/traces-datasets', methods=['GET'])
def get_traces_datasets():
    return jsonify([]) # 返回空列表总比 404 强

@app.route('/statistics-datasets', methods=['GET'])
def get_stats_datasets():
    return jsonify([])

@app.route('/emulation-executions', methods=['GET'])
def get_executions():
    # 解决 /emulation-executions?ids=true...
    return jsonify([{"id": 1, "name": "Standard-Execution"}])

# 为了保险，把这个万能拦截器也加上，给所有空对象一个默认字段
@app.after_request
def add_header(response):
    # 确保所有返回都是 JSON 格式
    response.headers["Content-Type"] = "application/json"
    return response
# --- 一个万能补丁：如果前端请求了我们没写的路由，统一返回空 JSON 而不是 404 ---
@app.errorhandler(404)
def not_found(e):
    return jsonify({}), 200

@app.route('/')
def serve_index():
    return send_from_directory('management-system/csle-mgmt-webapp/build', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('management-system/csle-mgmt-webapp/build', path)
# --- 1. 万能登录钥匙 ---
@app.route('/login', methods=['POST', 'GET'])
@app.route('/authenticate', methods=['POST', 'GET'])
@app.route('/api/auth/login', methods=['POST', 'GET'])
def fake_login():
    # 增加了一些常见的 Token 字段，确保前端总有一个能对上
    return jsonify({
        "token": "li-super-token-666",
        "access_token": "li-super-token-666",
        "id_token": "li-super-token-666",
        "status": "success",
        "success": True,
        "username": "admin",
        "roles": ["admin"],
        "user": {"id": 1, "username": "admin", "firstName": "Li", "lastName": "Admin"}
    }), 200

# --- 2. 核心数据路由 (解决转圈的关键) ---
@app.route('/simulations', methods=['GET'])
@app.route('/simulations/1', methods=['GET'])
def get_simulations_all():
    # 构造一个包含所有可能字段的“大礼包”
    data = {
        "id": 1,
        "name": "Standard-POMDP-Simulation",
        "status": "RUNNING",
        "execution_id": "exec-001",
        "num_steps": 100,
        "config": {
            "containers_config": {
                "containers": [
                    {"name": "attacker", "image": "csle-attacker", "ip": "172.18.0.2"},
                    {"name": "victim", "image": "csle-victim", "ip": "172.18.0.3"}
                ]
            },
            "executions": [],
            "network_config": {"subnets": []},
            "simulation_env_config": {"name": "env1"}
        },
        "simulation_trace": {"steps": []},
        "metrics": {"cpu": [], "memory": []}
    }
    # 如果是列表页就返回列表，如果是详情页就直接返回对象
    if request.path == '/simulations':
        return jsonify([data])
    return jsonify(data)

@app.route('/emulation-executions', methods=['GET'])
def get_executions_full():
    # 前端报错说 executions.length 不存在，我们给它一个带数据的列表
    return jsonify([{
        "id": 1,
        "name": "Exec-001",
        "status": "COMPLETED",
        "progress": 100,
        "emulation_name": "Campus-Network",
        "start_time": "2026-01-27 10:00:00"
    }])

@app.route('/nodes', methods=['GET'])
def get_nodes():
    return jsonify([
        {"id": 1, "name": "Attacker", "ip": "172.18.0.2", "role": "ATTACKER"},
        {"id": 2, "name": "Victim", "ip": "172.18.0.3", "role": "VICTIM"}
    ])

# --- 3. 基础健康检查 (有些前端会先调这个) ---
@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "CSLE-Bridge"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
