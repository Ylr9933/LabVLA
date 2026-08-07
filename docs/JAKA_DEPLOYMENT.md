# JAKA LabVLA 部署说明

本文档描述 JAKA 8 维机械臂模型的部署契约和启动方法。部署代码复用通用
LabVLA 的 WebSocket、模型加载、归一化反变换、delta action 还原和并发控制
实现，只在 JAKA 入口增加模型契约校验。

## 1. 文件结构

```text
deployment/
  serve_labvla.py   通用 LabVLA 推理核心和 WebSocket 服务
  serve_jaka.py     JAKA checkpoint 校验和专用入口
  deploy_jaka.sh    JAKA 默认环境与启动参数封装
```

`serve_jaka.py` 不复制通用推理逻辑，而是先验证 checkpoint，再把请求转交给
`serve_labvla.py`。这样 JAKA 和其他 LabVLA 部署使用同一套协议和后处理代码。

## 2. JAKA 模型契约

部署入口只接受以下 schema：

```text
schema_id:           jaka_v21_arm_only
state_dim:           8
action_dim:          8
delta_mask:          [true, true, true, true, true, true, false, false]
gripper_action_dims: [7]
```

state 和 action 的 canonical 布局都是：

```text
[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,
 reserved_arm_slot, gripper_openness]
```

第 6 维是保留位，固定为 0；第 7 维是夹爪 openness。部署服务不会自动接受
6 维 state，也不会把 `observation.agv` 拼到底盘动作中。

## 3. 输入输出协议

服务使用与原 LabVLA 相同的 WebSocket + msgpack 协议。客户端连接后首先收到
metadata，之后反复发送 observation 并接收 action result。

### 3.1 请求

```python
{
    "camera_1_rgb": image0,
    "camera_2_rgb": image1,       # 可选，按 checkpoint camera mapping 使用
    "camera_3_rgb": image2,       # 可选
    "state": np.ndarray(shape=(8,), dtype=np.float32),
    "prompt": "执行当前任务"
}
```

也接受以下兼容字段：

```text
language_instruction  -> prompt
observation/state      -> state
```

至少需要一张有效图像、一个非空 prompt 和 8 维有限值 state。缺字段、维度
错误或 NaN/Inf 会返回结构化错误，不会生成全零动作。

### 3.2 响应

```python
{
    "actions": np.ndarray(shape=(50, 8), dtype=np.float32),
    "policy_timing": {"infer_ms": ...}
}
```

动作已经完成训练侧归一化的逆变换和 delta 到 absolute 的还原，客户端可直接
按时间执行。其布局为：

```text
[joint_1_target, ..., joint_6_target, 0, gripper_openness_target]
```

`OUTPUT_CHUNK_SIZE` 可以只返回前 N 步用于调试，但不会改变模型的内部
`CHUNK_SIZE`。

## 4. 默认配置

`deployment/deploy_jaka.sh` 的默认值：

```text
VLM:        /data/rbc/VLM/Qwen3-VL-4B-Instruct
HOST:       127.0.0.1
PORT:       31002
DEVICE:     cuda
GPU:        CUDA_VISIBLE_DEVICES 或 0
CONDA_ENV:  labvla
CHUNK_SIZE: 50
```

默认监听地址和端口与原 `deployment/deploy.sh` 一致：`127.0.0.1:31002`。

## 5. 启动服务

checkpoint 必须是训练输出目录或其中的 checkpoint 目录，并包含当前训练流程
保存的 `labvla_schema.json` 及归一化统计 sidecar。示例：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA

PRETRAINED_PATH=/data1/xuezirui/dev/LabVLA_JAKA/outputs/<job>/checkpoint-5000 \
CUDA_VISIBLE_DEVICES=0 \
bash deployment/deploy_jaka.sh
```

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA

PRETRAINED_PATH=/data2/xuezirui/outputs/labvla_finetune_jaka_mobile_20260806_100801/checkpoint-25000 \
CUDA_VISIBLE_DEVICES=5 \
bash /data1/xuezirui/dev/LabVLA_JAKA/deployment/deploy_jaka_mobile.sh

```

cd /data1/xuezirui/dev/LabVLA_JAKA
conda activate /data/rbc/miniconda3/envs/labvla
PRETRAINED_PATH=/data2/xuezirui/outputs/labvla_finetune_jaka_20260801_093950/checkpoint-30000 \
CUDA_VISIBLE_DEVICES=5 \
bash deployment/deploy_jaka.sh

使用其他端口：

```bash
PORT=31003 \
PRETRAINED_PATH=/path/to/checkpoint-5000 \
bash deployment/deploy_jaka.sh
```

如果 checkpoint 没有保存训练时的 prompt，可以设置默认 prompt：

```bash
DEFAULT_PROMPT="完成当前操作" \
PRETRAINED_PATH=/path/to/checkpoint-5000 \
bash deployment/deploy_jaka.sh
```

如果需要限制返回长度：

```bash
OUTPUT_CHUNK_SIZE=10 \
PRETRAINED_PATH=/path/to/checkpoint-5000 \
bash deployment/deploy_jaka.sh
```

## 6. 局域网访问

默认只监听本机，适合本机客户端。绑定到局域网地址时必须配置认证 token：

```bash
HOST=0.0.0.0 \
LABVLA_WS_AUTH_TOKEN='replace-with-a-long-random-token' \
PRETRAINED_PATH=/path/to/checkpoint-5000 \
bash deployment/deploy_jaka.sh
```

没有 token 时，服务会拒绝非 loopback 地址，避免误暴露控制接口。

## 7. 启动前检查

```bash
bash -n deployment/deploy_jaka.sh
/data/rbc/miniconda3/envs/labvla/bin/python -m py_compile \
    deployment/serve_jaka.py deployment/serve_labvla.py
```

启动日志应明确显示：

```text
schema_id=jaka_v21_arm_only
action_dim=8
state_dim=8
action_mode=delta
```

如果 schema、维度、delta mask 或夹爪索引不匹配，`serve_jaka.py` 会在加载大
模型前直接退出，避免错误模型进入机器人控制链路。
