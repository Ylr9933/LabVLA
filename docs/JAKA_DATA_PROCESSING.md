# JAKA 数据链路与训练说明

本文档定义当前项目对 JAKA 数据的训练契约，说明原始字段如何变成模型输入、
统计量如何生成和使用，以及启动训练前必须检查的条件。

当前实现对应的是 **JAKA 机械臂 8 维训练**。底盘数据虽然记录在原始数据中，
但尚未作为 policy 的 action 目标接入训练。

## 1. 训练入口

训练脚本：

```text
launch/finetune/train_jaka.sh
```

当前默认配置：

```text
LabVLA checkpoint: /data1/xuezirui/LabVLA-5B-Base
Qwen processor:   /data/rbc/VLM/Qwen3-VL-4B-Instruct
dataset root:     /data1/xuezirui/data_all/lerobot_v2_data_10
stats file:       /data1/xuezirui/data_all/lerobot_v2_data_10/meta/stats_labvla_jaka_8d.json
chunk size:       50
action mode:      delta
```

启动命令：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA
conda activate /data/rbc/miniconda3/envs/labvla
bash launch/finetune/train_jaka.sh
```

训练时直接读取原始 parquet 和视频。字段映射在内存中完成，不会创建新的
parquet 数据集，也不会修改原始数据。

## 2. 原始数据契约

`meta/info.json` 中当前相关字段如下：

```text
observation.joints       [6]
observation.gripper      [2]
observation.agv          [9]
observation.images.front  video
action                   [7]
```

夹爪字段的定义是：

```text
observation.gripper = [gripper_position, gripper_openness]
action              = [joint_1, ..., joint_6, gripper_openness]
```

原始 `observation.state` 的 13 维复合向量不直接作为模型 state。训练使用
`observation.joints` 和 `observation.gripper` 显式构造 canonical state。

## 3. Canonical 8 维布局

`schemas/jaka_v21.py:SCHEMA` 定义当前 schema，
`JakaStateGripperTransformFn` 执行映射：

```text
state  = [observation.joints[0:6], 0, observation.gripper[1]]
action = [action[0:6],              0, action[6]]
```

因此每个向量的含义为：

| 索引 | 含义 | 处理 |
|---|---|---|
| 0-5 | JAKA 六个关节 | 参与 delta 和归一化 |
| 6 | 保留的第七个 arm slot | 固定为 0，不参与 delta，归一化后仍为 0 |
| 7 | 夹爪 openness | absolute，使用 q01/q99 归一化 |

当前 schema 的 delta mask 为：

```text
[true, true, true, true, true, true, false, false]
```

这意味着前 6 个动作维度使用“动作减当前 state”的 delta，保留位和夹爪
保持 absolute。

## 4. 底盘数据的边界

原始数据包含：

```text
observation.agv = [
    agv_x, agv_y, agv_theta,
    agv_linear, agv_angular,
    agv_power_percent, agv_is_moving,
    agv_charge_state, agv_estop_state
]
```

其中 `agv_linear` 和 `agv_angular` 是底盘运动相关的观测量，但当前正式
`action` 字段仍然只有 7 维机械臂 action。当前 schema 不会自动把
`observation.agv` 当作动作，也不会把观测速度直接伪装成控制指令。

因此当前训练结果是机械臂模型，不是机械臂加底盘联合控制模型。要训练联合
控制，必须先确定底盘 action 的真实语义、时间对齐方式和动作维度，再单独
定义 mobile schema 与 transform。

## 5. 归一化流程

训练样本按以下顺序处理：

```text
原始 parquet
    -> JAKA 字段映射为 8 维 state/action
    -> action delta 计算
    -> state/action 归一化
    -> 拼接字段与 padding 到 32 维
    -> 输入模型
```

### 5.1 统计文件

训练脚本通过 `--external_stats_path` 使用：

```text
/data1/xuezirui/data_all/lerobot_v2_data_10/meta/stats_labvla_jaka_8d.json
```

文件包含：

```text
observation.state  canonical 8 维 state 统计
action              canonical 8 维 delta-action 统计
action_abs          canonical 8 维 absolute-action 统计
_chunk_size         统计时使用的 action chunk 长度
```

其中 `action` 的前 6 维统计的是 delta 分布，`action_abs` 保留原始绝对动作
分布。当前 `ActionMode=delta`，所以训练归一化使用 `action`。

### 5.2 归一化公式

机械臂关节默认使用 mean/std：

```text
x_norm = (x - mean) / (std + 1e-6)
```

夹爪维度使用 q01/q99 映射到约 `[-1, 1]`：

```text
x_norm = 2 * (x - q01) / (q99 - q01 + 1e-6) - 1
```

当前配置下：

```text
state[0:6]  mean/std
state[7]    q01/q99
action[0:6] mean/std
action[7]   q01/q99
```

第 6 个保留维度原始值恒为 0，因此不会产生有意义的控制信号。

## 6. 重新生成统计量

统计脚本会遍历原始数据计算统计量，但只写出一个 JSON，不会重写 parquet、
视频或 `meta/info.json`：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA
conda activate /data/rbc/miniconda3/envs/labvla

python -m data_process stats \
    --dataset /data1/xuezirui/data_all/lerobot_v2_data_10 \
    --schema /data1/xuezirui/dev/LabVLA_JAKA/schemas/jaka_v21.py:SCHEMA \
    --chunk_size 50 \
    --out /data1/xuezirui/data_all/lerobot_v2_data_10/meta/stats_labvla_jaka_8d.json
```

统计脚本中的 JAKA canonicalization 与训练时的
`JakaStateGripperTransformFn` 对齐，避免统计空间和训练输入空间不一致。

## 7. 训练前检查

当前已验证：

```text
schema:       jaka_v21_arm_only
episodes:     21
frames:       13447
coverage:     100%
state dim:    8
action dim:   8
chunk size:   50
stats file:   与重新生成结果 SHA256 完全一致
```

同时已检查以下路径存在：

```text
/data1/xuezirui/LabVLA-5B-Base
/data/rbc/VLM/Qwen3-VL-4B-Instruct
/data1/xuezirui/data_all/lerobot_v2_data_10
/data1/xuezirui/data_all/lerobot_v2_data_10/meta/stats_labvla_jaka_8d.json
/data1/xuezirui/dev/LabVLA_JAKA/configs/deepspeed_zero2.json
```

因此，从数据契约、schema、统计量和训练脚本配置看，可以直接开始当前的
机械臂 8 维训练。实际 GPU 训练尚未在本次检查中启动；启动前仍需确认当前
shell 使用的是 `labvla` 环境，并确保 PyTorch/CUDA/cuDNN 可正常导入。
