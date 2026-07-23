# JAKA + AGV 联合数据、训练与部署

本文档定义 mobile 变体。它与 arm-only 变体完全分离：使用独立 schema、统计
文件、训练脚本和部署入口，不改变原有 JAKA 机械臂训练和部署。

## 1. 适用数据

当前原始数据的正式 `action` 仍是 7 维机械臂 action：

```text
action = [joint_1, ..., joint_6, gripper_openness]
```

底盘相关字段位于：

```text
observation.agv = [
    agv_x, agv_y, agv_theta,
    agv_linear, agv_angular,
    agv_power_percent, agv_is_moving,
    agv_charge_state, agv_estop_state
]
```

本 mobile 方案明确采用 `agv_linear` 和 `agv_angular` 作为底盘速度目标。
它们是记录到数据中的速度观测，不一定等价于底盘控制器接收的原始 command；
如果后续拿到真实 command 列，应修改 schema/transform 使用 command，而不是
继续沿用观测速度。

当前 `/data1/xuezirui/data_all/lerobot_v2_data_10` 中这两列实际为常量 0，因而
只能验证字段链路，不能用于有意义的 mobile policy 训练。mobile 训练入口会
检查 action 统计量，并在底盘标签恒定时拒绝启动。

## 2. 10 维 canonical contract

schema：

```text
schemas/jaka_v21_mobile.py:SCHEMA
```

如果 parquet 的 `action` 已经扩展为 9 维 `[6 joints, gripper, linear, angular]`，
使用：

```text
schemas/jaka_v21_mobile_action9.py:SCHEMA
```

state 和 action 的布局：

```text
[j1, j2, j3, j4, j5, j6, 0, gripper_openness,
 agv_linear, agv_angular]
```

对应的 delta mask：

```text
[true, true, true, true, true, true, false, false, false, false]
```

| 维度 | 含义 | 训练后处理 |
|---|---|---|
| 0-5 | 六个机械臂关节 | delta + mean/std |
| 6 | 保留 arm slot | 固定 0 |
| 7 | 夹爪 openness | absolute + q01/q99 |
| 8 | AGV linear velocity | absolute + mean/std |
| 9 | AGV angular velocity | absolute + mean/std |

`jaka_v21_mobile` 针对当前 7 维 action 数据，从 `observation.agv[3:5]` 生成
底盘动作标签；`jaka_v21_mobile_action9` 针对已经包含底盘动作的 9 维 action
数据，直接使用 `action[7:9]`。两种 schema 不混用。

## 3. 生成统计文件

统计计算只读取原始数据并写入一个独立 sidecar，不重写 parquet：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA
conda activate /data/rbc/miniconda3/envs/labvla

python -m data_process stats \
    --dataset /data1/xuezirui/data_all/lerobot_v2_data_10 \
    --schema /data1/xuezirui/dev/LabVLA_JAKA/schemas/jaka_v21_mobile.py:SCHEMA \
    --chunk_size 50 \
    --out /data1/xuezirui/data_all/lerobot_v2_data_10/meta/stats_labvla_jaka_mobile_10d.json
```

拿到非恒定底盘 action 后，生成的 sidecar 应放在：

```text
/data1/xuezirui/data_all/lerobot_v2_data_10/meta/stats_labvla_jaka_mobile_10d.json
```

它应包含 10 维 `observation.state`、`action` 和 `action_abs` 统计，且
`_chunk_size=50`。当前数据的底盘标签恒定为 0，统计脚本会拒绝写出这个
sidecar，避免误启动无效的 mobile 训练。

## 4. 训练

独立训练脚本：

```text
launch/finetune/train_jaka_mobile.sh
```

启动：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA
conda activate /data/rbc/miniconda3/envs/labvla
bash launch/finetune/train_jaka_mobile.sh
```

`train_jaka.sh` 不会读取这个 mobile schema，因此原来的 arm-only 训练不受
影响。

## 5. 部署

独立部署入口：

```text
deployment/serve_jaka_mobile.py
deployment/deploy_jaka_mobile.sh
```

启动：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA

PRETRAINED_PATH=/path/to/jaka_mobile/checkpoint-5000 \
CUDA_VISIBLE_DEVICES=0 \
bash deployment/deploy_jaka_mobile.sh
```

默认地址仍为：

```text
ws://127.0.0.1:31002
```

mobile 服务要求 checkpoint 的 schema 为 `jaka_v21_mobile`，并在加载大模型
之前检查 state/action 维度为 10。它复用通用 LabVLA 的归一化逆变换和 delta
还原，因此返回的是 absolute action：

```text
[joint_1_target, ..., joint_6_target, 0,
 gripper_openness_target, agv_linear_target, agv_angular_target]
```

客户端必须发送 10 维 state：

```text
[j1, j2, j3, j4, j5, j6, 0, gripper_openness,
 agv_linear, agv_angular]
```

响应默认是 `[50, 10]` 的 action chunk。可以用 `OUTPUT_CHUNK_SIZE` 只返回
前 N 步进行联调。

## 6. 风险边界

这套 mobile 方案不会自动推断底盘 action，也不会使用电量、充电、急停等状态
作为动作目标。它只使用明确约定的线速度和角速度两维。

正式接入真实底盘前，必须确认：

1. `agv_linear/agv_angular` 的单位和控制接口单位一致；
2. 记录值是 command 还是 measured velocity；
3. action 的时间戳与机械臂 action 对齐；
4. 输出速度是否需要安全限幅、死区和急停策略。

模型服务不替代机器人控制器的安全限制。
