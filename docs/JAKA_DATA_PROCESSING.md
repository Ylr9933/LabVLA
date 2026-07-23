# JAKA 数据处理

本文档对应 `scripts/prepare_jaka_dataset.py`，用于把原始 LeRobot v2.1
JAKA 数据转换为 LabVLA 使用的 canonical 数据集。

## 1. 输入数据

脚本要求源数据的 `meta/info.json` 声明以下字段：

```text
observation.joints   [6]
observation.gripper  [2]
action               [至少 7]
observation.images.front  video
```

`observation.gripper` 的默认布局是：

```text
[gripper_position, gripper_openness]
```

因此默认使用索引 `1` 的 `gripper_openness`。如果数据定义不同，使用
`--gripper_state_index` 显式指定。

脚本不会使用原始的 13 维 `observation.state` 作为模型输入，而是从
`observation.joints` 和 `observation.gripper` 明确构造 canonical state。

## 2. Canonical 布局

### 无底盘

```text
state/action = [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6,
                reserved_arm_slot, gripper_openness]
```

维度为 8：

```text
index 0..5: 六个 JAKA 关节
index 6:    固定为 0
index 7:    gripper_openness
```

action 的 delta mask 为：

```text
[true, true, true, true, true, true, false, false]
```

### 含底盘

底盘字段会追加在 canonical arm/gripper 前缀之后：

```text
state/action = [joint_1..joint_6, 0, gripper_openness,
                base_0, base_1, ...]
```

底盘 state/action 的字段名、索引和 action mode 必须通过命令行显式提供。
脚本不会根据 `observation.agv` 或 action 宽度自动猜测底盘语义。

底盘 action 为 `delta` 时，追加维度的 delta mask 为 `true`；为 `abs` 时
为 `false`。机械臂前 6 维始终为 delta，夹爪始终为 absolute。

## 3. 输出内容

输出目录包含：

```text
output/
  data/chunk-*/episode_*.parquet   canonical state/action
  videos/                          原视频，优先硬链接，跨文件系统时复制
  meta/info.json                   更新后的 feature shape/names
  meta/labvla_manifest.json        训练 schema
  meta/stats.json                  state/action/action_abs 统计量
```

`stats.json` 的 action 统计按照 `--chunk_size` 计算。`action` 统计使用
delta domain，`action_abs` 统计保留 absolute domain，并写入 `_chunk_size`
用于训练时校验。

## 4. 无底盘运行

```bash
python scripts/prepare_jaka_dataset.py \
    --source /data1/xuezirui/data_all/lerobot_v2_data_10 \
    --output /data1/xuezirui/data_all/jaka_canonical_arm \
    --chunk_size 50
```

如果夹爪 openness 不是 `observation.gripper[1]`：

```bash
python scripts/prepare_jaka_dataset.py \
    --source /path/to/raw_jaka \
    --output /path/to/jaka_canonical_arm \
    --gripper_state_index 1
```

## 5. 含底盘运行

下面命令只表示参数形式。底盘 action 字段必须是真实的动作字段，不能
把仅包含底盘观测的 `observation.agv` 当作 action 字段。

```bash
python scripts/prepare_jaka_dataset.py \
    --source /path/to/raw_jaka_mobile \
    --output /path/to/jaka_canonical_mobile \
    --chunk_size 50 \
    --base_state_key observation.agv \
    --base_state_indices 0,1,2 \
    --base_action_key action.base \
    --base_action_indices 0,1,2 \
    --base_action_mode delta
```

以下参数必须成组出现：

```text
--base_state_key
--base_state_indices
--base_action_key
--base_action_indices
```

启用底盘后还必须指定：

```text
--base_action_mode delta|abs
```

索引越界、字段缺失、state/action 维度不一致、NaN/Inf、帧数不一致都会
直接报错，不会静默填充或猜测。

## 6. 原数据保护

`--source` 只读。脚本只在 `--output` 中创建新数据、重写 parquet 和生成
统计文件，不会修改源数据。

如果 output 已存在，必须显式使用：

```bash
--force
```

这只会删除并重建 output，不会删除或修改 source。

## 7. 已验证结果

使用当前 JAKA 数据集验证：

```text
source: /data1/xuezirui/data_all/lerobot_v2_data_10
frames: 13447
arm-only state/action: 8 / 8
reserved index 6: 0.0
manifest: schema validation passed
stats chunk_size: 50
```

使用显式 3 维底盘字段映射测试：

```text
with-base state/action: 11 / 11
delta mask: [T,T,T,T,T,T,F,F,T,T,T]
```

该测试只验证处理流程，不代表 `observation.agv` 本身包含可训练的底盘
action。
