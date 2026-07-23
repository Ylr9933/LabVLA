# JAKA 数据处理

本文档说明当前 JAKA 的训练数据链路。当前方案是训练时动态映射，原始
LeRobot 数据直接读取，不会生成或重写另一份数据集。

## 1. 当前训练方案

训练脚本为：

```text
launch/finetune/train_jaka.sh
```

默认路径已经配置为：

```text
LabVLA: /data1/xuezirui/LabVLA-5B-Base
Qwen:   /data/rbc/VLM/Qwen3-VL-4B-Instruct
数据:   /data1/xuezirui/data_all/lerobot_v2_data_10
```

运行：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA
bash launch/finetune/train_jaka.sh
```

训练时由 `schemas/jaka_v21.py` 和
`JakaStateGripperTransformFn` 完成字段映射。它们在样本进入归一化、delta
计算和模型之前执行，因此不会修改源 parquet、视频或 `meta/info.json`。

## 2. 输入数据

原始数据的 `meta/info.json` 需要声明以下字段：

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

因此训练时固定使用索引 `1` 的 `gripper_openness`。

脚本不会使用原始的 13 维 `observation.state` 作为模型输入，而是从
`observation.joints` 和 `observation.gripper` 明确构造 canonical state。

## 3. Canonical 布局

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

## 4. 服务于训练的统计 JSON

当前训练使用：

```text
/data1/xuezirui/data_all/lerobot_v2_data_10/meta/stats_labvla_jaka_8d.json
```

这个文件已经是 8 维 canonical state/action 的统计量，训练脚本通过
`--external_stats_path` 加载它。它不包含数据副本，也不会触发数据集重写。

## 5. 原数据保护

训练过程只读取：

```text
/data1/xuezirui/data_all/lerobot_v2_data_10/meta/info.json
/data1/xuezirui/data_all/lerobot_v2_data_10/data/
/data1/xuezirui/data_all/lerobot_v2_data_10/videos/
```

不会修改源数据，也不会创建新的 parquet 数据集。
