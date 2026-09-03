# JAKA 数据处理流程

本文档是当前 JAKA 数据从 raw episode 到 LabVLA LeRobot v2.1 数据集的唯一流程说明。
目前只维护两种 robot 模式和三种相机数量：

```text
非 mobile：JAKA 机械臂，8 维 canonical state/action
mobile：    JAKA 机械臂 + AGV 底盘，10 维 canonical state/action
相机：      1 路、2 路或 3 路，按 image0..imageN 连续编号
```

转换器位于：

```text
data_process/convert_jaka_rgb3_to_lerobot.py
```

通用统计入口仍然是：

```text
python -m data_process stats
```

## 1. Raw 数据要求

转换器要求 raw root 下存在：

```text
<raw-root>/episodes/episode_000000/
    video.mp4
    video_side.mp4
    video_wrist.mp4
    frames.csv
    frames_side.csv
    frames_wrist.csv
    states.csv
    actions.csv
    manifest.json
```

例如带底盘数据的 raw root：

```text
/data1/xuezirui/move_data/raw_datasets/jaka_raw
```

转换器以 `--cameras` 的第一路相机时间戳建立输出时间轴，其余选中相机使用最近时间戳帧；state
在选定时间戳线性插值。只保留选中相机和 state 时间范围的公共区间，避免用边界
样本制造错误同步。

`actions.csv` 不参与当前转换。机械臂 action 使用下一个采样 state；mobile
底盘 action 使用 `states.csv` 中的实测速度：

```text
observation.agv[3] = agv_linear_m_s
observation.agv[4] = agv_angular_rad_s
```

这两个量是 measured velocity，不是控制器 command。它们作为速度目标训练，部署
时模型也输出线速度和角速度。

## 2. 转换器用法

进入仓库并激活训练环境：

```bash
cd /data1/xuezirui/dev/LabVLA_JAKA
conda activate /data/rbc/miniconda3/envs/labvla
```

### 2.1 非 mobile

不加 `--mobile`，生成 arm-only 8 维数据：

```bash
python -m data_process.convert_jaka_rgb3_to_lerobot \
    --raw-root /path/to/raw_jaka_rgb3 \
    --output-parent /path/to/output_parent \
    --only both \
    --overwrite
```

指定相机数量和顺序，例如只用 front + wrist：

```bash
python -m data_process.convert_jaka_rgb3_to_lerobot \
    --raw-root /path/to/raw_jaka_rgb3 \
    --output-parent /path/to/output_parent \
    --cameras front,wrist \
    --only both \
    --overwrite
```

输出：

```text
<output-parent>/jaka_rgb2_lerobot_30hz
<output-parent>/jaka_rgb2_lerobot_10hz
```

非 mobile 原始字段：

```text
observation.joints       float32[6]
observation.gripper      float32[2]
action                   float32[7]
image0..imageN            选中的 1/2/3 路视频
```

训练 schema 按相机数量选择：

```text
schemas/jaka_v21.py:SCHEMA_RGB1 / SCHEMA_RGB2 / SCHEMA_RGB3
```

canonical 8 维布局：

```text
[joint1..joint6, 0, gripper_openness]
```

### 2.2 Mobile

带底盘数据必须加 `--mobile`：

```bash
python -m data_process.convert_jaka_rgb3_to_lerobot \
    --raw-root /data1/xuezirui/move_data/raw_datasets/jaka_raw \
    --output-parent /data1/xuezirui/move_data \
    --cameras front,side,wrist \
    --mobile \
    --only both \
    --overwrite
```

输出：

```text
/data1/xuezirui/move_data/jaka_mobile_rgb3_lerobot_30hz
/data1/xuezirui/move_data/jaka_mobile_rgb3_lerobot_10hz
```

例如 mobile 两路相机：

```bash
python -m data_process.convert_jaka_rgb3_to_lerobot \
    --raw-root /data1/xuezirui/jaka_raw_three_camera \
    --output-parent /data1/xuezirui/data_lab \
    --cameras front,side,wrist \
    --mobile --only both --overwrite
```

输出目录会自动使用 `jaka_mobile_rgb2_lerobot_*` 前缀。`--cameras` 可选：

```text
front
front,side
front,side,wrist
```

第一路相机是主时间轴；所有选中相机必须在每个 episode 中同时存在。

mobile 原始字段：

```text
observation.joints       float32[6]
observation.gripper      float32[2]
observation.agv          float32[9]
action                   float32[7]
image0..imageN            选中的 1/2/3 路视频
```

mobile schema 按相机数量选择：

```text
schemas/jaka_v21_mobile.py:SCHEMA_RGB1 / SCHEMA_RGB2 / SCHEMA_RGB3
```

canonical 10 维布局：

```text
[joint1..joint6, 0, gripper_openness,
 agv_linear_m_s, agv_angular_rad_s]
```

前 6 个关节 action 是 delta；保留位、夹爪和底盘速度是 absolute：

```text
delta_mask = [true, true, true, true, true, true, false, false, false, false]
```

## 3. 生成训练 stats

转换器会写基础 `meta/stats.json` 和一个初始 sidecar。正式训练前，统一使用
`data_process stats` 重新生成与训练代码完全一致的 canonical stats。

### 3.1 非 mobile 8 维 stats

```bash
python -m data_process stats \
    --dataset /path/to/output_parent/jaka_rgb3_lerobot_10hz \
    --schema /data1/xuezirui/dev/LabVLA_JAKA/schemas/jaka_v21.py:SCHEMA_RGB3 \
    --chunk_size 50 \
    --out /path/to/output_parent/jaka_rgb3_lerobot_10hz/meta/stats_labvla_jaka_8d.json
```

训练入口：

```text
launch/finetune/train_jaka.sh
```

### 3.2 Mobile 10 维 stats

```bash
python -m data_process stats \
    --dataset /data1/xuezirui/data_lab/jaka_mobile_rgb3_lerobot_10hz \
    --schema /data1/xuezirui/dev/LabVLA_JAKA/schemas/jaka_v21_mobile.py:SCHEMA \
    --chunk_size 50 \
    --out /data1/xuezirui/data_lab/jaka_mobile_rgb3_lerobot_10hz/meta/stats_labvla_jaka_mobile_10d.json
```

stats 文件必须包含：

```text
observation.state   10 维
action              10 维 delta-action
action_abs          10 维 absolute-action
_chunk_size         50
q01/q99             必须存在
```

mobile stats 计算会检查底盘两维不是常数；如果 `agv_linear_m_s` 或
`agv_angular_rad_s` 没有有效变化，命令会拒绝写出 stats。

`SCHEMA` 与各个 `SCHEMA_RGBN` 在 stats 计算上的 state/action 语义相同。stats
阶段不读取图像，因此使用 `SCHEMA` 不会丢失相机；训练阶段按相机数量使用对应
的 `SCHEMA_RGB1/2/3`。

不要使用 `--no-quantile`，默认训练和部署需要 q01/q99。

## 4. 训练入口

非 mobile：

```bash
bash launch/finetune/train_jaka.sh
```

mobile：

```bash
bash launch/finetune/train_jaka_mobile.sh
```

mobile launcher 默认使用：

```text
数据：/data1/xuezirui/move_data/jaka_mobile_rgb3_lerobot_10hz
schema：schemas/jaka_v21_mobile.py:SCHEMA_RGB3
stats：meta/stats_labvla_jaka_mobile_10d.json
chunk_size：50
```

也可以直接指定另一个 dataset root：

```bash
JAKA_DATA_ROOT=/path/to/jaka_mobile_rgb3_lerobot_10hz \
bash launch/finetune/train_jaka_mobile.sh
```

如果数据不是 3 路相机，设置相机数量使 launcher 选择对应 schema：

```bash
JAKA_CAMERA_COUNT=1 JAKA_DATA_ROOT=/path/to/jaka_mobile_rgb1_lerobot_10hz \
bash launch/finetune/train_jaka_mobile.sh
```

非 mobile launcher 使用同样的 `JAKA_CAMERA_COUNT` 环境变量。

只训练 action expert：编辑 launcher：

```bash
TrainExpertOnly=true
```

联合训练 VLM 和 expert：

```bash
TrainExpertOnly=false
```

## 5. 转换后检查

建议至少检查以下内容：

```bash
python -m data_process preflight \
    --repos jaka_mobile_rgb3_lerobot_10hz \
    --data_root /data1/xuezirui/move_data \
    --schema /data1/xuezirui/dev/LabVLA_JAKA/schemas/jaka_v21_mobile.py:SCHEMA_RGB3 \
    --external_stats_map jaka_mobile_rgb3_lerobot_10hz=/data1/xuezirui/move_data/jaka_mobile_rgb3_lerobot_10hz/meta/stats_labvla_jaka_mobile_10d.json \
    --chunk_size 50
```

并确认：

```text
meta/info.json                 fps / episode / frame 数正确
meta/labvla_manifest.json      schema_id=jaka_v21_mobile
meta/stats_labvla_jaka_mobile_10d.json  state/action=10, chunk=50
data/chunk-000/*.parquet       observation.agv[9]
videos/chunk-000/              image0..imageN 选中的视频
```

如果重新转换或修改了数据，必须重新生成对应 stats；不要把 8 维 stats 和
mobile 数据混用，也不要把 mobile 10 维 stats 交给 `train_jaka.sh`。

## 6. 当前保留的工具边界

JAKA 专用 raw RGB3 转换只保留：

```text
data_process/convert_jaka_rgb3_to_lerobot.py
data_process/stats/
```

旧的单相机转换、单 episode 下采样和旧 arm-only split 脚本已移除。通用
`data_process scan/clean/validate/preflight/stats` 工具仍然保留，供所有
LeRobot v2.1 数据集使用。
