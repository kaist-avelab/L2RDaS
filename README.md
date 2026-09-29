# L2RDaS

Official code for [LiDAR-to-4D Radar Synthesis for Building Large-Scale
Tensor Datasets](https://openaccess.thecvf.com/content/CVPR2026F/papers/Jung_LiDAR-to-4D_Radar_Synthesis_for_Building_Large-Scale_Tensor_Datasets_CVPRF_2026_paper.pdf)
(CVPR Findings 2026).

This release contains three workflows only:

1. training the L2RDaS Synthesizer;
2. synthesizing radar data for K-Radar, KITTI, nuScenes, Dual-Radar/ARS548,
   and View-of-Delft (VoD); and
3. training and evaluating the RTNH 4D radar detector with real K-Radar and
   four synthesized external datasets.

RTNH is the only downstream detector included in this repository.

## Setup

The paper-era environment used Python 3.8, PyTorch 1.11.0, CUDA 11.3,
Open3D 0.15.1, and spconv for CUDA 11.3.

For background on the K-Radar environment, dataset, and revised-label
preparation, also refer to the official
[K-Radar detection setup guide](https://github.com/kaist-avelab/K-Radar/blob/main/docs/detection.md).
Some components in that repository are not included here; use the versions and
commands below as the authoritative setup for this L2RDaS release.

```bash
conda create -n l2rdas python=3.8 -y
conda activate l2rdas

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

cd ops
python setup.py develop
cd ..
```

Run all commands below from the repository root.

### Released artifacts

Download the three model files from
[GitHub Releases](https://github.com/kaist-avelab/L2RDaS/releases) and place
them in `checkpoints/`. The model binaries are distributed separately from the
normal Git history. `mask.pt` is included in the repository root.

| File | Purpose | Size | SHA-256 |
|---|---|---:|---|
| `checkpoints/L2RDaS_G_epoch23.pth` | trained L2RDaS Generator | 166,267,389 B | `ca8dd35795b845b13d2d0f2eb29dd334e24af31d99302124786a80e6b3344d6d` |
| `checkpoints/L2RDaS_D_epoch23.pth` | trained L2RDaS discriminator | 88,460,047 B | `99072d118c773d742d33d45e44d77f71048dd4f859178af71610259a4f20fe62` |
| `checkpoints/RTNH_model_8.pt` | RTNH trained with real K-Radar plus four synthesized external datasets | 69,419,293 B | `f186e9050a0df01114215c59852bd6869b18d328ad987d9ecfc4366922c24a26` |
| `mask.pt` | radar field-of-view mask used during synthesis and RTNH training | 4,719,339 B | `cdce875d3df7bca0622ebda55838291c623678588a7aa92033902237b5b633b5` |

The three model checksums are also stored in `checkpoints/SHA256SUMS`.

## 1. L2RDaS Training

The supplied training configuration is `configs/cfg_Lidar2Radar.yml`. It is a
portable configuration derived from the one used for the `val_11` run.
Training-related hyperparameters were retained, while machine-specific paths
and inactive entries were cleaned up for public release. A sanitized copy of
the archived command-line options is included as
`experiments/synthesis_val11/opt.txt`, with machine-specific paths converted
to the public repository layout.

### Models, configs, and entry points

| Role | File(s) | Description |
|---|---|---|
| Released checkpoints | `checkpoints/L2RDaS_G_epoch23.pth`<br>`checkpoints/L2RDaS_D_epoch23.pth` | Epoch-23 Generator and discriminator checkpoints; not required when training from scratch |
| Training config | `configs/cfg_Lidar2Radar.yml` | Portable configuration derived from the `val_11` training setup |
| Main entry point | `scripts/run.py` | Run with the `train-synthesis` task |
| Training pipeline | `pipelines/pipeline_for_GAN_v1_0.py` | Synthesizer optimization and checkpoint saving |
| Dataset preparation | `datasets/kradar_detection_v2_1.py` | K-Radar loading, OBIS, and C-RAE tensor preparation |
| Model implementation | `models/skeletons/l2r_net.py`<br>`models/GAN/` | LiDAR voxel feature encoder, Generator, and discriminator implementation |
| Saved CLI options | `experiments/synthesis_val11/opt.txt` | Sanitized `val_11` command-line options with portable paths |

The dataset loader applies OBIS before voxelization. It augments the LiDAR
input with class-specific Gaussian object points and bounding-box boundary
points sampled at 0.1 m intervals. The resulting seven input channels are
`x`, `y`, `z`, `intensity`, `Sedan Gaussian`, `Bus/Truck Gaussian`, and
`boundary`.

### Prepare K-Radar

Download K-Radar and set the following paths in
`configs/cfg_Lidar2Radar.yml`:

```yaml
DATASET:
  path_data:
    list_dir_kradar: ["/path/to/K-Radar/sequences"]
    split: ["./resources/split/train.txt", "./resources/split/test.txt"]
    revised_label_v2_0: "/path/to/K-Radar/revised_labels/v2_0"
    l2r_path: "./data/cache/l2r_data"
```

`l2r_path` should point to a writable cache directory. The supplied
configuration is already set for L2RDaS training with:

```yaml
DATASET:
  object_sample_mode: false
  object_sample:
    make_l2r_tensor: false
MODEL:
  SKELETON: l2r
```

### Run training

The following command matches the released `val_11` training options. It uses
a batch size of 1, 7 input channels, Generator base filters (`ngf`) of 32,
discriminator base filters (`ndf`) of 64, two discriminators (`num_D=2`),
100 constant-learning-rate epochs followed by 100 learning-rate decay epochs,
an Adam learning rate of `0.0002`, and `beta1=0.5`.

```bash
python scripts/run.py train-synthesis \
  --config configs/cfg_Lidar2Radar.yml \
  --gpu 0 -- \
  --name val_11 \
  --checkpoints_dir ./checkpoints \
  --dataroot ./data/cache/l2r_data \
  --batchSize 1 \
  --nThreads 2 \
  --input_nc 7 \
  --output_nc 1 \
  --ngf 32 \
  --ndf 64 \
  --n_downsample_global 4 \
  --n_blocks_global 9 \
  --num_D 2 \
  --n_layers_D 3 \
  --niter 100 \
  --niter_decay 100 \
  --lr 0.0002 \
  --beta1 0.5 \
  --display_freq 1000 \
  --print_freq 1000 \
  --save_latest_freq 5000 \
  --save_epoch_freq 1
```

During training, Generator and discriminator checkpoints are saved under:

```text
checkpoints/val_11/<epoch>_net_G.pth
checkpoints/val_11/<epoch>_net_D.pth
```

Training logs are written to the experiment log directory:

```text
logs/exp_<date>_<time>_L2RDaS/
```

The released epoch-23 checkpoints are provided under the stable filenames:

```text
checkpoints/L2RDaS_G_epoch23.pth
checkpoints/L2RDaS_D_epoch23.pth
```

## 2. How to Synthesize K-Radar, KITTI, etc.

The released `checkpoints/L2RDaS_G_epoch23.pth` checkpoint is used to synthesize
radar data for all supported datasets. By default, the exporter produces sparse
arrays with shape `N x 4`, where each row is represented as
`[x, y, z, radar_power]`. Use `--representation dense` when the full C-RAE
tensor representation is required.

### Models, configs, and entry points

| Role | File(s) | Description |
|---|---|---|
| Released Generator checkpoint | `checkpoints/L2RDaS_G_epoch23.pth` | Generator checkpoint used for radar synthesis across the supported datasets |
| Main entry point | `scripts/synthesize_dataset.py` | Loads the dataset-specific configuration, runs L2RDaS inference, and exports synthesized radar data |
| Model implementation | `models/skeletons/l2r_net.py`<br>`models/GAN/` | LiDAR feature encoding and Generator implementation used for L2RDaS inference |
| Radar FOV mask | `mask.pt` | Radar field-of-view mask referenced by the synthesis configurations |
| Dataset configs/loaders | See the table below | Dataset-specific configuration and loader for each supported dataset |

### Dataset configs and loaders

| Dataset | Config | Loader | Paths to edit under `DATASET.path_data` |
|---|---|---|---|
| K-Radar | `configs/cfg_Lidar2Radar_reference.yml` | `datasets/kradar_detection_v2_1_inference.py` | `list_dir_kradar`, `split`, `revised_label_v2_0` |
| KITTI | `configs/cfg_Lidar2Radar_reference_kitti.yml` | `datasets/kradar_detection_v2_1_kitti.py` | `kitti`, `split` |
| nuScenes | `configs/cfg_Lidar2Radar_reference_nuscenes.yml` | `datasets/kradar_detection_v2_1_nuscenes.py` | `nuscenes`, `nuscenes_info` |
| Dual-Radar/ARS548 | `configs/cfg_Lidar2Radar_reference_ars548.yml` | `datasets/kradar_detection_v2_1_ars548.py` | `ars548`, `ars548_info` |
| View-of-Delft (VoD) | `configs/cfg_Lidar2Radar_reference_vod_lidar.yml` | `datasets/kradar_detection_v2_1_vod.py` | `vod`, `split` |

The released synthesis configurations are set to load the L2RDaS Generator
checkpoint as follows:

```yaml
DATASET:
  object_sample:
    make_l2r_tensor: true
    l2r_path: "./checkpoints/L2RDaS_G_epoch23.pth"
```

To match the object-aware synthesis setup used in the paper, set
`DATASET.object_sample.file_path` in each configuration to the K-Radar
ground-truth database file. For example:

```yaml
DATASET:
  object_sample:
    file_path: "/path/to/K-Radar/kradar_gt_database/kradar_dbinfos_train.pkl"
```

This database provides the K-Radar object information used by the released
object-aware synthesis pipeline. Users should update this path according to
their local K-Radar installation.

### Synthesize sparse radar data

The following examples generate detector-ready sparse radar data for each
supported dataset.

```bash
# K-Radar
python scripts/synthesize_dataset.py \
  --config configs/cfg_Lidar2Radar_reference.yml \
  --checkpoint checkpoints/L2RDaS_G_epoch23.pth \
  --output-dir output/kradar \
  --split train --representation sparse --gpu 0

# KITTI
python scripts/synthesize_dataset.py \
  --config configs/cfg_Lidar2Radar_reference_kitti.yml \
  --checkpoint checkpoints/L2RDaS_G_epoch23.pth \
  --output-dir output/kitti \
  --split train --representation sparse --gpu 0

# nuScenes
python scripts/synthesize_dataset.py \
  --config configs/cfg_Lidar2Radar_reference_nuscenes.yml \
  --checkpoint checkpoints/L2RDaS_G_epoch23.pth \
  --output-dir output/nuscenes \
  --split train --representation sparse --gpu 0

# Dual-Radar / ARS548
python scripts/synthesize_dataset.py \
  --config configs/cfg_Lidar2Radar_reference_ars548.yml \
  --checkpoint checkpoints/L2RDaS_G_epoch23.pth \
  --output-dir output/ars548 \
  --split train --representation sparse --gpu 0

# View-of-Delft (VoD)
python scripts/synthesize_dataset.py \
  --config configs/cfg_Lidar2Radar_reference_vod_lidar.yml \
  --checkpoint checkpoints/L2RDaS_G_epoch23.pth \
  --output-dir output/vod \
  --split train --representation sparse --gpu 0
```

Existing output files are not overwritten by default. Use `--overwrite` only
when an existing synthesized file should be replaced.

For large datasets, synthesis can be divided into smaller subsets using
`--start-index` and `--stop-index`.

### Synthesize dense C-RAE tensors

To export the full C-RAE tensor instead of the sparse representation, use
`--representation dense`. For example, the following command exports the KITTI
training split in dense format:

```bash
python scripts/synthesize_dataset.py \
  --config configs/cfg_Lidar2Radar_reference_kitti.yml \
  --checkpoint checkpoints/L2RDaS_G_epoch23.pth \
  --output-dir output/kitti_dense \
  --split train --representation dense --gpu 0
```

## 3. How to Train the 4D Radar Detection Model (RTNH Only)

The supplied `configs/cfg_RTNH.yml` is a portable configuration derived from
the one used for the `exp_250508_030854_RTNH` experiment. That experiment was
configured with real K-Radar data and four synthesized external datasets.
RTNH model settings and training-related hyperparameters were retained, while
machine-specific filesystem paths and inactive entries were cleaned up for
public release.

The released RTNH checkpoint,
`checkpoints/RTNH_model_8.pt`, corresponds to `model_8.pt` from this
experiment.

### Models, configs, and entry points

| Role | File(s) | Description |
|---|---|---|
| Released RTNH checkpoint | `checkpoints/RTNH_model_8.pt` | `model_8.pt` from `exp_250508_030854_RTNH`; not required when training from scratch |
| Training/evaluation config | `configs/cfg_RTNH.yml` | Portable configuration derived from the saved RTNH experiment setup |
| Main entry point | `scripts/run.py` | Run with the `train-detector` or `evaluate-detector` task |
| Training/evaluation pipeline | `pipelines/pipeline_for_usingGAN_v1_0.py` | RTNH training, checkpoint saving, inference, and evaluation |
| Combined dataset loader | `datasets/kradar_detection_v2_1_all.py` | Loads real K-Radar data and the synthesized external datasets used by the released RTNH setup |
| RTNH network | `models/skeletons/rdr_base.py` | RTNH model assembly |
| Sparse radar processor | `models/pre_processor/rdr_sparse_processor.py` | Sparse radar preprocessing |
| 3D sparse backbone | `models/backbone_3d/rdr_sp_pw.py` | RTNH sparse 3D backbone |
| Detection head | `models/head/anchor_head.py` | RTNH anchor-based detection head |
| Radar FOV mask | `mask.pt` | Radar field-of-view mask referenced by the RTNH configuration |

### Prepare real and synthesized radar data

Edit `DATASET.path_data` in `configs/cfg_RTNH.yml` to match the local paths
for the required dataset roots, split files, metadata files, and K-Radar
`revised_label_v2_0` directory.

Then set the sparse radar paths as follows:

```yaml
DATASET:
  object_sample:
    file_path: "/path/to/K-Radar/kradar_gt_database/kradar_dbinfos_train.pkl"
  rdr_sparse:
    default_dir: "/path/to/K-Radar/radar_sparse_train"
    test_dir: "/path/to/K-Radar/radar_sparse_test"
    synthesized_dirs:
      kitti: "./output/kitti"
      nuscenes: "./output/nuscenes"
      ars548: "./output/ars548"
      vod: "./output/vod"
```

In the released setup, `default_dir` points to the real K-Radar sparse radar
data used for training, while `test_dir` points to the real K-Radar sparse
radar data used for evaluation. The four entries under `synthesized_dirs`
point to the synthesized external datasets generated using the workflow in
Section 2.

The expected sparse radar filenames are:

```text
K-Radar train: <default_dir>/<sequence>/rpc_0_<radar-index>_False_-1.npy
K-Radar test:  <test_dir>/<sequence>/rpc_0_<radar-index>_False.npy
External:      <synthesized_dir>/rpc_<sample-id>.npy
```

### Train RTNH

The released configuration uses a random seed of `2025`, a batch size of `8`,
four data-loading workers, the AdamW optimizer with a learning rate of
`0.001`, cosine annealing, and `25` training epochs.

```bash
python scripts/run.py train-detector \
  --config configs/cfg_RTNH.yml \
  --gpu 0
```

During training, model checkpoints and TensorBoard logs are written under the
experiment log directory:

```text
logs/exp_<date>_<time>_RTNH/models/model_<epoch>.pt
logs/exp_<date>_<time>_RTNH/train_iter/
logs/exp_<date>_<time>_RTNH/train_epoch/
```

### RTNH inference and evaluation

Use the released checkpoint as follows:

```bash
python scripts/run.py evaluate-detector \
  --config configs/cfg_RTNH.yml \
  --checkpoint checkpoints/RTNH_model_8.pt \
  --confidence-thresholds 0.3,0.5,0.7 \
  --gpu 0
```

By default, the released evaluation entry point loads the checkpoint using
strict state-dictionary matching. Evaluation predictions and KITTI-format
result files are written to the log directory created for the evaluation run.

## License

AVE Lab-authored code is released under the Apache License 2.0.

Code derived from pix2pixHD remains subject to its original license terms,
which are provided in:

`licenses/pix2pixHD-LICENSE.txt`

K-Radar, KITTI, nuScenes, Dual-Radar/ARS548, and View-of-Delft are third-party
datasets and are not covered by the repository's Apache-2.0 license. Users are
responsible for obtaining and using each dataset in accordance with its
respective license and terms of use.
