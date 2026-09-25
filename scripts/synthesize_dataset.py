#!/usr/bin/env python3
"""Synthesize dense radar tensors or detector-ready sparse radar points.

The dataset adapters generate samples from ``__getitem__``. This wrapper sends
their output to the directory selected on the command line.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_DATASETS = {
    "KRadarDetection_v2_1_for_INF",
    "KRadarDetection_v2_1_for_Kitti",
    "KRadarDetection_v2_1_for_nuscenes",
    "KRadarDetection_v2_1_for_ars548",
    "KRadarDetection_v2_1_for_vod",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dense tensors or detector-ready sparse points."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Generator checkpoint. Overrides DATASET.object_sample.l2r_path.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument(
        "--representation",
        choices=("sparse", "dense"),
        default="sparse",
        help="sparse saves detector-ready (x,y,z,power); dense saves C-RAE.",
    )
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--overwrite", action="store_true")
    args, legacy_args = parser.parse_known_args()
    if legacy_args[:1] == ["--"]:
        legacy_args = legacy_args[1:]
    return args, legacy_args


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_component(value):
    value = Path(str(value)).name
    # The historical nuScenes writer used split(".")[0], including for names
    # such as sample.pcd.bin.  Preserve that contract for the all-data loader.
    value = value.split(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def historical_filename(dict_item, fallback_index, representation, adapter_name):
    meta = dict_item.get("meta", {})
    dataset = meta.get("dataset", "unknown")
    seq = meta.get("seq", fallback_index)

    if dataset == "nuscenes":
        lidar_path = dict_item.get("lidar_path", {})
        if isinstance(lidar_path, dict):
            seq = lidar_path.get("lidar_path", seq)
    elif dataset == "kradar" or adapter_name == "KRadarDetection_v2_1_for_INF":
        radar_index = meta.get("idx", {}).get("rdr", fallback_index)
        seq = f"{seq}_{radar_index}"

    prefix = "rpc" if representation == "sparse" else "tensor"
    return f"{prefix}_{safe_component(seq)}.npy"


def main():
    args, legacy_args = parse_args()
    invocation_dir = Path.cwd()
    path_config = args.config.expanduser()
    path_output = args.output_dir.expanduser()
    path_checkpoint_override = args.checkpoint.expanduser() if args.checkpoint else None
    if not path_config.is_absolute():
        path_config = invocation_dir / path_config
    if not path_output.is_absolute():
        path_output = invocation_dir / path_output
    if path_checkpoint_override is not None and not path_checkpoint_override.is_absolute():
        path_checkpoint_override = invocation_dir / path_checkpoint_override
    path_config = path_config.resolve()
    path_output = path_output.resolve()
    if path_checkpoint_override is not None:
        path_checkpoint_override = path_checkpoint_override.resolve()

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.chdir(ROOT)
    default_legacy = [
        "--name",
        "l2rdas-export",
        "--checkpoints_dir",
        str(ROOT / "checkpoints"),
    ]
    sys.argv = [sys.argv[0], *default_legacy, *legacy_args]

    import numpy as np

    from options.test_options import TestOptions
    from utils.util_config import cfg, cfg_from_yaml_file
    from utils.util_pipeline import build_dataset

    loaded_cfg = cfg_from_yaml_file(str(path_config), cfg)
    if path_checkpoint_override is not None:
        if not path_checkpoint_override.is_file():
            raise FileNotFoundError(path_checkpoint_override)
        loaded_cfg.DATASET.object_sample.l2r_path = str(path_checkpoint_override)
    if not loaded_cfg.DATASET.object_sample.make_l2r_tensor:
        raise ValueError("The synthesis config must set make_l2r_tensor: true")
    if loaded_cfg.DATASET.NAME not in SUPPORTED_DATASETS:
        raise ValueError(
            "synthesize_dataset.py accepts only the K-Radar, KITTI, nuScenes, "
            "ARS548, and VoD generation adapters."
        )

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.no_instance = True

    class PipelineContext:
        pass

    context = PipelineContext()
    context.cfg = loaded_cfg
    dataset = build_dataset(context, split=args.split, option=opt)

    # The historical external adapters unconditionally built visualization
    # images during export.  Disable the side-effect without changing tensor
    # generation or sparsification.  The original methods remain untouched.
    if hasattr(dataset, "visualizer") and hasattr(dataset.visualizer, "save_images"):
        dataset.visualizer.save_images = lambda *_args, **_kwargs: None
    if hasattr(dataset, "generate_radar_tensor_for_vis"):
        dataset.generate_radar_tensor_for_vis = lambda dict_item: dict_item

    path_output.mkdir(parents=True, exist_ok=True)
    active_index = {"value": None}
    saved_index = {"value": None}

    def save_item(dict_item=None, **_unused):
        if dict_item is None:
            raise ValueError("The historical save callback received no item")
        index = active_index["value"]
        key = "rdr_sparse" if args.representation == "sparse" else "generated_tensor"
        if key not in dict_item:
            raise KeyError(f"Adapter did not produce {key}")
        filename = historical_filename(
            dict_item,
            fallback_index=index,
            representation=args.representation,
            adapter_name=loaded_cfg.DATASET.NAME,
        )
        destination = path_output / filename
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {destination}; pass --overwrite to replace it"
            )
        np.save(destination, dict_item[key])
        meta = dict_item.get("meta", {})
        record = {
            "index": index,
            "dataset": meta.get("dataset", ""),
            "sequence": meta.get("seq", ""),
            "file": filename,
            "bytes": destination.stat().st_size,
            "sha256": digest(destination),
        }
        index_writer.writerow(record)
        index_stream.flush()
        saved_index["value"] = index

    # Prevent the historical absolute-path writer from receiving control.
    if hasattr(dataset, "save_GT_rdr_pc"):
        dataset.save_GT_rdr_pc = save_item

    stop = len(dataset) if args.stop_index is None else min(args.stop_index, len(dataset))
    if args.start_index < 0 or args.start_index >= stop:
        raise ValueError(f"Invalid index interval [{args.start_index}, {stop})")

    chunk = f"{args.start_index:08d}_{stop:08d}"
    config_copy = path_output / f"config_{chunk}.yml"
    index_path = path_output / f"index_{chunk}.tsv"
    run_path = path_output / f"run_{chunk}.json"
    for path in (config_copy, index_path, run_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite metadata file {path}; pass --overwrite"
            )
    shutil.copy2(path_config, config_copy)

    checkpoint_value = str(loaded_cfg.DATASET.object_sample.l2r_path)
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    mask_path = ROOT / "mask.pt"
    run_record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_adapter": loaded_cfg.DATASET.NAME,
        "split": args.split,
        "start_index": args.start_index,
        "stop_index": stop,
        "representation": args.representation,
        "config_file": config_copy.name,
        "config_sha256": digest(config_copy),
        "generator_checkpoint": checkpoint_value,
        "generator_checkpoint_sha256": (
            digest(checkpoint_path) if checkpoint_path.is_file() else None
        ),
        "radar_fov_mask_sha256": digest(mask_path),
        "quantile_rate": float(loaded_cfg.DATASET.object_sample.QUANTILE_RATE),
        "synthesis_roi": list(loaded_cfg.DATASET.l2r.roi),
        "index_file": index_path.name,
        "index_hash_scheme": "per-file SHA-256 in a streaming TSV",
    }
    run_path.write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    completed = 0
    with index_path.open("w", encoding="utf-8", newline="") as index_stream:
        index_writer = csv.DictWriter(
            index_stream,
            fieldnames=("index", "dataset", "sequence", "file", "bytes", "sha256"),
            delimiter="\t",
        )
        index_writer.writeheader()
        for index in range(args.start_index, stop):
            active_index["value"] = index
            saved_index["value"] = None
            dict_item = dataset[index]
            if saved_index["value"] != index:
                save_item(dict_item=dict_item)
            completed += 1
            print(f"[{index + 1}/{stop}] completed")

    print(f"Wrote {completed} samples, {index_path}, and {run_path}")


if __name__ == "__main__":
    main()
