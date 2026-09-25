#!/usr/bin/env python3
"""Entry point for L2RDaS training and RTNH training/evaluation."""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run synthesis training, detector training, or evaluation."
    )
    parser.add_argument(
        "task",
        choices=("train-synthesis", "train-detector", "evaluate-detector"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Detector checkpoint; required by evaluate-detector.",
    )
    parser.add_argument(
        "--gpu",
        help="CUDA_VISIBLE_DEVICES value. Leave unset to inherit the environment.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        default="0.3,0.5,0.7",
        help="Comma-separated thresholds used for detector evaluation.",
    )
    parser.add_argument(
        "--allow-nonstrict-checkpoint",
        action="store_true",
        help="Allow missing/unexpected detector state keys during evaluation.",
    )
    args, legacy_args = parser.parse_known_args()
    if legacy_args[:1] == ["--"]:
        legacy_args = legacy_args[1:]
    return args, legacy_args


def save_executed_code(pipeline):
    """Store the launch script with the normal training logs."""
    shutil.copy2(Path(__file__).resolve(), Path(pipeline.path_log) / "executed_code.txt")


def build_synthesis_pipeline(path_config):
    import numpy as np

    from options.train_options import TrainOptions
    from pipelines.pipeline_for_GAN_v1_0 import PipelineGAN_v1_0

    opt = TrainOptions().parse()
    iter_path = Path(opt.checkpoints_dir) / opt.name / "iter.txt"
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(
                str(iter_path), delimiter=",", dtype=int
            )
        except Exception:
            start_epoch, epoch_iter = 1, 0
        print(
            "Resuming from epoch %d at iteration %d" % (start_epoch, epoch_iter)
        )

    opt.print_freq = (
        abs(opt.print_freq * opt.batchSize)
        // math.gcd(opt.print_freq, opt.batchSize)
        if opt.print_freq and opt.batchSize
        else 0
    )
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10
    opt.no_instance = True
    return PipelineGAN_v1_0(path_cfg=str(path_config), mode="train", option=opt)


def build_detector_pipeline(path_config, cfg, mode):
    from options.test_options import TestOptions
    from pipelines.pipeline_for_usingGAN_v1_0 import PipelineUsingGAN_v1_0

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = cfg.DATASET.Vis_mode
    opt.no_flip = True
    opt.no_instance = True
    return PipelineUsingGAN_v1_0(
        path_cfg=str(path_config), mode=mode, option=opt
    )


def main():
    args, legacy_args = parse_args()
    invocation_dir = Path.cwd()
    path_config = args.config.expanduser()
    if not path_config.is_absolute():
        path_config = invocation_dir / path_config
    path_config = path_config.resolve()
    path_checkpoint = None
    if args.checkpoint is not None:
        path_checkpoint = args.checkpoint.expanduser()
        if not path_checkpoint.is_absolute():
            path_checkpoint = invocation_dir / path_checkpoint
        path_checkpoint = path_checkpoint.resolve()

    # Historical configs use repository-relative resource paths.
    os.chdir(ROOT)
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # The legacy option classes parse sys.argv themselves.
    sys.argv = [sys.argv[0], *legacy_args]

    from utils.util_config import cfg, cfg_from_yaml_file

    loaded_cfg = cfg_from_yaml_file(str(path_config), cfg)
    if not loaded_cfg.GENERAL.LOGGING.IS_LOGGING:
        raise ValueError(
            "scripts/run.py requires GENERAL.LOGGING.IS_LOGGING: True because "
            "the training pipelines store their outputs under path_log"
        )
    makes_tensor = loaded_cfg.DATASET.object_sample.make_l2r_tensor

    if args.task == "train-synthesis":
        if makes_tensor:
            raise ValueError(
                "train-synthesis requires DATASET.object_sample.make_l2r_tensor: False"
            )
        pipeline = build_synthesis_pipeline(path_config)
        save_executed_code(pipeline)
        pipeline.train_network()
        return

    if not makes_tensor:
        raise ValueError(
            f"{args.task} requires DATASET.object_sample.make_l2r_tensor: True"
        )

    if args.task == "train-detector":
        pipeline = build_detector_pipeline(
            path_config,
            loaded_cfg,
            mode="train",
        )
        save_executed_code(pipeline)
        pipeline.train_network()
        return

    if path_checkpoint is None:
        raise ValueError("evaluate-detector requires --checkpoint")
    thresholds = [float(value) for value in args.confidence_thresholds.split(",")]
    pipeline = build_detector_pipeline(
        path_config,
        loaded_cfg,
        mode="test",
    )
    pipeline.load_dict_model(
        str(path_checkpoint), is_strict=not args.allow_nonstrict_checkpoint
    )
    pipeline.network.eval()
    save_executed_code(pipeline)
    pipeline.validate_kitti_conditional(
        list_conf_thr=thresholds,
        is_subset=False,
        is_print_memory=False,
    )


if __name__ == "__main__":
    main()
