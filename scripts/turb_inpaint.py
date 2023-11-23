"""
Reconstruct a large batch of gappy 2D flow fields from a model and save them as
a large numpy array. The conditioning strategy is from Lugmayr et al. (2022).
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.turb_datasets import load_data_inpaint
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.scheduler import get_schedule_jump


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    data = load_data_inpaint(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        mask_gt_name=args.mask_gt_name,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("creating resampling schedule...")
    resampling_indices = get_schedule_jump(
        t_T=args.diffusion_steps,
        n_sample=1,
        jump_length=args.jump_length,
        jump_n_sample=args.jump_n_sample,
        start_resampling=args.start_resampling,
    )

    logger.log("inpainting...")
    import os
    seed = args.seed*4 + int(os.environ["CUDA_VISIBLE_DEVICES"])
    th.manual_seed(seed)
    #noise = th.zeros(
    # noise = th.ones(
    #     (args.batch_size, args.in_channels, args.image_size, args.image_size),
    #     dtype=th.float32,
    #     device=dist_util.dev()
    # ) * 2
    # noise = th.from_numpy(
    #     np.load('../velocity_module-IS64-NC128-NRB3-DS4000-NScosine-LR1e-4-BS256-sample/fixed_noise_64x1x64x64.npy')
    # ).to(dtype=th.float32, device=dist_util.dev())
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        batch, model_kwargs, mask_gt = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        mask_gt = mask_gt.to(dist_util.dev())

        sample_fn = diffusion.p_sample_loop_inpaint
        sample = sample_fn(
            resampling_indices,
            batch,
            mask_gt,
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            #noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = sample.clamp(-1, 1)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}-seed{args.seed}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("inpainting complete")


def create_argparser():
    defaults = dict(
        dataset_path="",
        dataset_name="",
        mask_gt_name="_mask_gt",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        model_path="",
        jump_length=10,
        jump_n_sample=10,
        start_resampling=100000000,
        seed=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
