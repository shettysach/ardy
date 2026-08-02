# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ardy.model.cfg import AutoLatentClassifierFreeGuidedModel
from ardy.model.diffusion import DDIMSampler, Diffusion
from ardy.model.latent_utils import HybridMotionConverter


def get_three_mask_from_len(history_len, generation_len, future_len, num_frames, device):
    indices = torch.arange(num_frames, device=device)[None, :]  # (1, num_frames)
    history_mask = indices < history_len[:, None]
    generation_start = history_len
    generation_end = generation_start + generation_len
    generation_mask = (indices < generation_end[:, None]) & (indices >= generation_start[:, None])
    future_mask = indices >= generation_end[:, None]
    return history_mask, generation_mask, future_mask


def translate_normalized_root_motion(root_motion, translation, motion_rep):
    """The y coords of the input translation are zero."""
    assert (translation[:, 1] == 0).all(), "the y coords of the input translation are zero"

    root_motion = motion_rep.global_root_stats.unnormalize(root_motion)
    root_motion[:, :, motion_rep.slice_dict["root_pos"]] = (
        root_motion[:, :, motion_rep.slice_dict["root_pos"]] + translation[:, None, :]
    )
    root_motion = motion_rep.global_root_stats.normalize(root_motion)
    return root_motion


class Ardy(nn.Module):
    """Helper class for test time."""

    def __init__(
        self,
        denoiser: nn.Module,
        autoencoder: nn.Module,
        gen_horizon_len: int,
        num_base_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        cfg_type: Optional[str] = "regular",
    ):
        super().__init__()

        if cfg_type is None:
            cfg_type = "nocfg"
        # eval mode
        self.denoiser = AutoLatentClassifierFreeGuidedModel(denoiser.eval(), cfg_type=cfg_type)

        self.motion_rep = denoiser.motion_rep
        self.skeleton = self.motion_rep.skeleton
        self.gen_horizon_len = gen_horizon_len

        self.autoencoder = autoencoder

        self.diffusion = Diffusion(num_base_steps=num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        self.hybrid = HybridMotionConverter.from_model(self)

        self.num_frames_per_token = self.autoencoder.num_frames_per_token

        self.device = device
        self.to(device)

    def train(self, mode: bool):
        self.denoiser.train(mode)
        return self

    def eval(self):
        self.denoiser.eval()
        return self

    def denoising_step(
        self,
        x: torch.Tensor,
        history_len: torch.Tensor,
        generation_len: torch.Tensor,
        future_len: torch.Tensor,
        history_mask: torch.Tensor,
        generation_mask: torch.Tensor,
        future_mask: torch.Tensor,
        history_token_mask: torch.Tensor,
        generation_token_mask: torch.Tensor,
        future_token_mask: torch.Tensor,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        t: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        num_denoising_steps: torch.Tensor,
        cfg_weight: Union[float, Tuple[float, float]],
    ) -> torch.Tensor:
        """Single denoising step.

        Returns:
            torch.Tensor: [B, T, D] noisy motion input to t-1
        """
        # subsample timesteps
        #   NOTE: do this at every step due to ONNX export, i.e. num_samp_stepsmay change dynamically when
        #       running onnx version so need to account for that.
        num_denoising_steps = num_denoising_steps[0]
        use_timesteps, map_tensor = self.diffusion.space_timesteps(num_denoising_steps)
        self.diffusion.calc_diffusion_vars(use_timesteps)

        # first compute initial clean prediction from denoiser
        t_map = map_tensor[t]

        # self.denoiser is the CFG wrapper (PyTorch or TRTCFGDenoiser); both take the
        # text and constraint guidance weights as two tensors. This is the single
        # place the combined cfg_weight (text, constraint) is turned into tensors.
        if isinstance(cfg_weight, (tuple, list)):
            w_text, w_cstr = cfg_weight
        else:
            w_text, w_cstr = cfg_weight, 0.0
        cfg_weight_text = torch.tensor([w_text], device=x.device, dtype=torch.float32)
        cfg_weight_cstr = torch.tensor([w_cstr], device=x.device, dtype=torch.float32)

        with torch.inference_mode():
            token_seq_pred_clean = self.denoiser(
                cfg_weight_text,
                cfg_weight_cstr,
                x,
                history_len,
                generation_len,
                future_len,
                history_mask,
                generation_mask,
                future_mask,
                history_token_mask,
                generation_token_mask,
                future_token_mask,
                text_feat,
                text_pad_mask,
                t_map,
                first_heading_angle,
                motion_mask,
                observed_motion,
            )

        # sampler computes next step noisy motion
        batch_size, num_token_dim = x.shape[0], x.shape[2]
        num_generation_tokens = self.gen_horizon_len // self.num_frames_per_token
        generation_token_t = x[generation_token_mask].reshape(batch_size, num_generation_tokens, num_token_dim)
        generation_token_clean = token_seq_pred_clean[generation_token_mask].reshape(
            batch_size, num_generation_tokens, num_token_dim
        )
        generation_token_tm1 = self.sampler(generation_token_t, generation_token_clean, t)

        # Clone: x is still needed by the caller; the masked write below must
        # not alias it.
        xm1 = x.clone()
        xm1[generation_token_mask] = generation_token_tm1.reshape(-1, num_token_dim)
        return xm1

    def _recenter_history(
        self,
        history_sequence: torch.Tensor,
        center_frame_index: torch.Tensor,
        requantize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recenter the hybrid history around ``center_frame_index``.

        Returns the recentered hybrid history, the center position, and the recentered (explicit)
        root motion.
        """
        root_motion, latent_body_motion = self.hybrid.get_root_and_latent_body_motion_from_hybrid(
            history_sequence,
        )
        new_root_motion, center_pos = self.motion_rep.recenter_root_motion(
            root_motion,
            center_frame_index,
            is_normalized=True,
            to_normalize=True,
            return_center_pos=True,
        )
        # quantize the latent body motion if the autoencoder uses quantization
        if requantize and self.autoencoder.encode_with_quantization:
            latent_body_motion = self.autoencoder.requantize(
                latent_body_motion,
            )
        # combine back to hybrid motion
        history_sequence = self.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(
            new_root_motion,
            latent_body_motion,
        )
        return history_sequence, center_pos, new_root_motion

    def _encode_init_history(
        self,
        init_history_sequence: torch.Tensor,
        batch_size: int,
        crop_history_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an explicit init history into a recentered hybrid sequence.

        Returns the hybrid history sequence, the initial global translation, and the first-frame
        heading angle.
        """
        init_history_len = init_history_sequence.shape[1]
        history_sequence, history_pad_mask = self.hybrid.get_hybrid_motion_from_explicit(
            motion=init_history_sequence,
            motion_len=torch.ones(batch_size, device=self.device, dtype=torch.long) * init_history_len,
            motion_pad_mask=torch.ones(batch_size, init_history_len, device=self.device, dtype=torch.bool),
        )
        # transl_center_mode == "last_history":
        center_frame_index = torch.ones(batch_size, device=self.device, dtype=torch.long) * (
            init_history_len - 1
        )  # [B,]

        history_sequence, center_pos, _ = self._recenter_history(history_sequence, center_frame_index, requantize=False)
        # update global translation
        global_transl = center_pos.clone()

        # set first frame heading angle
        heading_angles = self.motion_rep.get_root_heading_angle(self.motion_rep.unnormalize(init_history_sequence))
        first_heading_angle = heading_angles[:, 0]
        return history_sequence, global_transl, first_heading_angle

    def _generate_window(
        self,
        history_sequence: Optional[torch.Tensor],
        global_transl: torch.Tensor,
        history_start_frame: int,
        history_end_frame: int,
        total_frames: int,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: Optional[torch.Tensor],
        observed_motion: Optional[torch.Tensor],
        num_denoising_steps: torch.Tensor,
        cfg_weight: Union[float, Tuple[float, float]],
        indices: List[int],
        progress_bar=tqdm,
    ) -> torch.Tensor:
        """Generate a single window of ``gen_horizon_len`` frames.

        Builds the history/generation/future masks and conditioning, denoises a block of pure noise,
        and appends the newly generated tokens (in hybrid latent form) to ``history_sequence``.
        Returns the updated history.
        """
        device = self.device
        batch_size = text_feat.shape[0]
        num_frames_per_token = self.num_frames_per_token
        latent_embedding_dim = self.denoiser.latent_embedding_dim
        nframe_root_dim = self.denoiser.nframe_root_dim
        gen_horizon_len = self.gen_horizon_len
        num_generation_tokens = gen_horizon_len // num_frames_per_token
        generation_len = torch.ones(batch_size, device=device, dtype=torch.long) * gen_horizon_len

        history_token_end = (history_end_frame - history_start_frame) // num_frames_per_token
        generation_token_end = history_token_end + num_generation_tokens
        history_len = torch.ones(batch_size, device=device, dtype=torch.long) * history_token_end * num_frames_per_token
        future_len = torch.ones(batch_size, device=device, dtype=torch.long) * (
            total_frames - gen_horizon_len - history_end_frame
        )

        history_mask, generation_mask, future_mask = get_three_mask_from_len(
            history_len,
            generation_len,
            future_len,
            total_frames - history_start_frame,
            device,
        )
        #  update observed motion roots with global translation
        if motion_mask is not None:
            cur_motion_mask = motion_mask[:, history_start_frame:] * (
                ~history_mask[:, :, None]
            )  # only allow motion mask for non-history frames
            cur_observed_root_motion = self.motion_rep.extract_root(observed_motion[:, history_start_frame:])
            cur_observed_body_motion = self.motion_rep.extract_body(observed_motion[:, history_start_frame:])

            translated_observed_root_motion = translate_normalized_root_motion(
                cur_observed_root_motion, -global_transl, self.motion_rep
            )
            cur_observed_motion = self.motion_rep.concat_root_body(
                translated_observed_root_motion, cur_observed_body_motion
            )

            cur_observed_motion = (
                cur_observed_motion * cur_motion_mask
            )  # mask out the unobserved frames roots/joints that are non-zero due to translation
        else:
            cur_motion_mask = None
            cur_observed_motion = None

        history_token_mask, generation_token_mask, future_token_mask = self.hybrid.convert_frame_mask_to_token_mask(
            history_mask,
            generation_mask,
            future_mask,
            cur_motion_mask,
        )

        # init pure noise x_T and x
        shape = (
            batch_size,
            num_generation_tokens,
            nframe_root_dim + latent_embedding_dim,
        )
        x_t = torch.randn(shape, device=device)
        x = torch.zeros(
            batch_size,
            (total_frames - history_start_frame) // num_frames_per_token,
            nframe_root_dim + latent_embedding_dim,
            device=device,
        )
        if history_token_end > 0:
            x[:, :history_token_end] = history_sequence[:, -history_token_end:]
        x[:, history_token_end:generation_token_end] = x_t

        for i in progress_bar(indices):
            t = torch.tensor([i] * x_t.size(0), device=device)
            with torch.no_grad():
                x = self.denoising_step(
                    x,
                    history_len,
                    generation_len,
                    future_len,
                    history_mask,
                    generation_mask,
                    future_mask,
                    history_token_mask,
                    generation_token_mask,
                    future_token_mask,
                    text_feat,
                    text_pad_mask,
                    t,
                    first_heading_angle,
                    cur_motion_mask,
                    cur_observed_motion,
                    num_denoising_steps,
                    cfg_weight,
                )

        #  update the full history sequence
        if history_sequence is None:
            history_sequence = x[:, :generation_token_end]
        else:
            history_sequence = torch.cat(
                [
                    history_sequence,
                    x[:, history_token_end:generation_token_end],
                ],
                dim=1,
            )
        return history_sequence

    def __call__(
        self,
        num_frames: int,
        num_denoising_steps: int,
        pad_mask: torch.Tensor,
        first_heading_angle: Optional[torch.Tensor],
        motion_mask: torch.Tensor,
        observed_motion: torch.Tensor,
        *,
        text_feat: torch.Tensor,
        text_pad_mask: torch.Tensor,
        cfg_weight: Optional[float] = 2.0,
        progress_bar=tqdm,
        crop_history_length: Optional[int] = None,
        init_history_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample full denoising loop from checkpoint-specific text features."""
        batch_size = text_feat.shape[0]
        num_frames_per_token = self.num_frames_per_token
        motion_pad_mask = pad_mask
        motion_len = motion_pad_mask.sum(dim=-1)  # [batch_size], exact length of the motion output
        motion_len_pad = (
            torch.ceil(motion_len / num_frames_per_token).int() * num_frames_per_token
        )  # padded to multiple of num_frames_per_token, used for decoding

        gen_horizon_len = self.gen_horizon_len
        init_history_len = 0 if init_history_sequence is None else init_history_sequence.shape[1]
        num_autoregressive_steps = int(np.ceil((num_frames - init_history_len) / gen_horizon_len))
        num_gen_frames = num_autoregressive_steps * gen_horizon_len + init_history_len
        motion_pad_mask = torch.arange(num_gen_frames, device=self.device) < motion_len_pad[:, None]
        if num_frames < num_gen_frames:
            pad_num_frames = num_gen_frames - num_frames
            if motion_mask is not None:
                motion_mask = torch.cat(
                    [
                        motion_mask,
                        torch.zeros(
                            batch_size,
                            pad_num_frames,
                            motion_mask.shape[-1],
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )
            if observed_motion is not None:
                observed_motion = torch.cat(
                    [
                        observed_motion,
                        torch.zeros(
                            batch_size,
                            pad_num_frames,
                            observed_motion.shape[-1],
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )

        generation_len = torch.ones(batch_size, device=self.device, dtype=torch.long) * gen_horizon_len
        assert crop_history_length is None or crop_history_length % num_frames_per_token == 0, (
            "crop_history_length should be a multiple of num_frames_per_token"
        )

        if init_history_sequence is not None:
            assert first_heading_angle is None
            assert crop_history_length is None, (
                "not supporting combination of crop_history_length and init_history_sequence"
            )
            history_sequence, global_transl, first_heading_angle = self._encode_init_history(
                init_history_sequence, batch_size, crop_history_length
            )
        else:
            history_sequence = None
            global_transl = torch.zeros(
                (batch_size, self.motion_rep.nfeats_dict["root_pos"]),
                device=self.device,
            )

        # sample loop
        indices = list(range(num_denoising_steps))[::-1]
        num_denoising_steps = torch.tensor(
            [num_denoising_steps], device=self.device
        )  # this and t need to be tensor for onnx export
        # init diffusion with correct num steps before looping
        use_timesteps = self.diffusion.space_timesteps(num_denoising_steps[0])[0]
        self.diffusion.calc_diffusion_vars(use_timesteps)

        for auto_step in range(num_autoregressive_steps):
            history_end_frame = auto_step * gen_horizon_len + init_history_len
            history_start_frame = (
                0 if crop_history_length is None else max(0, history_end_frame - crop_history_length)
            )  # set history start frame to 0 if no crop, otherwise set to the first frame of the cropped history, the start frame should be non-negative

            history_sequence = self._generate_window(
                history_sequence,
                global_transl,
                history_start_frame,
                history_end_frame,
                num_gen_frames,
                text_feat,
                text_pad_mask,
                first_heading_angle,
                motion_mask,
                observed_motion,
                num_denoising_steps,
                cfg_weight,
                indices,
                progress_bar=progress_bar,
            )

            #  recenter the history sequence as specified
            # transl_center_mode == "last_history":
            center_frame_index = (history_end_frame + generation_len - 1).clamp(min=0)  # [B,]
            history_sequence, center_pos, new_root_motion = self._recenter_history(
                history_sequence, center_frame_index, requantize=True
            )

            # update global translation
            global_transl = global_transl + center_pos

            # update first heading angle if history cropping is applied
            if crop_history_length is not None:
                # new_root_motion holds root features only — unnormalize with the
                # root-slice stats, not the full-feature stats.
                heading_angles = self.motion_rep.get_root_heading_angle(
                    self.motion_rep.global_root_stats.unnormalize(new_root_motion),
                )
                full_history_len = new_root_motion.shape[1]
                cropped_history_start_frame = max(0, full_history_len - crop_history_length)
                first_heading_angle = heading_angles[:, cropped_history_start_frame]

        root_motion, latent_body_motion = self.hybrid.get_root_and_latent_body_motion_from_hybrid(
            history_sequence,
        )
        new_root_motion = translate_normalized_root_motion(root_motion, global_transl, self.motion_rep)
        history_sequence = self.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(
            new_root_motion,
            latent_body_motion,
        )
        if crop_history_length is not None:
            motion_output = self.hybrid.get_explicit_motion_from_hybrid_autoregressive(
                history_sequence,
                motion_pad_mask,
                motion_len_pad,
                motion_mask=motion_mask,
                crop_history_length=crop_history_length,
            )
        else:
            motion_output = self.hybrid.get_explicit_motion_from_hybrid(
                history_sequence,
                motion_pad_mask,
                motion_len_pad,
                motion_mask=motion_mask,
            )
        motion_pred = motion_output[:, :num_frames]

        return motion_pred
