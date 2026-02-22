"""
Example for exporting LiteTracker sub-components to ONNX format and using with a wrapper.
Please note that there might be discrepancies between the PyTorch and ONNX wrapper. Please verify carefully for your use case.

Usage:
    # Export only:
    uv run onnx_example.py --checkpoint path/to/weights.pth --output_dir ./onnx_export

    # Export + verify against PyTorch:
    uv run onnx_example.py --checkpoint path/to/weights.pth --output_dir ./onnx_export --verify
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src.lite_tracker import LiteTracker, posenc
from src.model_utils import sample_features5d, bilinear_sampler


def load_model(checkpoint_path: str) -> LiteTracker:
    """
    Load a LiteTracker model from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file (.pth).

    Returns:
        LiteTracker: The loaded model in evaluation mode.
    """
    model = LiteTracker()
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_fnet(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export the feature encoder (BasicEncoder) to ONNX.

    Creates a dummy input matching the model's expected resolution and exports
    the feature extraction CNN with dynamic batch size support.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX file.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        str: Path to the exported ONNX file.
    """
    output_path = os.path.join(output_dir, "fnet.onnx")
    H, W = model.model_resolution
    dummy_input = torch.randn(1, 3, H, W)

    torch.onnx.export(
        model.fnet,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["frame"],
        output_names=["features"],
        dynamic_axes={
            "frame": {0: "batch"},
            "features": {0: "batch"},
        },
    )
    print(f"  Exported fnet -> {output_path}")
    return output_path


def export_corr_mlp(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export the correlation MLP to ONNX.

    The MLP takes a flattened correlation volume (r^2 * r^2 values, where r = 2*corr_radius+1)
    and produces a compressed correlation embedding.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX file.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        str: Path to the exported ONNX file.
    """
    output_path = os.path.join(output_dir, "corr_mlp.onnx")
    r = 2 * model.corr_radius + 1
    input_size = r * r * r * r  # 49 * 49 = 2401
    dummy_input = torch.randn(1, input_size)

    torch.onnx.export(
        model.corr_mlp,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["corr_volume"],
        output_names=["corr_emb"],
        dynamic_axes={
            "corr_volume": {0: "num_points"},
            "corr_emb": {0: "num_points"},
        },
    )
    print(f"  Exported corr_mlp -> {output_path}")
    return output_path


def export_updateformer(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export the EfficientUpdateFormer transformer to ONNX.

    The transformer takes per-point, per-frame tokens and an attention mask, and
    outputs coordinate/visibility/confidence deltas.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX file.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        str: Path to the exported ONNX file.
    """
    output_path = os.path.join(output_dir, "updateformer.onnx")

    B, N, T = 1, 100, 4
    dummy_input = torch.randn(B, N, T, model.input_dim)
    dummy_mask = torch.ones(B, T, N, dtype=torch.bool)

    torch.onnx.export(
        model.updateformer,
        (dummy_input, dummy_mask),
        output_path,
        opset_version=opset_version,
        input_names=["input_tensor", "mask"],
        output_names=["delta"],
        dynamic_axes={
            "input_tensor": {0: "batch", 1: "num_points", 2: "window_size"},
            "mask": {0: "batch", 1: "window_size", 2: "num_points"},
            "delta": {0: "batch", 1: "num_points", 2: "window_size"},
        },
    )
    print(f"  Exported updateformer -> {output_path}")
    return output_path


def export_all(model: LiteTracker, output_dir: str, opset_version: int = 17):
    """
    Export all LiteTracker sub-components to ONNX.

    Exports fnet, corr_mlp, and updateformer as separate ONNX files into the
    specified output directory.

    Args:
        model (LiteTracker): The source LiteTracker model.
        output_dir (str): Directory to save the exported ONNX files.
        opset_version (int): ONNX opset version to use. Defaults to 17.

    Returns:
        dict: Mapping from component name to output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Exporting LiteTracker sub-components to ONNX...")
    paths = {
        "fnet": export_fnet(model, output_dir, opset_version),
        "corr_mlp": export_corr_mlp(model, output_dir, opset_version),
        "updateformer": export_updateformer(model, output_dir, opset_version),
    }
    print("Export complete!")
    return paths


# ---------------------------------------------------------------------------
# ONNX-based LiteTracker wrapper
# ---------------------------------------------------------------------------

class OnnxLiteTracker:
    """
    A drop-in replacement for LiteTracker that uses ONNX Runtime for inference.

    This class replicates the full stateful tracking pipeline of LiteTracker.forward(),
    but uses ONNX models for the three neural network forward passes (fnet, corr_mlp,
    updateformer). All stateful logic (buffers, flow estimation, etc.) is handled in
    numpy/torch.

    Usage:
        tracker = OnnxLiteTracker("./onnx_export")
        for frame in video_frames:
            coords, vis, conf = tracker(frame_tensor, queries_tensor)
    """

    def __init__(
        self,
        onnx_dir: str,
        window_len: int = 16,
        stride: int = 4,
        corr_radius: int = 3,
        corr_levels: int = 4,
        model_resolution: tuple = (384, 512),
        iters: int = 1,
        input_dim: int = 1110,
        latent_dim: int = 128,
        providers: list = None,
    ):
        """
        Initialize the ONNX-based LiteTracker.

        Args:
            onnx_dir (str): Directory containing the exported ONNX models.
            window_len (int): Max length of the temporal window for tracking.
            stride (int): Stride scale for feature extraction.
            corr_radius (int): Radius for correlation computation.
            corr_levels (int): Number of correlation pyramid levels.
            model_resolution (tuple): Resolution to which input frames are resized; (H, W).
            iters (int): Number of update iterations per frame.
            input_dim (int): Dimension of the transformer input tokens.
            latent_dim (int): Dimension of the feature encoder output channels.
            providers (list): ONNX Runtime execution providers. Defaults to CUDA + CPU.
        """
        import onnxruntime as ort

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.model_resolution = model_resolution
        self.iters = iters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inv_sigmoid_true_val = 4.6

        # Load ONNX sessions
        self.fnet_session = ort.InferenceSession(
            os.path.join(onnx_dir, "fnet.onnx"), providers=providers
        )
        self.corr_mlp_session = ort.InferenceSession(
            os.path.join(onnx_dir, "corr_mlp.onnx"), providers=providers
        )
        self.updateformer_session = ort.InferenceSession(
            os.path.join(onnx_dir, "updateformer.onnx"), providers=providers
        )

        # Precompute time embedding (same as LiteTracker.__init__)
        from src.model_utils import get_1d_sincos_pos_embed_from_grid
        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(1, window_len, 1)
        self.time_emb = get_1d_sincos_pos_embed_from_grid(input_dim, time_grid[0])

        self.reset()

    def reset(self):
        """
        Reset all internal state, buffers, and caches for a new video sequence.

        Call this method before processing a new video to clear all temporal
        state from the previous sequence.
        """
        self.online_ind = 0
        self.ema_flow_buffer = None
        self.corr_embs_buffer = None
        self.coords_buffer = None
        self.vis_buffer = None
        self.conf_buffer = None
        self.track_feat_cache = [None] * self.corr_levels

    def _run_fnet(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Run the feature encoder via ONNX Runtime.

        Args:
            frame (torch.Tensor): Normalized input frame of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Feature map of shape [B, latent_dim, H//stride, W//stride].
        """
        result = self.fnet_session.run(
            None, {"frame": frame.numpy()}
        )
        return torch.from_numpy(result[0])

    def _run_corr_mlp(self, corr_volume: torch.Tensor) -> torch.Tensor:
        """
        Run the correlation MLP via ONNX Runtime.

        Args:
            corr_volume (torch.Tensor): Flattened correlation volume of shape [N_points, r^4].

        Returns:
            torch.Tensor: Correlation embeddings of shape [N_points, 256].
        """
        result = self.corr_mlp_session.run(
            None, {"corr_volume": corr_volume.numpy()}
        )
        return torch.from_numpy(result[0])

    def _run_updateformer(
        self, input_tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the update transformer via ONNX Runtime.

        Args:
            input_tensor (torch.Tensor): Transformer input of shape [B, N, T, input_dim].
            mask (torch.Tensor): Attention mask of shape [B, T, N] (bool).

        Returns:
            torch.Tensor: Delta predictions of shape [B, N, T, 4] (dx, dy, dvis, dconf).
        """
        result = self.updateformer_session.run(
            None,
            {
                "input_tensor": input_tensor.numpy(),
                "mask": mask.numpy(),
            },
        )
        return torch.from_numpy(result[0])

    def _interpolate_time_embed(self, t: int) -> torch.Tensor:
        """
        Interpolate the temporal positional embedding to match the current window size.

        Uses linear interpolation when the current window size differs from the
        precomputed embedding length.

        Args:
            t (int): Target temporal length.

        Returns:
            torch.Tensor: Interpolated time embedding of shape [1, t, input_dim].
        """
        T = self.time_emb.shape[1]
        if t == T:
            return self.time_emb
        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb

    def _get_support_points(self, coords, r, reshape_back=True):
        """
        Generate a grid of support points around each coordinate for local feature sampling.

        Mirrors LiteTracker.get_support_points() exactly.

        Args:
            coords (torch.Tensor): Input coordinates of shape [B, T, N, 3].
            r (int): Radius for the support grid.
            reshape_back (bool): Whether to reshape the output for downstream use.

        Returns:
            torch.Tensor: Support points for each coordinate.
        """
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)
        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)
        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack((zgrid, xgrid, ygrid), dim=-1)
        delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl
        if reshape_back:
            third_dim = int((2 * r + 1) ** 2)
            return coords_lvl.reshape(B, N, third_dim, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl

    def _get_track_feat(self, fmaps, queried_coords, support_radius=0):
        """
        Extract track features at the queried coordinates and their support points.

        Mirrors LiteTracker.get_track_feat() exactly.

        Args:
            fmaps (torch.Tensor): Feature maps of shape [B, T, C, H, W].
            queried_coords (torch.Tensor): Coordinates to sample features from.
            support_radius (int): Radius for support points.

        Returns:
            tuple: (central track features, all support track features).
        """
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1][:, None]), queried_coords[:, None]],
            dim=-1,
        )
        support_points = self._get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )

    def _get_correlation_feat(self, fmaps, queried_coords):
        """
        Compute correlation features for the queried coordinates using bilinear sampling.

        Mirrors LiteTracker.get_correlation_feat() exactly.

        Args:
            fmaps (torch.Tensor): Feature maps of shape [B, T, D, H, W].
            queried_coords (torch.Tensor): Coordinates to sample features from.

        Returns:
            torch.Tensor: Correlation features of shape [B, T, N, r, r, D].
        """
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self._get_support_points(sample_coords, r, reshape_back=False)
        correlation_feat = bilinear_sampler(
            fmaps.reshape(B * T, D, 1, H_, W_), support_points
        )
        return correlation_feat.view(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )

    def __call__(self, frame: torch.Tensor, queries: torch.Tensor):
        """
        Predict tracks for the given frame and queries.

        This replicates LiteTracker.forward() but uses ONNX models for neural network
        forward passes. All stateful orchestration (buffer management, flow momentum,
        coordinate initialization) is handled identically to the PyTorch version.

        Args:
            frame (torch.Tensor): Input frame of shape [B, C, H, W] (float32, 0-255 range).
            queries (torch.Tensor): Point queries of shape [B, N, 3]; first channel is the
                frame index, the rest are (x, y) coordinates.

        Returns:
            tuple:
                coords (torch.Tensor): Predicted coordinates [B, 1, N, 2].
                vis (torch.Tensor): Predicted visibility mask [B, 1, N].
                conf (torch.Tensor): Predicted confidence [B, 1, N].
        """
        original_shape = frame.shape
        frame = F.interpolate(
            frame, size=self.model_resolution, mode="bilinear", align_corners=True
        )
        queries_scaled = queries.clone()
        queries_scaled[:, :, 1] *= (self.model_resolution[1] - 1) / (
            original_shape[3] - 1
        )  # W
        queries_scaled[:, :, 2] *= (self.model_resolution[0] - 1) / (
            original_shape[2] - 1
        )  # H

        B, C, H, W = frame.shape
        B, N, __ = queries_scaled.shape
        frame = 2 * (frame / 255.0) - 1.0

        T = 1

        queried_frames = queries_scaled[:, :, 0].long()

        # Downscale the query coords for the smaller size feat maps
        queried_coords = queries_scaled[..., 1:3].clone()
        queried_coords = queried_coords / self.stride

        # Run feature encoder via ONNX
        fmaps = self._run_fnet(frame.float())
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )

        # Build feature pyramid
        fmaps_pyramid = [fmaps]
        track_feat_support_pyramid = []
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        is_track_initialized_now = (queried_frames == self.online_ind)[
            :, None, :, None
        ]  # B 1 N 1

        for i in range(self.corr_levels):
            if self.online_ind == 0:
                _, track_feat_support = self._get_track_feat(
                    fmaps_pyramid[i],
                    queried_coords / 2**i,
                    support_radius=self.corr_radius,
                )
                self.track_feat_cache[i] = torch.zeros_like(track_feat_support)
                self.track_feat_cache[i] += (
                    track_feat_support * is_track_initialized_now
                )
            else:
                if is_track_initialized_now.any():
                    _, track_feat_support = self._get_track_feat(
                        fmaps_pyramid[i],
                        queried_coords / 2**i,
                        support_radius=self.corr_radius,
                    )
                    self.track_feat_cache[i] += (
                        track_feat_support * is_track_initialized_now
                    )

            track_feat_support_pyramid.append(self.track_feat_cache[i].unsqueeze(1))

        # Initialize vis and conf for the current frame with zeros
        vis_init = torch.zeros((B, T, N, 1)).float()
        conf_init = torch.zeros((B, T, N, 1)).float()
        coords_init = queried_coords.reshape(B, T, N, 2).float()

        vis_init = torch.where(
            is_track_initialized_now.expand_as(vis_init),
            self.inv_sigmoid_true_val,
            vis_init,
        )
        conf_init = torch.where(
            is_track_initialized_now.expand_as(conf_init),
            self.inv_sigmoid_true_val,
            conf_init,
        )

        if self.online_ind == 0:
            self.ema_flow_buffer = torch.zeros_like(coords_init)

        # Handle tracks that are initialized in the previous frame.
        is_track_previsouly_initialized = (queried_frames < self.online_ind)[
            :, None, :, None
        ]  # B 1 N 1
        if self.online_ind > 0:
            vis_init = torch.where(
                is_track_previsouly_initialized.expand_as(vis_init),
                self.vis_buffer[:, -1],
                vis_init,
            )
            conf_init = torch.where(
                is_track_previsouly_initialized.expand_as(conf_init),
                self.conf_buffer[:, -1],
                conf_init,
            )
            # If there is only one frame processed so far, we initialize the coordinates
            # with the previous frame's coordinates
            if self.online_ind == 1:
                coords_init = torch.where(
                    is_track_previsouly_initialized.expand_as(coords_init),
                    self.coords_buffer[:, -1],
                    coords_init,
                )
            # If there is more, we use the exponential moving average of the flow
            else:
                last_flow = self.coords_buffer[:, -1] - self.coords_buffer[:, -2]
                cached_flow = self.ema_flow_buffer
                alpha = 0.8
                accumulated_flow = alpha * last_flow + (1 - alpha) * cached_flow
                self.ema_flow_buffer = accumulated_flow
                coords_init = torch.where(
                    is_track_previsouly_initialized.expand_as(coords_init),
                    self.coords_buffer[:, -1] + accumulated_flow,
                    coords_init,
                )

        # forward_window logic
        coords, viss, confs = self._forward_window(
            fmaps_pyramid=fmaps_pyramid,
            coords=coords_init,
            track_feat_support_pyramid=track_feat_support_pyramid,
            queried_frames=queried_frames,
            vis=vis_init,
            conf=conf_init,
            iters=self.iters,
            is_track_previsouly_initialized=is_track_previsouly_initialized,
        )

        coords[:, :, :, 0] *= (original_shape[3] - 1) / (
            self.model_resolution[1] - 1
        )  # W
        coords[:, :, :, 1] *= (original_shape[2] - 1) / (
            self.model_resolution[0] - 1
        )  # H

        viss = torch.sigmoid(viss)
        confs = torch.sigmoid(confs)

        viss = viss * confs
        thr = 0.6
        viss = viss > thr

        self.online_ind += 1
        return coords, viss, confs

    def _forward_window(
        self,
        fmaps_pyramid,
        coords,
        track_feat_support_pyramid,
        queried_frames,
        vis,
        conf,
        is_track_previsouly_initialized,
        iters=4,
    ):
        """
        Run the tracking update for a window of frames using ONNX sub-models.

        Replicates LiteTracker.forward_window() exactly, substituting PyTorch
        model calls with ONNX Runtime inference for corr_mlp and updateformer.

        Args:
            fmaps_pyramid (list[torch.Tensor]): Feature maps at different scales [B, T, C, H, W].
            coords (torch.Tensor): Track coordinates for each frame [B, T, N, 2].
            track_feat_support_pyramid (list[torch.Tensor]): Template features at different
                scales [B, 1, r^2, N, C].
            queried_frames (torch.Tensor): Frame indices of the queries [B, N].
            vis (torch.Tensor): Visibility logits for tracks [B, T, N, 1].
            conf (torch.Tensor): Confidence logits for tracks [B, T, N, 1].
            is_track_previsouly_initialized (torch.Tensor): Mask for tracks initialized in
                previous frames [B, 1, N, 1].
            iters (int): Number of update iterations.

        Returns:
            tuple: (coords, vis, conf) for the current window.
        """
        device = fmaps_pyramid[0].device
        B = fmaps_pyramid[0].shape[0]
        N = coords.shape[2]
        r = 2 * self.corr_radius + 1

        num_new_frames = 1
        num_prev_frames = min(self.online_ind, (self.window_len - 1))
        current_window_size = (
            num_prev_frames + num_new_frames
        )  # total number of frames in the current window

        # Compute the frame indices for the current window
        left_ind = max(0, self.online_ind - self.window_len + 1)  # inclusive
        right_ind = self.online_ind + 1  # not inclusive
        frame_indices = (
            torch.arange(left_ind, right_ind, device=device)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(B, -1, N)
        )  # shape [B, T, N]

        # `attention_mask` is a boolean mask: True if the track is initialized at or before
        # this frame
        attention_mask = (
            queried_frames.unsqueeze(1).expand(B, -1, N) <= frame_indices
        )  # B T N

        corr_embs = torch.empty(1, device=device)
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            coords_init = coords.view(-1, N, 2)
            # Extract correlation embeddings from the new frames
            corr_embs_list = []
            for i in range(self.corr_levels):
                corr_feat = self._get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                # Run corr_mlp via ONNX
                corr_flat = corr_volume.reshape(
                    B * num_new_frames * N, r * r * r * r
                ).float()
                corr_embs_list.append(self._run_corr_mlp(corr_flat))
            corr_embs = torch.cat(corr_embs_list, dim=-1).view(
                B, num_new_frames, N, -1
            )

            # If it is the first time step, we skip the transformer as there is nothing
            # to compute, simply use the computed `corr_embs` as well as the initial
            # values of `coords`, `vis` and `conf` to initialize the buffers.
            if self.online_ind == 0:
                break

            prev_coords = self.coords_buffer.detach()
            prev_vis = self.vis_buffer.detach()
            prev_conf = self.conf_buffer.detach()
            prev_corr_embs = self.corr_embs_buffer.detach()

            current_window_coords = torch.cat([prev_coords, coords], dim=1)
            current_window_vis = torch.cat([prev_vis, vis], dim=1)
            current_window_conf = torch.cat([prev_conf, conf], dim=1)
            current_window_corr_embs = torch.cat([prev_corr_embs, corr_embs], dim=1)

            transformer_input = [
                current_window_vis,
                current_window_conf,
                current_window_corr_embs,
            ]
            rel_coords_forward = (
                current_window_coords[:, :-1] - current_window_coords[:, 1:]
            )
            rel_coords_backward = (
                current_window_coords[:, 1:] - current_window_coords[:, :-1]
            )
            rel_coords_forward = F.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
            rel_coords_backward = F.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))

            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]], device=device
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale
            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )
            transformer_input.append(rel_pos_emb_input)
            x = (
                (torch.cat(transformer_input, dim=-1))
                .permute(0, 2, 1, 3)
                .reshape(B * N, current_window_size, -1)
            )
            x = x + self._interpolate_time_embed(current_window_size)
            x = x.view(B, N, current_window_size, -1)

            # Run updateformer via ONNX
            delta = self._run_updateformer(x.float(), attention_mask)
            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

            # Update the values of the current frame only for the points that are
            # initialized before this frame.
            vis[is_track_previsouly_initialized] = (
                vis[is_track_previsouly_initialized]
                + delta_vis[:, -num_new_frames:][is_track_previsouly_initialized]
            )
            conf[is_track_previsouly_initialized] = (
                conf[is_track_previsouly_initialized]
                + delta_conf[:, -num_new_frames:][is_track_previsouly_initialized]
            )
            coords[is_track_previsouly_initialized.expand_as(coords)] = (
                coords[is_track_previsouly_initialized.expand_as(coords)]
                + delta_coords[:, -num_new_frames:][
                    is_track_previsouly_initialized.expand_as(coords)
                ]
            )

        # Update buffers
        if self.online_ind == 0:
            self.coords_buffer = coords
            self.vis_buffer = vis
            self.conf_buffer = conf
            self.corr_embs_buffer = corr_embs
        else:
            self.coords_buffer = torch.cat([self.coords_buffer, coords], dim=1)
            self.vis_buffer = torch.cat([self.vis_buffer, vis], dim=1)
            self.conf_buffer = torch.cat([self.conf_buffer, conf], dim=1)
            self.corr_embs_buffer = torch.cat(
                [self.corr_embs_buffer, corr_embs], dim=1
            )

        if current_window_size == self.window_len:
            self.coords_buffer = self.coords_buffer[:, 1:]
            self.vis_buffer = self.vis_buffer[:, 1:]
            self.conf_buffer = self.conf_buffer[:, 1:]
            self.corr_embs_buffer = self.corr_embs_buffer[:, 1:]

        coords = coords[..., :2] * float(self.stride)
        vis = vis[..., 0]
        conf = conf[..., 0]
        return coords, vis, conf


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_export(checkpoint_path: str, onnx_dir: str, num_frames: int = 5):
    """
    Verify ONNX export by comparing outputs against PyTorch model.

    Runs both models on identical synthetic inputs and checks numerical agreement.
    Uses a tolerance of 1e-2 for coordinate and confidence comparisons to account
    for floating-point differences between PyTorch and ONNX Runtime backends.

    Args:
        checkpoint_path (str): Path to the LiteTracker checkpoint (.pth).
        onnx_dir (str): Directory containing the exported ONNX models.
        num_frames (int): Number of synthetic frames to test. Defaults to 5.

    Returns:
        bool: True if all frames pass verification, False otherwise.
    """
    print("\nVerifying ONNX export against PyTorch...")

    # Load PyTorch model
    pt_model = load_model(checkpoint_path)
    pt_model.eval()

    # Create ONNX tracker
    onnx_tracker = OnnxLiteTracker(onnx_dir, providers=["CPUExecutionProvider"])

    # Create synthetic inputs
    B, C, H, W = 1, 3, 480, 640
    N = 10
    torch.manual_seed(42)
    frames = [
        torch.randint(0, 255, (B, C, H, W), dtype=torch.float32)
        for _ in range(num_frames)
    ]
    queries = torch.cat(
        [
            torch.zeros(B, N, 1),  # all queries start at frame 0
            torch.rand(B, N, 1) * (W - 1),
            torch.rand(B, N, 1) * (H - 1),
        ],
        dim=2,
    )

    # Run both models
    pt_model.reset()
    onnx_tracker.reset()

    max_coord_diff = 0.0
    max_conf_diff = 0.0
    all_passed = True

    for i, frame in enumerate(frames):
        with torch.no_grad():
            pt_coords, pt_vis, pt_conf = pt_model(frame, queries)

        onnx_coords, onnx_vis, onnx_conf = onnx_tracker(frame, queries)

        coord_diff = (pt_coords.float() - onnx_coords.float()).abs().max().item()
        conf_diff = (pt_conf.float() - onnx_conf.float()).abs().max().item()
        vis_match = (pt_vis == onnx_vis).all().item()

        max_coord_diff = max(max_coord_diff, coord_diff)
        max_conf_diff = max(max_conf_diff, conf_diff)

        # Use a reasonable tolerance for float32 comparisons
        coord_ok = coord_diff < 1e-2
        conf_ok = conf_diff < 1e-2

        status = "PASS" if (coord_ok and conf_ok) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(
            f"  Frame {i}: {status} | "
            f"coord_diff={coord_diff:.6f}, conf_diff={conf_diff:.6f}, vis_match={vis_match}"
        )

    print(f"\n  Max coord diff: {max_coord_diff:.6f}")
    print(f"  Max conf diff:  {max_conf_diff:.6f}")

    if all_passed:
        print("\n  ✓ All frames passed verification!")
    else:
        print(
            "\n  ✗ Some frames exceeded tolerance. This may be due to floating-point "
            "differences between PyTorch and ONNX Runtime. Consider increasing tolerance "
            "or ensuring consistent dtypes."
        )

    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export LiteTracker sub-components to ONNX format."
    )
    parser.add_argument(
        "-w", "--checkpoint",
        required=True,
        help="Path to LiteTracker checkpoint (.pth)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="./onnx_export",
        help="Output directory for ONNX models (default: ./onnx_export)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification after export (compares PyTorch vs ONNX outputs)",
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    export_all(model, args.output_dir, opset_version=args.opset)

    if args.verify:
        verify_export(args.checkpoint, args.output_dir)
