import torch
import torch.nn.functional as F
import numpy as np
import math

import sys
sys.path.append(".")
from threestudio.utils.typing import *
from threestudio.utils.ops import (
    get_ray_directions,
)


class DiagBatchLoader:
    def __init__(self, split: str="train", height=832, width=480):
        self.n_view = 1 # only one view for diag
        self.total_frame = 81
        self.default_elevation_deg = 10.0
        self.default_fovy_deg = 40.0
        self.height = height
        self.width = width
    
    def get_cam_params(self, default_azimuth_deg, default_camera_distance):
        azimuth_deg = torch.FloatTensor(
            np.linspace(default_azimuth_deg[0], default_azimuth_deg[1], self.total_frame)
        )
        camera_distance = torch.FloatTensor([default_camera_distance] * self.n_view * self.total_frame) \
            if isinstance(default_camera_distance, float) else torch.FloatTensor(self.default_camera_distance)
            
        elevation_deg = torch.FloatTensor([self.default_elevation_deg]*self.n_view*self.total_frame) \
            if isinstance(self.default_elevation_deg, float) else torch.FloatTensor(self.default_elevation_deg)

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )
        
        # prepare all camera position HERE

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        self.c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        self.c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.default_fovy_deg] * self.n_view * self.total_frame)) \
            if isinstance(self.default_fovy_deg, float) else torch.deg2rad(torch.FloatTensor(self.default_fovy_deg))

        self.heights: List[int] = (
            [self.height] if isinstance(self.height, int) else self.height
        )
        self.widths: List[int] = (
            [self.width] if isinstance(self.width, int) else self.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.prev_height = self.height
        self.timestamps = torch.linspace(-1, 1, steps=self.total_frame).unsqueeze(-1).repeat(1, self.n_view).permute(1, 0).reshape(-1, 1) 
    
    def get_batch_init(self,t) -> Dict[str, Any]:
        batch = self.get_batch(0)
        n = batch['timestamps'].shape[0]
        batch['timestamps'] = batch['timestamps'][t,:].repeat(n, 1) # bullet-time
        return batch
    
    def get_batch(self, batch_idx) -> Dict[str, Any]:
        # max(batch_idx) = 9
        lower_bound = -30 # + batch_idx * 180 / 10 # -30 to 150 每次转18度
        default_azimuth_deg = [lower_bound, lower_bound + 60] # 
        # default_azimuth_deg = [-1, 1]
        raidus = 4.0
        self.get_cam_params(default_azimuth_deg, raidus)
        batch = {
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "height": self.height,
            "width": self.width,
            "c2w": self.c2w4x4, #[N_view, 4, 4]
            "timestamps": self.timestamps, # [N_view, 1]
            "fovy": self.fovy, # [N_view]
        }
        return batch

if __name__ == "__main__":
    loader = DiagBatchLoader()
    # for i in range(10):
    batch = loader.get_batch_init(11)
    print(batch["azimuth"])
    print(batch.keys())