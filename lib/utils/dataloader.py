#  Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import numpy as np
import torch

def prepare_depth(depth_path, input_frames, device, depth_model):
    os.makedirs(depth_path, exist_ok=True)
    depth_path  = f"{depth_path}/depth_gt_raw.pt" 
    print("run VideoDepthAnything and save.")
    with torch.no_grad():
        depth_gt_raw = depth_model.get_depth_maps(input_frames)
    torch.save(depth_gt_raw.cpu(), depth_path)
    depth_gt_raw = depth_gt_raw.to(device)
    return depth_gt_raw