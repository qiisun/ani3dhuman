import sys
sys.path.append(".")

import torch
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image
import cv2
import os
import argparse

from lib.guidance.Hack_wan_control import WanControlGuidance
from models.Grounded_SAM_2.sam2_utils import Segmenter
from diffsynth import save_video, VideoData


DEFAULT_TEXT_PROMPT = "a person moving, clean background, rotating camera, photorealistic, high quality"


def open_rgb_image(image_path):
    image = Image.open(image_path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


class Rerenderer:
    def __init__(self, text_prompt, reference_image,use_14b_model, rerender_args, device="cuda"):
        self.guidance = WanControlGuidance(text_prompt, reference_image=reference_image, use_14b_model=use_14b_model)
        self.human_segmentor = Segmenter()
        self.device = device
        self.rerender_args = rerender_args
    
    def get_pose(self, video_path, skel_path):
        self.guidance.get_pose(video_path, skel_path)
        
    def get_human_mask(self, first_frame_pil):
        self.human_segmentor.set_text("person.")
        mask = torch.from_numpy(self.human_segmentor.get_mask(self.re_render_pth, first_frame_pil)).float().to(self.device).detach() # [T, H, W] 1-表示人体区域
        return mask
        
    def get_render(self, video_tensor, mid_save_dir=None, rerender_path=None):
        """
        video_tensor: [1, 3, T, H, W], [-1, 1]
        output: pseudo_gt: [1, 3, T, H, W], [-1, 1]
        """
        # define path
        init_render_pth = f"{mid_save_dir}/raw_rgb.mp4"
        init_skel_pth = f"{mid_save_dir}/skel.mp4"
        re_render_pth = f"{mid_save_dir}/rerender.mp4" if rerender_path is None else rerender_path
        self.re_render_pth = re_render_pth
        video_pil = [to_pil_image((video_tensor[0,:,i].cpu()+1)/2.0) for i in range(video_tensor.shape[2])] # [T, PIL]
        self.video_pil = video_pil
        save_video(video_pil, init_render_pth, fps=30, quality=9)
            
        # prepare segmentation masks
        self.human_segmentor.set_text("person.")
        mask_human_init = torch.from_numpy(self.human_segmentor.get_mask(init_render_pth, video_pil[0])).float().to(self.device).detach() # [T, H, W] 1-表示人体区域
        self.human_segmentor.set_text("clothes.")
        mask_garment_init = torch.from_numpy(self.human_segmentor.get_mask(init_render_pth, video_pil[0])).float().to(self.device).detach() # [T, H, W] 1-表示服装区域
        mask_preserve = mask_human_init * (1-mask_garment_init)

        # prepare control video
        self.guidance.get_pose(init_render_pth, init_skel_pth)
        control_video = VideoData(init_skel_pth, height=self.rerender_args['height'], width=self.rerender_args['width'])
        additional_args = {"control_video": control_video, "save_vid_pth": re_render_pth, "mask_preserve": mask_preserve}

        # generative re-rendering
        pseudo_gt, video_pil = self.guidance.video2video(input_video=video_tensor, **self.rerender_args, **additional_args) # [-1,1]
        self.first_frame_pil = video_pil[0]
        return pseudo_gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Rerendering Script")
    parser.add_argument("--id", type=str, default="g1", help="The identity args.id (e.g., g1)")
    parser.add_argument("--motion", type=str, default="walk", help="The motion name (e.g., walk)")
    parser.add_argument("--use_14b", action="store_true", help="Whether to use the 14b model")
    parser.add_argument("--method", default="sde", help="The denoising method to use (e.g., sde)")
    parser.add_argument(
        "--text_prompt",
        "--prompt",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help="Text prompt for video rerendering. You can use {id} and {motion} placeholders.",
    )
    
    args = parser.parse_args()
    
    print(f"Processing: args.id={args.id}, args.motion={args.motion}")

    root_path = "."
    text_prompt = args.text_prompt.format(id=args.id, motion=args.motion)
    print(f"Text prompt: {text_prompt}")
    
    video_path = f"{root_path}/data/dnerf/{args.motion}-{args.id}-supp/mid/raw_rgb.mp4"
    mid_save_dir = f"{root_path}/data/dnerf/{args.motion}-{args.id}-supp/mid"
    
    os.makedirs(mid_save_dir, exist_ok=True)
    
    render_tensor = read_video(video_path, pts_unit='sec')[0]
    if render_tensor.shape[-1] == 4:
        print(f"Warning: input video has alpha channel, dropping alpha: {video_path}")
        render_tensor = render_tensor[..., :3]
    elif render_tensor.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA video, got shape {render_tensor.shape} from {video_path}")
    render_tensor = render_tensor.to("cuda") / 255.0 * 2 - 1 # [-1, 1], [T, H, W, 3]
    render_tensor = render_tensor.permute(3, 0, 1, 2).unsqueeze(0) # [1, 3, T, H, W]
     
    rerender_path = f"{mid_save_dir}/rerender_{args.method}.mp4"
    
    if False: # os.path.exists(rerender_path):
        print("skip re-rendering")
    else:
        ref_image_path = f"./data/images/refer_{args.id}.png"
        if not os.path.exists(ref_image_path):
            print(f"Warning: Reference image not found at {ref_image_path}")
        
        ref_image = open_rgb_image(ref_image_path)

        rerender_args = dict(
            height=832,
            width=480,
            denoising_strength=0.7, # higher for more significant changes, lower for more fidelity to original video
            sgs=0.05, # higher -> fidelty
            denoising_methods=args.method,
            num_inference_steps=int(10 * 0.7),
        )
        
        rerenderer = Rerenderer(text_prompt, ref_image, use_14b_model=args.use_14b, rerender_args=rerender_args)
        video_tensor = rerenderer.get_render(render_tensor, mid_save_dir=mid_save_dir, rerender_path=rerender_path)
        
        print(f"Done. Shape: {video_tensor.shape}, Range: [{video_tensor.min()}, {video_tensor.max()}]")