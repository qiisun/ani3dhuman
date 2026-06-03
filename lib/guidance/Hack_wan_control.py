import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from easy_dwpose import DWposeDetector
import torch
import cv2
from torchvision.io import read_video

def resize_and_pad(image: Image.Image, target_size: tuple, pad_color=(255, 255, 255)) -> Image.Image:
    original_w, original_h = image.size
    target_h, target_w = target_size
    scale = min(target_w / original_w, target_h / original_h)

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized_image = image.resize((new_w, new_h), Image.LANCZOS)

    if image.mode == 'RGBA':
        padded_image = Image.new('RGBA', (target_w, target_h), pad_color + (255,))
    else:
        padded_image = Image.new('RGB', (target_w, target_h), pad_color)

    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image

class WanControlGuidance:
    def __init__(self, text_prompt=None, reference_image=None, use_14b_model=False):
        self.use_14b_model = use_14b_model
        if use_14b_model:
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ],
            )
            self.model_name = "Control-L"
        else:
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cuda"),
                    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cuda"),
                    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cuda"),
                    ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cuda"),
                ],
            )
            self.model_name = "Control-S"
        self.pipe.enable_vram_management()
        self.text_prompt = text_prompt
        self.negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        self.reference_image = reference_image
        print("[INFO] Initialization Wan-Control Model.")
        
    def get_pose(self, input_video_path, output_video_path):
        dwpose = DWposeDetector(device="cpu")
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        skeleton_frames = []

        print("Processing video frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            skeleton = dwpose(pil_img, output_type="pil", include_hands=True, include_face=True, detect_resolution=768)
            skeleton_frames.append(skeleton.resize((frame.shape[1], frame.shape[0])))

        cap.release()
        save_video(skeleton_frames, output_video_path, fps=fps)
        return 
            
    @torch.autocast("cuda", dtype=torch.bfloat16) # very important!
    def video2video(self, input_video, control_video, 
                    height=None, width=None, num_inference_steps=50, 
                    denoising_strength=0.5, save_vid_pth=None, denoising_methods='t2v', 
                    sgs=0, mask_preserve=None):
        video_tensor, video = self.pipe(
            input_video=input_video,
            prompt=self.text_prompt,
            negative_prompt=self.negative_prompt,
            control_video=control_video, reference_image=self.reference_image,
            height=height, width=width, num_frames=81,
            seed=1, tiled=True if self.use_14b_model else False,
            return_tensor_only=False,
            num_inference_steps=num_inference_steps,
            denoising_methods=denoising_methods,
            denoising_strength=denoising_strength if not denoising_methods=='t2v' else 1.0,
            mask_preserve=mask_preserve,
            self_guidance_scale=sgs,
            cfg_scale=5.0,
        )
        if save_vid_pth is not None:
            save_video(video, save_vid_pth, fps=30, quality=9)
        video_tensor = video_tensor.to(input_video.device)
        return video_tensor, video

if __name__ == "__main__":
    raw_image_pth = "debug/rerender/refer_g5_A_girl_is_walking,_against_a_pristine_white_background,_rotating_camera/raw.mp4"
    video = read_video(raw_image_pth, pts_unit='sec')[0].to("cuda")/255.0 * 2 - 1 # [-1, 1]
    height, width = video.shape[1], video.shape[2]
    video = video.permute(3, 0, 1, 2).unsqueeze(0) # [1,3,T,H,W]
    textprompt = "A girl is squatting down, photorealistic and high quality, white background"
    reference_image = resize_and_pad(Image.open("data/images/refer_g2.png"), target_size=(height, width))
    guidance = WanControlGuidance(text_prompt=textprompt, reference_image=reference_image, use_14b_model=False)
    guidance.get_pose(raw_image_pth, "debug/skel.mp4")
