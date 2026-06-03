import torch
from PIL import Image
from diffsynth import save_video, VideoData
from torchvision.io import read_video

class WanGuidance:
    def __init__(self, text_prompt=None, preencode_prompt=False, use_14b_model=False, use_fe=False, prompt_target=""):
        self.use_14b_model = use_14b_model
        self.use_fe = use_fe
        if not self.use_fe: # basic SDEdit
            from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
        else: # with FlowEdit
            from diffsynth.pipelines.wan_video_new1 import WanVideoPipeline, ModelConfig
            self.prompt_target = prompt_target
        if use_14b_model:
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ],
            )

        else:
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cuda"),
                    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ],
                preencode_prompt=preencode_prompt,
            )
        self.pipe = pipe
        self.pipe.enable_vram_management()
        self.preencode_prompt = preencode_prompt
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，裙子中间出现缝隙"
        if not self.preencode_prompt:
            self.text_prompt = text_prompt
            self.negative_prompt = negative_prompt
        else:
            self.text_embedding_pos = pipe.prompter.encode_prompt(text_prompt, positive=True, device=pipe.device)
            self.text_embedding_neg = pipe.prompter.encode_prompt(negative_prompt, positive=False, device=pipe.device)
        print("[INFO] Initialization Wan1.3B Model.")

    def text2video(self):
        # Text-to-video
        video = self.pipe(
            text_prompt_embds = self.text_embedding_pos,
            negative_prompt_embds = self.text_embedding_neg,
            seed=0, tiled=False,
        )
        save_video(video, "video1.mp4", fps=15, quality=5)

    @torch.autocast("cuda", dtype=torch.bfloat16) # very important!
    def video2video(self, input_video, 
                    height=None, width=None, num_inference_steps=50, 
                    denoising_strength=0.5, save_vid_pth=None, denoising_methods='t2v', 
                    sgs=0, mask_preserve=None):
        video_tensor, video = self.pipe(
            input_video=input_video,
            prompt=self.text_prompt,
            negative_prompt=self.negative_prompt,
            height=height, width=width, num_frames=81,
            seed=1, tiled=True if self.use_14b_model else False,
            return_tensor_only=False,
            num_inference_steps=num_inference_steps,
            denoising_methods=denoising_methods,
            denoising_strength=denoising_strength if not denoising_methods=='t2v' else 1.0,
            mask_preserve=mask_preserve,
            self_guidance_scale=sgs,
            cfg_scale=2.0,
        )
        if save_vid_pth is not None:
            save_video(video, save_vid_pth, fps=30, quality=9)
        video_tensor = video_tensor.to(input_video.device)
        return video_tensor, video

if __name__ == "__main__":
    # video = VideoData("video1.mp4", height=480, width=832)
    video = read_video("debug/video1.mp4", pts_unit='sec')[0].to("cuda")/255.0 * 2 - 1 # [-1, 1]
    height, width = video.shape[1], video.shape[2]
    video = video.permute(3, 0, 1, 2).unsqueeze(0) # [1,3,T,H,W]
    textprompt = "A man is walking, photorealistic and high quality, white background"
    guidance = WanGuidance(textprompt, use_14b_model=True)
    # guidance.text2video("test")
    guidance.video2video(video=video, height=height, width=width, save_vid=True, return_tensor=False)