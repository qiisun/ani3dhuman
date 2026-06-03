# python rerender.py --id g2 --motion walk2 --method sde
# python rerender.py --id g3 --motion walk2 --method sde
# CUDA_VISIBLE_DEVICES=0 python rerender.py --id b1 --motion pr --method sde # --use_14b

CUDA_VISIBLE_DEVICES=0 python rerender.py --id b1 --motion pr --method sde --prompt "a man running, black background, fixed camera, photorealistic, high quality" # --use_14b