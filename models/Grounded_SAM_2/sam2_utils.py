import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import sys
sys.path.append("models/Grounded_SAM_2/")  # NOQA
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images


class Segmenter:
    def __init__(self):
        """
        Step 1: Environment settings and model initialization
        """
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # init sam image predictor and video predictor model
        sam2_path = "models/Grounded_SAM_2/"
        sam2_checkpoint = os.path.join(sam2_path, "checkpoints/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        self.video_predictor = video_predictor
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.image_predictor = image_predictor


        # init grounding dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        self.processor = processor
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.grounding_model = grounding_model

    def set_text(self, text):

        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot
        self.text = text
    
    def get_image_mask(self,image):
        # run Grounding DINO on the image
        inputs = self.processor(images=image, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"].cpu().numpy()
        OBJECTS = results[0]["labels"]

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # convert the mask shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        
        return masks
      
    
    def load_cuda(self):
        # self.video_predictor.to("cuda")
        # self.image_predictor.to("cuda")
        self.grounding_model.to("cuda")
        
    def offload_cuda(self):
        # self.video_predictor.to("cpu")
        # self.image_predictor.to("cpu")
        self.grounding_model.to("cpu")
       
    def get_mask(self, original_video_path, first_frame, visualization=False):
        
        self.load_cuda()

        # init video predictor state
        inference_state = self.video_predictor.init_state(video_path=original_video_path)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)


        """
        Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
        """

        # prompt grounding dino to get the box coordinates on specific frame
        # img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
        # image = Image.open(img_path)
        

        # run Grounding DINO on the image
        inputs = self.processor(images=first_frame, text=self.text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[first_frame.size[::-1]]
        )

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(np.array(first_frame.convert("RGB")))

        # process the detection results
        input_boxes = results[0]["boxes"].cpu().numpy()
        OBJECTS = results[0]["labels"]

        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # convert the mask shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        """
        Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        """

        PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

        assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if PROMPT_TYPE_FOR_VIDEO == "point":
            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif PROMPT_TYPE_FOR_VIDEO == "box":
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif PROMPT_TYPE_FOR_VIDEO == "mask":
            for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        segments_all = []

        for frame_idx, segments in video_segments.items():
            # choose the first
            segments_all.append(segments[1][0]) # TODO: check the segments 取得Top物体的mask
        segments_all = np.stack(segments_all, axis=0) # [N, H, W]
        assert segments_all.sum() > segments_all.shape[2] * segments_all.shape[1] * 1, "No segmentation found"
        """
        Step 5: Visualize the segment results across the video and save them
        """
        if visualization:
            segments_all = {key: segments_all[key] for key in [1]}
            save_dir = "debug/tracking_results"

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            video_dir = "debug/videos/garment"
            # if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
            # extract frames from the original video
            vidcap = cv2.VideoCapture(original_video_path)
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(os.path.join(video_dir, f"{count:05d}.jpg"), image)     # save frame as JPEG file
                success, image = vidcap.read()
                count += 1
            vidcap.release()
            
            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


            ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
            for frame_idx, segments in video_segments.items():
                img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
                
                object_ids = list(segments.keys())
                masks = list(segments.values())
                masks = np.concatenate(masks, axis=0)
                # masks[2] = ~(~masks[0] * ~masks[1] * ~masks[2])
                
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                    mask=masks, # (n, h, w)
                    class_id=np.array(object_ids, dtype=np.int32),
                )
                box_annotator = sv.BoxAnnotator()
                annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
                label_annotator = sv.LabelAnnotator()
                annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
                mask_annotator = sv.MaskAnnotator()
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


            """
            Step 6: Convert the annotated frames to video
            """

            output_video_path = "./debug/children_tracking_demo_video.mp4"
            create_video_from_images(save_dir, output_video_path)
        
        self.offload_cuda()
        return segments_all

def get_first_frame_as_pil(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return None

    ret, frame = cap.read()

    cap.release()

    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        return pil_image
    else:
        print(f"错误：无法从视频文件 {video_path} 读取第一帧")
        return None
    
if __name__ == "__main__":
    segmenter = Segmenter(text="garment.")
    human_segmenter = Segmenter(text="garment.")
    # video_path = "notebooks/videos/garment.mp4"
    video_path = "debug/rerender/refer_h4_A_man_is_walk,_against_a_pristine_white_background,_rotating_camera/raw_0.mp4"
    first_frame = get_first_frame_as_pil(video_path)
    segments = segmenter.get_mask(original_video_path=video_path, first_frame=first_frame, visualization=False)
    human_segments = human_segmenter.get_mask(original_video_path=video_path, first_frame=first_frame, visualization=False)
    
    
    
    # segment the 
    # masks = segmenter.get_image_mask(first_frame) # [1,1, H, W] numpy
    # import ipdb; ipdb.set_trace()
    # image_array_uint8 = (masks[0,0] * 255).astype(np.uint8)
    # image_pil = Image.fromarray(image_array_uint8)
    # image_pil.save('debug/2.png')
