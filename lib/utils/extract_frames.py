import cv2
import os

def extract_frames(video_path, output_dir, new_size=(480,832)): # wh
    """
    从视频中提取帧并保存为 (video_name)xxx.png 格式的文件。

    Args:
        video_path (str): 输入视频文件的完整路径。
        output_dir (str): 保存提取帧的输出目录。
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件未找到: {video_path}")
        return

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_dir, video_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if frame is not None:
            if frame.shape[0] != new_size[0] or frame.shape[1] != new_size[1]:
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        if ret:
            output_filename = f"r_{frame_count:03d}.png"
            output_filepath = os.path.join(output_dir, output_filename)

            # 保存帧为 PNG 图片
            cv2.imwrite(output_filepath, frame)

            frame_count += 1
        else:
            break

    cap.release()
    print(f"视频解帧完成。共提取 {frame_count} 帧到目录: {output_dir}")

if __name__ == "__main__":
    # -------- Ablative Experiments --------
    # video_dir = "debug/rerender/refer_g5_A_girl_is_walking,_against_a_pristine_white_background,_rotating_camera"
    # video_list = [
                #     "Control_S_sde_0.5_30_MS0.mp4",  # w/o sgs 
                #   "Control_S_sde_0.5_10_MS1_sgs0.3.mp4", # full 1
                #   "Control_S_sde_0.5_30_MS1_sgs0.3.mp4", # full 2
                #   "raw.mp4",
                #   "Iter_0_Control_S_mcs_0.5_30_ms1_sgs0.0.mp4",# w/o sde
                #   "Iter_0_Control_S_sde_0.5_30_ms1_sgs0.0.mp4", # w/o sgs
                #   "Iter_0_Control_S_sde_0.5_30_ms1_sgs0.3.mp4", # full 3
                #   "Iter_0_S_sde_0.5_30_ms1_sgs0.3.mp4", # w/o personalized
                #   "Iter_0_S_sde_0.5_30_ms1_sgs0.1.mp4",  # w/o personalized
                #   ] 
                
    # -------- Prepare dataset --------
    # video_dir = "/data2/sunqi/Animate3D/LHM/exps/vis_meshs/video_human_benchmark/human-lrm-1B/train_data/test_image7/dance2/"
    # video_dir = "./debug/rerender/refer_g3_A_girl_is_walking,_against_a_pristine_white_background,_rotating_camera"
    # video_dir = "debug/rerender/refer_h1_A_girl_is_walking,_against_a_pristine_white_background,_rotating_camera"
    # video_dir = "/data2/sunqi/Animate3D/debug/rerender/refer_h2_A_man_is_walk,_against_a_pristine_white_background,_rotating_camera"
    video_dir = "/data2/sunqi/Animate3D/debug/rerender/refer_g3_A_girl_is_walking,_against_a_pristine_white_background,_rotating_camera"
    # video_dir = "/data2/sunqi/Animate3D/debug/Awesome-Training-Free-WAN2.1-Editing"
    video_list = [
                # "Iter_0_CS_sde_0.9_27_ms1_sgs0.2.mp4", # easy
                #   "raw_0.mp4",
                #   "skel_0.mp4",
                #   "Iter_0_CS_ncsdedit_0.5_15_ms1_sgs0.2.mp4",
                # "Iter_0_CS_sdedit_0.5_15_ms1_sgs0.2.mp4",
                # "Iter_0_CS_hfs_0.5_15_ms1_sgs0.2.mp4",
                # "Iter_0_CS_sde_0.6_36_ms1_sgs0.2.mp4",
                "Iter_0_CS_sde_0.6_18_ms1_sgs0.2.mp4"
                # "flowalign.mp4",
                # "flowedit.mp4",
                #   "0128_04.mp4",
                #   "0121_02.mp4",
                #   "0113_06.mp4",
                #   "0102_02.mp4",
                #   "0008_01.mp4",
                  ]
    # video_list = ["d5.mp4"]
    for video_name in video_list:
        input_video = os.path.join(video_dir, video_name)
        output_folder = video_dir

        extract_frames(input_video, output_folder)