import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import logging
import copy
import datetime
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
import comfy.ops
import comfy.model_patcher
from comfy.utils import ProgressBar

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))
folder_paths.add_model_folder_path("nlf", os.path.join(folder_paths.models_dir, "nlf"))

from .vitpose_utils.utils import bbox_from_detector, crop, load_pose_metas_from_kp2ds_seq, aaposemeta_to_dwpose_scail

def convert_openpose_to_target_format(frames, max_people=2):
    NUM_BODY = 18
    NUM_FACE = 70
    NUM_HAND = 21

    results = []
    for frame in frames:
        canvas_width = frame['canvas_width']
        canvas_height = frame['canvas_height']
        people = frame['people'][:max_people]

        bodies = []
        hands = []
        faces = []
        body_scores = []
        hand_scores = []
        face_scores = []

        for person in people:
            pose_raw = person.get('pose_keypoints_2d') or []
            if len(pose_raw) != NUM_BODY * 3:
                continue

            pose = np.array(pose_raw).reshape(-1, 3)
            pose_xy = np.stack([pose[:, 0] / canvas_width, pose[:, 1] / canvas_height], axis=1)
            bodies.append(pose_xy)
            body_scores.append(pose[:, 2])

            face_raw = person.get('face_keypoints_2d') or []
            if len(face_raw) == NUM_FACE * 3:
                face = np.array(face_raw).reshape(-1, 3)
                face_xy = np.stack([face[:, 0] / canvas_width, face[:, 1] / canvas_height], axis=1)
                faces.append(face_xy)
                face_scores.append(face[:, 2])

            hand_left_raw = person.get('hand_left_keypoints_2d') or []
            hand_right_raw = person.get('hand_right_keypoints_2d') or []
            if len(hand_left_raw) == NUM_HAND * 3:
                hand_left = np.array(hand_left_raw).reshape(-1, 3)
                hand_left_xy = np.stack([hand_left[:, 0] / canvas_width, hand_left[:, 1] / canvas_height], axis=1)
                hands.append(hand_left_xy)
                hand_scores.append(hand_left[:, 2])
            if len(hand_right_raw) == NUM_HAND * 3:
                hand_right = np.array(hand_right_raw).reshape(-1, 3)
                hand_right_xy = np.stack([hand_right[:, 0] / canvas_width, hand_right[:, 1] / canvas_height], axis=1)
                hands.append(hand_right_xy)
                hand_scores.append(hand_right[:, 2])

        result = {
            'bodies': {
                'candidate': np.array(bodies, dtype=np.float32),
                'subset': np.array([np.arange(NUM_BODY) for _ in bodies], dtype=np.float32) if bodies else np.array([])
            },
            'hands': np.array(hands, dtype=np.float32),
            'faces': np.array(faces, dtype=np.float32),
            'body_score': np.array(body_scores, dtype=np.float32),
            'hand_score': np.array(hand_scores, dtype=np.float32),
            'face_score': np.array(face_scores, dtype=np.float32)
        }
        results.append(result)
    return results

def scale_faces(poses, pose_2d_ref):
    # Input: two lists of dict, poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2
    # Scale the facial keypoints in poses according to the center point of the face
    # That is: calculate the distance from the center point (idx: 30) to other facial keypoints in ref,
    # and the same for poses, then get scale_n as the ratio
    # Clamp scale_n to the range 0.8-1.5, then apply it to poses
    # Note: poses are modified in place

    ref = pose_2d_ref[0]
    pose_0 = poses[0]

    face_0 = pose_0['faces']  # shape: (1, 68, 2)
    face_ref = ref['faces']

    # Guard against missing face keypoints (e.g., subject facing away)
    if len(face_0) == 0 or len(face_ref) == 0:
        return 1.0

    # Extract numpy arrays
    face_0 = np.array(face_0[0])      # (68, 2)
    face_ref = np.array(face_ref[0])

    # Center point (nose tip or face center)
    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    # Calculate distance to center point
    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    # Avoid the 0 distance of the center point itself
    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    # Clamp to [0.8, 1.5]
    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose['faces']
        if len(face) > 0:
            # Extract numpy array
            face = np.array(face[0])      # (68, 2)
            center = face[center_idx]
            scaled_face = (face - center) * scale_n + center
            poses[i]['faces'][0] = scaled_face

        body = pose['bodies']
        candidate = body['candidate']
        candidate_np = np.array(candidate[0])   # (14, 2)
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]['bodies']['candidate'][0] = scaled_candidate

    return scale_n

class PoseDetectionVitPoseToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": ("POSEMODEL",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "ViTPose to DWPose format pose detection node."

    def process(self, vitpose_model, images):

        detector = vitpose_model["yolo"]
        pose_model = vitpose_model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        dwposes = [aaposemeta_to_dwpose_scail(meta) for meta in pose_metas]
        swap_hands = True
        out_dict = {"poses": dwposes, "swap_hands": swap_hands}
        return out_dict,


class ConvertOpenPoseKeypointsToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "max_people": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "tooltip": "Maximum number of people to process per frame"}),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Convert OpenPose format keypoints to DWPose format."

    def process(self, keypoints, max_people=2):
        swap_hands = False
        out_dict = {"poses": convert_openpose_to_target_format(keypoints, max_people=max_people), "swap_hands": swap_hands}
        return out_dict,


class RenderNLFPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "Optional DW pose model for 2D drawing"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Optional reference DW pose model for alignment"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu", "tooltip": "Taichi device to use for rendering"}),
                "scale_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to scale hand keypoints when aligning DW poses"}),
                "render_backend": (["taichi", "torch"], {"default": "taichi", "tooltip": "Rendering backend to use"}),
            }
    }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(self, nlf_poses, width, height, dw_poses=None, ref_dw_pose=None, draw_face=True, draw_hands=True, render_device="gpu", scale_hands=True, render_backend="taichi"):

        from .NLFPoseExtract.nlf_render import render_nlf_as_images, render_multi_nlf_as_images, shift_dwpose_according_to_nlf, process_data_to_COCO_format, intrinsic_matrix_from_field_of_view
        from .NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down
        if render_backend == "taichi":
            try:
                import taichi as ti
                device_map = {
                    "cpu": ti.cpu,
                    "gpu": ti.gpu,
                    "opengl": ti.opengl,
                    "cuda": ti.cuda,
                    "vulkan": ti.vulkan,
                    "metal": ti.metal,
                }
                ti.init(arch=device_map.get(render_device.lower()))
            except:
                logging.warning("Taichi selected but not installed. Falling back to torch rendering.")
                render_backend = "torch"

        if isinstance(nlf_poses, dict):
            pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
        else:
            pose_input = nlf_poses

        dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
        swap_hands = dw_poses.get("swap_hands", False) if dw_poses is not None else False

        ori_camera_pose = intrinsic_matrix_from_field_of_view([height, width])
        ori_focal = ori_camera_pose[0, 0]

        num_people = dw_pose_input[0]['bodies']['candidate'].shape[0] if dw_poses is not None else 0

        if dw_poses is not None and ref_dw_pose is not None and num_people == 1:
            ref_dw_pose_input = copy.deepcopy(ref_dw_pose["poses"])

            # Find the first valid pose
            pose_3d_first_driving_frame = None
            for pose in pose_input:
                if pose.shape[0] == 0:
                    continue
                candidate = pose[0].cpu().numpy()
                if np.any(candidate):
                    pose_3d_first_driving_frame = candidate
                    break
            if pose_3d_first_driving_frame is None:
                raise ValueError("No valid pose found in pose_input.")

            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)
            poses_2d_ref = ref_dw_pose_input[0]['bodies']['candidate'][0][:14]
            poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * width
            poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * height

            poses_2d_subset = ref_dw_pose_input[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices, valid_upper_indices, valid_lower_indices = [], [], []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]

            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)

            valid_indices = [1] + valid_lower_indices if len(valid_upper_indices) < 4 else [1] + valid_lower_indices + valid_upper_indices # align body or only lower body

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]

            if len(valid_lower_indices) >= 4:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)
            else:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)

            scale_face = scale_faces(list(dw_pose_input), list(ref_dw_pose_input))   # poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2

            logging.info(f"Scale - m: {scale_m}, face: {scale_face}")
            shift_dwpose_according_to_nlf(pose_input, dw_pose_input, ori_camera_pose, new_camera_intrinsics, height, width, swap_hands=swap_hands, scale_hands=scale_hands, scale_x=scale_m, scale_y=scale_m*scale_s)

            intrinsic_matrix = new_camera_intrinsics
        else:
            intrinsic_matrix = ori_camera_pose

        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

        return (frames_tensor.cpu().float(), mask.cpu().float())

class SaveNLFPosesAs3D:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "filename_prefix": ("STRING", {"default": "nlf_pose_3d"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 300.0, "step": 0.1, "tooltip": "Frames per second for the output animation"}),
            "cylinder_radius": ("FLOAT", {"default": 21.5, "tooltip": "Radius of the cylinders representing bones"}),
            },
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_3d"
    CATEGORY = "WanVideoWrapper"

    def save_3d(self, nlf_poses, filename_prefix, fps, cylinder_radius):
        from .NLFPoseExtract.nlf_render import get_cylinder_specs_list_from_poses
        from .render_3d.export_utils import save_cylinder_specs_as_glb_animation
        try:
            if isinstance(nlf_poses, dict):
                pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
            else:
                pose_input = nlf_poses

            cylinder_specs_list = get_cylinder_specs_list_from_poses(pose_input, include_missing=True)
            logging.info(f"Generated {len(cylinder_specs_list)} frames of cylinder specs")

            output_dir = folder_paths.get_output_directory()
            full_output_folder = os.path.join(output_dir, filename_prefix)
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            filename = f"{filename_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
            filepath = os.path.join(full_output_folder, filename)

            logging.info(f"Saving as GLB animation to {full_output_folder}")
            logging.info(f"Starting GLB animation export. Frames: {len(cylinder_specs_list)}")
            save_cylinder_specs_as_glb_animation(cylinder_specs_list, filepath, fps=fps, radius=cylinder_radius)
            logging.info(f"Saved GLB: {filepath}")
        except Exception as e:
            logging.error(f"Error in SaveNLFPosesAs3D: {e}")
            raise e

        return (filepath,)


# NLF model loader

class NLFModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_model": (folder_paths.get_filename_list("nlf"), {
                    "tooltip": "NLF model (.safetensors) from ComfyUI/models/nlf/",
                }),
            },
        }

    RETURN_TYPES = ("NLF_MODEL",)
    RETURN_NAMES = ("nlf_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SCAIL-Pose"

    def loadmodel(self, nlf_model):
        from safetensors.torch import load_file
        from .nlf_model.model import NLFModel
        from .nlf_model.multiperson import MultipersonNLF, load_detector

        model_path = folder_paths.get_full_path_or_raise("nlf", nlf_model)
        logging.info(f"Loading NLF model from {model_path}")
        sd = load_file(model_path)

        crop_sd = {}
        detector_sd = {}
        for k, v in sd.items():
            if k.startswith('detector.'):
                detector_sd[k[len('detector.'):]] = v
            elif not k.startswith('cano_all.'):
                crop_sd[k] = v

        crop_model = NLFModel.from_state_dict(crop_sd, operations=comfy.ops.manual_cast).eval()

        # Wrap in ModelPatcher for ComfyUI memory management
        load_device = mm.get_torch_device()
        model_patcher = comfy.model_patcher.ModelPatcher(
            crop_model, load_device=load_device, offload_device=offload_device
        )

        detector = None
        if detector_sd:
            logging.info(f"Loading bundled RT-DETR detector ({len(detector_sd)} keys)")
            detector = load_detector(detector_sd)

        # SMPL canonical points: first 1024 are vertices, last 24 are joints
        canonical_points = sd.get('cano_all.smpl', crop_model.canonical_locs())
        num_vertices = 1024 if 'cano_all.smpl' in sd else 0

        pipeline = MultipersonNLF(
            crop_model=crop_model,
            model_patcher=model_patcher,
            detector=detector,
            canonical_points=canonical_points,
            num_vertices=num_vertices,
        )

        logging.info("NLF model loaded successfully")
        return (pipeline,)


class NLFPredictPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_model": ("NLF_MODEL",),
                "images": ("IMAGE", {"tooltip": "Input images (BHWC format)"}),
            },
            "optional": {
                "per_batch": ("INT", {"default": 1, "min": -1, "max": 10000, "step": 1,
                    "tooltip": "Images per batch. -1 = all at once. 1 = lowest VRAM usage."}),
                "num_aug": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Number of test-time augmentations. More = slower but more accurate."}),
                "detector_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Person detection confidence threshold"}),
            },
        }

    RETURN_TYPES = ("NLFPRED", "BBOX",)
    RETURN_NAMES = ("pose_results", "bboxes")
    FUNCTION = "predict"
    CATEGORY = "SCAIL-Pose"

    def predict(self, nlf_model, images, per_batch=1, num_aug=1, detector_threshold=0.3):
        num_images = images.shape[0]
        batch_size = num_images if per_batch == -1 else per_batch

        # Convert to NCHW on GPU once
        images_nchw = images.permute(0, 3, 1, 2)

        # Phase 1: Detect persons per frame
        nlf_model.detector.load()
        all_boxes = []
        pbar_det = ProgressBar(num_images)
        for i in tqdm(range(0, num_images, batch_size), desc="Detecting"):
            end_idx = min(i + batch_size, num_images)
            all_boxes.extend(nlf_model.detector.detect(images_nchw[i:end_idx].to(device), threshold=detector_threshold))
            pbar_det.update(end_idx - i)

        # Phase 2: Estimate poses per frame
        mm.load_model_gpu(nlf_model.model_patcher)
        all_joints3d = []
        pbar_nlf = ProgressBar(num_images)
        for i in tqdm(range(0, num_images, batch_size), desc="Estimating poses"):
            end_idx = min(i + batch_size, num_images)

            result = nlf_model.detect_and_estimate(
                images_nchw[i:end_idx].to(device), num_aug=num_aug, boxes=all_boxes[i:end_idx],
            )

            all_joints3d.extend(result['poses3d'])
            pbar_nlf.update(end_idx - i)

        all_boxes = [b.to(offload_device) for b in all_boxes]
        all_joints3d = [j.to(offload_device) for j in all_joints3d]

        pose_results = {
            'joints3d_nonparam': [all_joints3d],
        }

        formatted_boxes = []
        for box in all_boxes:
            if box.numel() == 0 or box.shape[0] == 0:
                formatted_boxes.append([0.0, 0.0, 0.0, 0.0])
            else:
                x, y, w, h = box[0, :4].cpu().tolist()
                formatted_boxes.append([x, y, x + w, y + h])

        return (pose_results, formatted_boxes)


NODE_CLASS_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": PoseDetectionVitPoseToDWPose,
    "RenderNLFPoses": RenderNLFPoses,
    "ConvertOpenPoseKeypointsToDWPose": ConvertOpenPoseKeypointsToDWPose,
    "SaveNLFPosesAs3D": SaveNLFPosesAs3D,
    "NLFModelLoader": NLFModelLoader,
    "NLFPredictPoses": NLFPredictPoses,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": "Pose Detection VitPose to DWPose",
    "RenderNLFPoses": "Render NLF Poses",
    "ConvertOpenPoseKeypointsToDWPose": "Convert OpenPose Keypoints to DWPose",
    "SaveNLFPosesAs3D": "Save NLF Poses as 3D Animation",
    "NLFModelLoader": "NLF Model Loader",
    "NLFPredictPoses": "NLF Predict Poses",
}
