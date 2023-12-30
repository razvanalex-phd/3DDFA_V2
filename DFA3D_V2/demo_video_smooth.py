# coding: utf-8

__author__ = "cleardusk"

import argparse
from collections import deque

import cv2
import imageio
import numpy as np
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from tqdm import tqdm

# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark, get_suffix
from utils.render import render

SMOOTHING_RADIUS = 1


def detect_landmarks(
    face_boxes,
    tddfa,
    n_pre,
    queue_ver,
    dense_flag,
    pre_ver,
    i,
    frame_bgr,
):
    if i == 0:
        # detect
        boxes = face_boxes(frame_bgr)
        boxes = [boxes[0]]
        param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        # refine
        param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy="landmark")
        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        # padding queue
        for _ in range(n_pre):
            queue_ver.append(ver.copy())
        queue_ver.append(ver.copy())

    else:
        param_lst, roi_box_lst = tddfa(
            frame_bgr, [pre_ver], crop_policy="landmark"
        )

        roi_box = roi_box_lst[0]
        # todo: add confidence threshold to judge the tracking is failed
        if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

        ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        queue_ver.append(ver.copy())

    return ver


def mouth_bounding_box(points):
    x_min = int(points[0, 48:68].min())
    x_max = int(points[0, 48:68].max())
    y_min = int(points[1, 48:68].min())
    y_max = int(points[1, 48:68].max())
    return x_min, x_max, y_min, y_max


def draw_mouth_shape(img_draw, ver_ave):
    for i in range(48, 68):
        cv2.circle(
            img_draw,
            (int(round(ver_ave[0, i])), int(round(ver_ave[1, i]))),
            1,
            (0, 255, 0),
            -1,
        )


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), "edge")
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode="same")
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(
            trajectory[:, i], radius=SMOOTHING_RADIUS
        )

    return smoothed_trajectory


def estimate_motion(prev_pts, curr_pts, transforms):
    # Find transformation matrix
    m = cv2.estimateAffine2D(prev_pts, curr_pts)[0]

    # Extract translation
    dx = m[0][2]
    dy = m[1][2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    # Store transformation
    transforms.append([dx, dy, da])


def mouth_crop_view(x1, x2, y1, y2, width, height):
    dx = x2 - x1
    dy = y2 - y1
    mid_x = x1 + dx / 2
    mid_y = y1 + dy / 2

    # Resize image to fit
    if dx > width:
        raise NotImplementedError("Scaling not implemented")
    if dy > height:
        raise NotImplementedError("Scaling not implemented")

    x1 = int(mid_x - width // 2)
    x2 = int(mid_x + width // 2)
    y1 = int(mid_y - height // 2)
    y2 = int(mid_y + height // 2)

    return x1, x2, y1, y2


def crop_mouth(img_draw, x1, x2, y1, y2):
    # Crop the image
    img_draw = img_draw[y1:y2, x1:x2, :]
    return img_draw


# FIXME: Idea to implement: make something that works on long video.
# If no face is in the frame, the frames will be skipped.
def mouth_tracking(
    n_pre,
    n_next,
    start,
    end,
    width,
    height,
    dense_flag,
    face_boxes,
    tddfa,
    reader,
):
    # the simple implementation of average smoothing by looking ahead by
    # n_next frames
    # assert the frames of the video >= n
    n = n_pre + n_next + 1
    queue_ver = deque()

    # run
    pre_ver = None
    prev_pts = None
    transforms = []
    points = []

    for i, frame in tqdm(enumerate(reader)):
        if start > 0 and i < start:
            continue
        if end > 0 and i > end:
            break

        try:
            frame_bgr = frame[..., ::-1]  # RGB->BGR
            ver = detect_landmarks(
                face_boxes,
                tddfa,
                n_pre,
                queue_ver,
                dense_flag,
                pre_ver,
                i,
                frame_bgr,
            )
            pre_ver = ver
        except:
            points.append(None)

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            x1, x2, y1, y2 = mouth_bounding_box(ver_ave)
            x1, x2, y1, y2 = mouth_crop_view(x1, x2, y1, y2, width, height)

            # Estimate motion of the center point
            curr_pts = np.int32([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
            if len(points) > 0 and points[-1] is not None:
                estimate_motion(points[-1], curr_pts, transforms)
            points.append(curr_pts)

            queue_ver.popleft()

    # we will loose the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())

        ver_ave = np.mean(queue_ver, axis=0)

        x1, x2, y1, y2 = mouth_bounding_box(ver_ave)
        x1, x2, y1, y2 = mouth_crop_view(x1, x2, y1, y2, width, height)

        # Estimate motion of the center point
        curr_pts = np.int32([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
        if len(points) > 0:
            estimate_motion(points[-1], curr_pts, transforms)
        points.append(curr_pts)

        queue_ver.popleft()

    return points, transforms


def process_video(n_pre, n_next, transforms, points, reader, fps, video_wfp):
    # Given a video path
    writer = imageio.get_writer(video_wfp, fps=fps)

    n = n_pre + n_next + 1

    trajectory = np.cumsum(np.array(transforms, dtype=np.float32), axis=0)
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    for i, frame in tqdm(enumerate(reader)):
        if i >= len(transforms_smooth):
            break
        frame_bgr = frame[..., ::-1]  # RGB->BGR
        crt_points = points.pop(0)

        x1, x2 = crt_points[0, 0], crt_points[2, 0]
        y1, y2 = crt_points[0, 1], crt_points[1, 1]

        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        t_matrix = np.array(
            [
                [np.cos(da), -np.sin(da), dx],
                [np.sin(da), np.cos(da), dy],
            ]
        )
        new_points = t_matrix @ np.array([[x1, y1, 1], [x2, y2, 1]]).T
        x1_new, x2_new = map(int, new_points[0])
        y1_new, y2_new = map(int, new_points[1])

        # frame_bgr = crop_mouth(frame_bgr.copy(), x1_new, x2_new, y1_new, y2_new)

        frame_bgr = cv2.rectangle(
            frame_bgr.copy(), (x1, y1), (x2, y2), (0, 255, 0), 3
        )
        frame_bgr = cv2.rectangle(
            frame_bgr.copy(), (x1_new, y1_new), (x2_new, y2_new), (255, 0, 0), 3
        )

        writer.append_data(frame_bgr[:, :, ::-1])  # BGR -> RGB

    writer.close()
    print(f"Dump to {video_wfp}")


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["OMP_NUM_THREADS"] = "4"

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == "gpu"
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    fn = args.video_fp.split("/")[-1]
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()["fps"]
    suffix = get_suffix(args.video_fp)
    video_wfp = (
        f'examples/results/videos/{fn.replace(suffix, "")}_{args.opt}_smooth.mp4'
    )

    points, transforms = mouth_tracking(
        args.n_pre,
        args.n_next,
        args.start,
        args.end,
        args.width,
        args.height,
        args.opt in ("2d_dense", "3d"),
        face_boxes,
        tddfa,
        reader,
    )
    process_video(
        args.n_pre, args.n_next, transforms, points, reader, fps, video_wfp
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The smooth demo of video of 3DDFA_V2"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="configs/mb1_120x120.yml"
    )
    parser.add_argument("-f", "--video_fp", type=str)
    parser.add_argument(
        "-m", "--mode", default="cpu", type=str, help="gpu or cpu mode"
    )
    parser.add_argument(
        "-n_pre", default=30, type=int, help="the pre frames of smoothing"
    )
    parser.add_argument(
        "-n_next", default=30, type=int, help="the next frames of smoothing"
    )
    parser.add_argument(
        "-width",
        default=256,
        type=int,
        help="the width in pixes of the resulted video",
    )
    parser.add_argument(
        "-height",
        default=256,
        type=int,
        help="the height in pixes of the resulted video",
    )
    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="2d_sparse",
        choices=["2d_sparse", "2d_dense", "3d"],
    )
    parser.add_argument(
        "-s", "--start", default=-1, type=int, help="the started frames"
    )
    parser.add_argument(
        "-e", "--end", default=-1, type=int, help="the end frame"
    )
    parser.add_argument("--onnx", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
