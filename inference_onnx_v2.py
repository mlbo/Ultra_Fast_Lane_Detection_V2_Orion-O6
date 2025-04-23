# ---------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script of onnx model infer on onnxruntime for Ultra-Fast-Lane-Detection-V2.
"""
import cv2
import onnxruntime as ort
import onnx
import argparse
import os
import sys
import numpy as np
import math
import datetime

# Define the absolute path to the utils package by going up four directory levels from the current file location
_abs_path = os.path.join(os.getcwd(), "../../../../")
# Append the utils package path to the system path, making it accessible for imports
sys.path.append(_abs_path)
from utils.tools import get_file_list
from utils.ulfd_process import preprocess_image_ufld_v2, post_process_ufld_v2, draw_lanes_v2, softmax

# --- Configuration for UFLDv2 (Adjust based on your model/config) ---
IMG_WIDTH = 1600  # Model input width (e.g., cfg.train_width)
IMG_HEIGHT = 320 # Model input height (e.g., cfg.train_height)
CROP_RATIO = 0.6 # Crop ratio used during training (e.g., cfg.crop_ratio)

# Anchors (example for CULane, adjust based on your cfg.row_anchor/col_anchor)
NUM_ROW = 72 # e.g., cfg.num_row
NUM_COL = 81 # e.g., cfg.num_col
ROW_ANCHOR = np.linspace(0.42, 1, NUM_ROW)
COL_ANCHOR = np.linspace(0, 1, NUM_COL)
# --- End Configuration ---



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        default="test_data_v2", # Keep default or point to specific V2 test images
        help="path to the image file path or dir path.",
    )
    parser.add_argument(
        "--onnx_path",
        default="./model/Ultra-Fast-Lane-Detection-V2.onnx", # Default to V2 model
        help="path to the UFLDv2 ONNX model file",
    )
    parser.add_argument(
        "--output_dir", default="./output_v2", help="path to the result output"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Load ONNX model
    print(f"Loading model: {args.onnx_path}")
    # session_options = ort.SessionOptions()
    model = ort.InferenceSession(args.onnx_path)
    onnx_model = onnx.load(args.onnx_path)
    total_params = 0
    for initializer in onnx_model.graph.initializer:
        # Calculate the number of elements (parameters) in the tensor
        param_count = np.prod(initializer.dims)
        total_params += param_count

    # Convert total parameters to millions
    params_in_millions = total_params / 1_000_000
    print(f"Total number of parameters in the ONNX model: {params_in_millions:.2f} M")

    # Consider enabling optimizations if needed
    # session_options.inter_op_num_threads = 4
    # session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Get input/output names
    input_name = model.get_inputs()[0].name
    output_names = [output.name for output in model.get_outputs()]
    print(f"Input Name: {input_name}")
    print(f"Output Names: {output_names}")

    image_list = get_file_list(args.image_path)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(image_list)} images...")
    datas = []
    for img_path in image_list:
        print(f"  Inferring: {os.path.basename(img_path)}")
        try:
            # Preprocess
            input_data, original_img = preprocess_image_ufld_v2(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT), crop_ratio=CROP_RATIO)
            ori_h, ori_w = original_img.shape[:2]
            datas.append(input_data)
            # Inference
            begin = datetime.datetime.now()
            outputs_raw = model.run(output_names, {input_name: input_data})
            end = datetime.datetime.now()
            time_diff = (end - begin).total_seconds() * 1000
            print(f" CPU ONNX Inference time: {time_diff:.2f} ms")
            # Map raw outputs to dictionary using model output names
            outputs_dict = {name: data for name, data in zip(output_names, outputs_raw)}

            # Post-process - Pass CROP_RATIO
            lane_coords = post_process_ufld_v2(
                outputs_dict, 
                (ori_h, ori_w), 
                CROP_RATIO, 
                NUM_ROW,  # Pass NUM_ROW
                NUM_COL,  # Pass NUM_COL
                ROW_ANCHOR,  # Pass ROW_ANCHOR
                COL_ANCHOR  # Pass COL_ANCHOR
            )

            # Draw results
            result_img = draw_lanes_v2(original_img, lane_coords)

            # Save output
            output_filename = os.path.join(output_dir, "onnx_v2_" + os.path.basename(img_path))
            cv2.imwrite(output_filename, result_img)
            print(f"    Result saved to: {output_filename}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    # Save the processed input data for quatanization calibration
    datas = np.concatenate(datas, axis=0)
    np.save("datasets/cal_v2.npy", datas)

    print("Processing complete.")