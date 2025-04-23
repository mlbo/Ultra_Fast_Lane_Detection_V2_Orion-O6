# ---------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script of onnx model infer on onnxruntime for Ultra-Fast-Lane-Detection-V2,
modified to process video input and display FPS/inference time.
"""
import cv2
# import onnxruntime as ort # NOE_Engine handles runtime internally
import argparse
import os
import sys
import numpy as np
import math
import datetime
import time # Import time module for FPS calculation

# Define the absolute path to the utils package by going up four directory levels from the current file location
_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
# Append the utils package path to the system path, making it accessible for imports
sys.path.append(_abs_path)
from utils.ulfd_process import preprocess_image_ufld_v2, post_process_ufld_v2, draw_lanes_v2, softmax
from utils.NOE_Engine import EngineInfer

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
        "--video_path", # Changed from image_path
        default="test_video.mp4", # Default to a video file
        help="path to the video file.",
    )
    parser.add_argument(
        "--onnx_path",
        default="Ultra-Fast-Lane-Detection-V2.cix", # Default to V2 model
        help="path to the UFLDv2 ONNX model file (.cix for NOE)",
    )
    parser.add_argument(
        "--output_dir", default="./output_v2/npu_video", help="path to the result output directory" # Changed output dir
    )
    parser.add_argument(
        "--save_output", action='store_true', help="Save the processed video output." # Added save option
    )
    # Add the new argument
    parser.add_argument(
        "--no_display", action='store_true', help="Do not display the video playback window.",
        default=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Load ONNX model using NOE_Engine
    print(f"Loading model: {args.onnx_path}")
    try:
        model = EngineInfer(args.onnx_path)
    except Exception as e:
        print(f"Error loading NPU engine: {e}")
        print("Please ensure the model path is correct and the NPU environment is set up.")
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Video Input Setup ---
    print(f"Opening video: {args.video_path}")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        sys.exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not determine video FPS. Defaulting to 30.")
        fps = 30 # Set a default FPS if reading fails
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # --- Video Output Setup (Optional) ---
    out_writer = None
    if args.save_output:
        # Ensure filename is safe for the filesystem
        base_name = os.path.basename(args.video_path)
        name, ext = os.path.splitext(base_name)
        output_filename = os.path.join(output_dir, f"npu_v2_{name}.mp4") # Force mp4 extension

        # Use MP4V codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            out_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
            if not out_writer.isOpened():
                print(f"Error: Could not open video writer for {output_filename}")
                print("Video saving disabled.")
                args.save_output = False
            else:
                print(f"Saving processed video to: {output_filename}")
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            print("Video saving disabled.")
            args.save_output = False
            if out_writer:
                out_writer.release()
            out_writer = None


    # --- Processing Loop ---
    frame_count = 0
    total_proc_time = 0
    npu_time_list = []

    while True:
        start_frame_time = time.time() # Record frame processing start time

        ret, frame = cap.read()
        if not ret:
            print("End of video reached or cannot read frame.")
            break

        original_img = frame.copy() # Keep original frame for drawing
        ori_h, ori_w = original_img.shape[:2]

        try:
            # --- Preprocess Frame ---
            # Crop the image (keeping the bottom part defined by CROP_RATIO)
            cut_height = int(ori_h * (1 - CROP_RATIO))
            img_cropped = frame[cut_height:, :, :]

            # Resize the cropped image to the model's input size
            img_resized = cv2.resize(img_cropped, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)

            # Normalize to [0, 1] and transpose to (C, H, W)
            img_normalized = img_resized.astype(np.float32) / 255.0
            input_data = np.transpose(img_normalized, (2, 0, 1))

            # Add batch dimension (B, C, H, W) and ensure contiguous memory
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data)

            # --- Inference ---
            begin_npu = datetime.datetime.now()
            # NOE_Engine expects a list of inputs, even if there's only one
            outputs = model.forward([input_data])
            end_npu = datetime.datetime.now()
            npu_time_diff_ms = (end_npu - begin_npu).total_seconds() * 1000
            npu_time_list.append(npu_time_diff_ms)

            # --- Prepare Outputs Dictionary and Reshape ---
            # Map raw outputs (list) to the dictionary keys expected by post-processing
            # IMPORTANT: Verify the order of outputs from model.forward() matches these keys!
            if len(outputs) != 4:
                 print(f"Error: Expected 4 outputs from model, but got {len(outputs)}. Skipping frame.")
                 continue

            outputs_dict_raw = {
                'loc_row': outputs[0],
                'loc_col': outputs[1],
                'exist_row': outputs[2],
                'exist_col': outputs[3]
            }

            outputs_dict_reshaped = {}
            try:
                # Reshape loc_row: Expected shape (1, 200, 72, 4)
                outputs_dict_reshaped['loc_row'] = np.reshape(outputs_dict_raw['loc_row'], (1, 200, NUM_ROW, 4))
                # Reshape exist_row: Expected shape (1, 2, 72, 4) for argmax(1) later
                outputs_dict_reshaped['exist_row'] = np.reshape(outputs_dict_raw['exist_row'], (1, 2, NUM_ROW, 4))
                # Reshape loc_col: Expected shape (1, 100, 81, 4)
                outputs_dict_reshaped['loc_col'] = np.reshape(outputs_dict_raw['loc_col'], (1, 100, NUM_COL, 4))
                 # Reshape exist_col: Expected shape (1, 2, 81, 4) for argmax(1) later
                outputs_dict_reshaped['exist_col'] = np.reshape(outputs_dict_raw['exist_col'], (1, 2, NUM_COL, 4))

                # Pass the reshaped dictionary to post-processing
                lane_coords = post_process_ufld_v2(
                    outputs_dict_reshaped, 
                    (ori_h, ori_w), 
                    CROP_RATIO, 
                    NUM_ROW,  # Pass NUM_ROW
                    NUM_COL,  # Pass NUM_COL
                    ROW_ANCHOR,  # Pass ROW_ANCHOR
                    COL_ANCHOR  # Pass COL_ANCHOR
                 )

            except ValueError as e:
                 print("Skipping frame.")
                 continue # Skip processing this frame

            # --- Draw results ---
            result_img = draw_lanes_v2(original_img, lane_coords)

            # --- Calculate FPS and Display Info ---
            end_frame_time = time.time()
            frame_proc_time = end_frame_time - start_frame_time
            total_proc_time += frame_proc_time
            current_fps = 1.0 / frame_proc_time if frame_proc_time > 0 else 0

            # Prepare text for display
            text_fps = f"FPS: {current_fps:.1f}" # Display current FPS
            text_npu = f"NPU: {npu_time_diff_ms:.1f}ms" # Display NPU time for this frame

            # Get text size to position it in the top-right corner
            (text_fps_w, text_fps_h), _ = cv2.getTextSize(text_fps, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (text_npu_w, text_npu_h), _ = cv2.getTextSize(text_npu, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Position text
            margin = 10
            text_x = frame_width - max(text_fps_w, text_npu_w) - margin
            text_y_fps = margin + text_fps_h
            text_y_npu = text_y_fps + text_npu_h + margin // 2 # Position NPU time below FPS

            # Draw text on the image
            cv2.putText(result_img, text_fps, (text_x, text_y_fps), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(result_img, text_npu, (text_x, text_y_npu), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # --- Show/Save Frame ---
            if not args.no_display:
                cv2.imshow("Lane Detection - NPU", result_img)

            if args.save_output and out_writer is not None:
                out_writer.write(result_img)

            frame_count += 1

            # Press 'q' to exit the loop (only relevant if display is enabled)
            # Or handle keyboard interrupt (Ctrl+C) for no-display mode
            key = -1
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                 print("Exiting loop...")
                 break

        except Exception as e:
            print(f"\n--- Error processing frame {frame_count} ---")
            import traceback
            traceback.print_exc() # Print detailed exception information
            print("Continuing to next frame...")
            # Optionally break here if errors are critical:
            # break
            continue

    # --- Cleanup ---
    cap.release()
    if out_writer is not None:
        print("Releasing video writer...")
        out_writer.release()
    # Only destroy windows if they were created
    if not args.no_display:
        cv2.destroyAllWindows()
    print("Cleaning up NPU model...")
    model.clean()

    # --- Final Statistics ---
    if frame_count > 0 and total_proc_time > 0:
        avg_proc_time_ms = (total_proc_time / frame_count) * 1000
        avg_fps = frame_count / total_proc_time
        avg_npu_time_ms = sum(npu_time_list) / len(npu_time_list) if npu_time_list else 0
        print(f"\n--- Processing Summary ---")
        print(f"Processed {frame_count} frames.")
        print(f"Average total processing time per frame: {avg_proc_time_ms:.2f} ms")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average NPU inference time: {avg_npu_time_ms:.2f} ms")
    else:
        print("\nNo frames were processed successfully.")

    print("Script finished.")