# UFLD V2 specific functions
import numpy as np
import cv2


def preprocess_image_ufld_v2(img_path, target_size=(1600, 320), crop_ratio=0.6):
    """
    Preprocesses an image for UFLDv2 ONNX inference.
    Matches the logic in trt_infer.py: crop, resize, normalize 0-1, transpose.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    ori_h, ori_w = img.shape[:2]
    cut_height = int(ori_h * (1 - crop_ratio)) # Calculate dynamic cut height

    img_cropped = img[cut_height:, :, :]
    img_resized = cv2.resize(img_cropped, target_size, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite("test.jpg", img_resized)

    # Normalize to [0, 1] and transpose to (C, H, W)
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_data = np.transpose(img_normalized, (2, 0, 1))

    # Add batch dimension (B, C, H, W)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.ascontiguousarray(input_data) # Ensure contiguous memory
    return input_data, img # Return original image for drawing

def post_process_ufld_v2(outputs, ori_img_shape, crop_ratio, num_row, num_col, row_anchor, col_anchor):
    """
    Post-processes the UFLDv2 model outputs to get lane coordinates in the original image space.
    Adapted from pred2coords in demo.py/trt_infer.py using numpy.
    outputs: Dictionary mapping output names ('loc_row', 'loc_col', 'exist_row', 'exist_col') to numpy arrays.
    ori_img_shape: Tuple (height, width) of the original image before preprocessing.
    crop_ratio: The ratio of the image height kept from the bottom during preprocessing.
    """
    # Extract outputs - ensure keys match your ONNX model's output names
    try:
        loc_row = outputs['loc_row']      # Shape: (batch, num_grid_row, num_cls_row, num_lane_row) e.g., (1, 200, 72, 4)
        exist_row = outputs['exist_row']  # Shape: (batch, num_cls_row, num_lane_row) e.g., (1, 72, 4)
        loc_col = outputs['loc_col']      # Shape: (batch, num_grid_col, num_cls_col, num_lane_col) e.g., (1, 100, 81, 4)
        exist_col = outputs['exist_col']  # Shape: (batch, num_cls_col, num_lane_col) e.g., (1, 81, 4)
    except KeyError as e:
        print(f"Error: Output key {e} not found in model outputs. Available keys: {list(outputs.keys())}")
        print("Please ensure the 'expected_output_keys' list in the main script matches your model's output names.")
        return [] # Return empty list on error

    batch_size, num_grid_row, num_cls_row, num_lane_row = loc_row.shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = loc_col.shape

    ori_img_h, ori_img_w = ori_img_shape
    cut_height = int(ori_img_h * (1.0 - crop_ratio)) # Calculate cut height used in preprocessing
    cropped_img_h = ori_img_h - cut_height

    # --- Process Rows (Horizontal location prediction for each row anchor) ---
    # Find the grid cell index with the maximum probability for each row anchor and lane
    max_indices_row = loc_row.argmax(1) # Shape: (batch, num_cls_row, num_lane_row)
    # Determine if each row anchor exists for each lane based on max probability in existence prediction
    valid_row = exist_row.argmax(1)     # Shape: (batch, num_cls_row, num_lane_row)

    coords = []
    # Define which lanes are primarily defined by row anchors (typically inner lanes)
    # Adjust these indices [1, 2] if your model's lane ordering is different (e.g., [0, 1, 2, 3])
    row_lane_idx = [1, 2]
    # Define which lanes are primarily defined by col anchors (typically outer lanes)
    col_lane_idx = [0, 3]

    # Process row anchors (lanes 1 and 2 in this example)
    for i in row_lane_idx:
        lane_points = []
        # Check if the lane likely exists by summing valid anchor points
        # The threshold (num_cls_row / 2) can be adjusted based on desired sensitivity (Changed from /4 to /2 like trt_infer)
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(num_cls_row): # Iterate through each row anchor (y-position)
                if valid_row[0, k, i]: # If this anchor point is predicted to exist for this lane
                    # Get the grid cell index with the highest probability
                    max_idx = max_indices_row[0, k, i]
                    # Define a small window around the max index for refinement (local_width=1 in demo)
                    local_width = 1
                    start = max(0, max_idx - local_width)
                    end = min(num_grid_row - 1, max_idx + local_width) + 1 # Inclusive end for slicing
                    all_indices_in_window = np.arange(start, end)

                    # Calculate refined position using softmax weighted average over the window
                    softmax_probs = softmax(loc_row[0, start:end, k, i])
                    # Weighted average of indices + 0.5 for center alignment within the grid cell
                    refined_grid_pos = (softmax_probs * all_indices_in_window).sum() + 0.5

                    # Map the refined grid position (0 to num_grid_row-1) back to the original image width
                    x_coord = (refined_grid_pos / (num_grid_row - 1)) * ori_img_w
                    # Get the y-coordinate from the row anchor (already normalized relative to original height) and scale by original height
                    y_coord = row_anchor[k] * ori_img_h
                    lane_points.append((int(round(x_coord)), int(round(y_coord))))
            if lane_points: # Only add the list of points if any were found for this lane
                coords.append(lane_points)

    # --- Process Columns (Vertical location prediction for each col anchor) ---
    # Find the grid cell index with the maximum probability for each col anchor and lane
    max_indices_col = loc_col.argmax(1) # Shape: (batch, num_cls_col, num_lane_col)
    # Determine if each col anchor exists for each lane
    valid_col = exist_col.argmax(1)     # Shape: (batch, num_cls_col, num_lane_col)

    # Process col anchors (lanes 0 and 3 in this example)
    for i in col_lane_idx:
        lane_points = []
        # Check if the lane likely exists
        if valid_col[0, :, i].sum() > num_cls_col / 4: # Threshold can be adjusted
            for k in range(num_cls_col): # Iterate through each col anchor (x-position)
                if valid_col[0, k, i]: # If this anchor point is predicted to exist
                    max_idx = max_indices_col[0, k, i]
                    local_width = 1
                    start = max(0, max_idx - local_width)
                    end = min(num_grid_col - 1, max_idx + local_width) + 1
                    all_indices_in_window = np.arange(start, end)

                    softmax_probs = softmax(loc_col[0, start:end, k, i])
                    refined_grid_pos = (softmax_probs * all_indices_in_window).sum() + 0.5

                    y_coord = (refined_grid_pos / (num_grid_col - 1)) * ori_img_h
                    # --- End simpler scaling ---

                    # Get the x-coordinate from the col anchor (already normalized relative to original width) and scale by original width
                    x_coord = col_anchor[k] * ori_img_w
                    lane_points.append((int(round(x_coord)), int(round(y_coord))))
            if lane_points: # Only add if points were found
                coords.append(lane_points)

    # Sort coordinates within each lane based on y-coordinate (bottom-up) for consistent drawing
    for lane in coords:
        lane.sort(key=lambda p: p[1], reverse=True) # Sort by y, descending (bottom first)

    return coords

def draw_lanes_v2(image, coords, color=(0, 255, 0), radius=5, thickness=-1, connect_points=True):
    """Draws lane coordinates onto the image, optionally connecting points."""
    vis_img = image.copy()
    for lane in coords:
        # Draw points
        for coord in lane:
            # Ensure coordinate is within image bounds before drawing
            if 0 <= coord[0] < vis_img.shape[1] and 0 <= coord[1] < vis_img.shape[0]:
                cv2.circle(vis_img, coord, radius, color, thickness)

        # # Connect points within the lane
        # if connect_points and len(lane) > 1:
        #     for i in range(len(lane) - 1):
        #          # Ensure both points are valid before drawing line
        #          p1 = lane[i]
        #          p2 = lane[i+1]
        #          if (0 <= p1[0] < vis_img.shape[1] and 0 <= p1[1] < vis_img.shape[0] and
        #              0 <= p2[0] < vis_img.shape[1] and 0 <= p2[1] < vis_img.shape[0]):
        #             cv2.line(vis_img, p1, p2, color, thickness=max(1, thickness // 2 + 1)) # Thinner line

    return vis_img

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)