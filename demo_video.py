import argparse
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from scipy import signal

from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box
)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path')
    parser.add_argument('--video', type=str, required=True, help='path to input video file')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser

def preprocess_image(frame, device, img_size, half=False):
    """
    Preprocess image for model input with proper aspect ratio preservation
    """
    # Calculate scaling factor while maintaining aspect ratio
    h, w = frame.shape[:2]
    ratio = img_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # Resize image maintaining aspect ratio
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create square image with padding
    square_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Calculate padding
    dx = (img_size - new_w) // 2
    dy = (img_size - new_h) // 2
    # Place resized image in center
    square_img[dy:dy+new_h, dx:dx+new_w] = resized
    
    # Convert to RGB and normalize
    square_img = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
    square_img = square_img.transpose(2, 0, 1)
    square_img = np.ascontiguousarray(square_img)
    
    # Convert to tensor
    img = torch.from_numpy(square_img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img, ratio, (dx, dy)

def show_seg_result(frame, masks, ratio, padding, is_demo=False):
    """
    Show segmentation results with proper alignment and mask handling
    """
    da_mask, ll_mask = masks
    h, w = frame.shape[:2]
    
    # Convert masks to numpy arrays if they're tensors
    if isinstance(da_mask, torch.Tensor):
        da_mask = da_mask.cpu().numpy()
    if isinstance(ll_mask, torch.Tensor):
        ll_mask = ll_mask.cpu().numpy()
    
    # Ensure masks are 2D arrays
    if da_mask.ndim > 2:
        da_mask = da_mask.squeeze()
    if ll_mask.ndim > 2:
        ll_mask = ll_mask.squeeze()
    
    # Extract the valid portion of the masks using padding info
    dx, dy = padding
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    
    # Ensure masks are in the correct format for resize
    da_mask = da_mask.astype(np.float32)
    ll_mask = ll_mask.astype(np.float32)
    
    # Extract valid portions and resize
    da_mask_valid = da_mask[dy:dy+new_h, dx:dx+new_w]
    ll_mask_valid = ll_mask[dy:dy+new_h, dx:dx+new_w]
    
    # Resize masks to match frame dimensions
    da_mask_resized = cv2.resize(da_mask_valid, (w, h))
    ll_mask_resized = cv2.resize(ll_mask_valid, (w, h))
    
    # Create color masks
    da_mask_color = np.zeros_like(frame)
    ll_mask_color = np.zeros_like(frame)
    
    # Set colors for driving area (green) and lane lines (red)
    da_mask_color[da_mask_resized > 0.5] = [0, 255, 0]
    ll_mask_color[ll_mask_resized > 0.5] = [0, 0, 255]
    
    # Blend masks with original frame
    blend = cv2.addWeighted(frame, 1, da_mask_color, 0.3, 0)
    blend = cv2.addWeighted(blend, 1, ll_mask_color, 0.3, 0)
    
    return blend

def count_lanes(ll_seg_mask, frame_height, frame_width):
    """
    Count the number of lanes on left and right sides
    """
    # Resize mask to frame dimensions
    ll_seg_mask = cv2.resize(ll_seg_mask, (frame_width, frame_height))
    
    # Convert mask to binary image
    binary_mask = (ll_seg_mask > 0.5).astype(np.uint8) * 255
    
    # Define regions for left and right sides
    mid_point = frame_width // 2
    height_point = int(frame_height * 0.7)  # Analysis point at 70% of height
    
    # Get horizontal profile at the analysis height
    horizontal_profile = binary_mask[height_point, :]
    
    # Split into left and right sides
    left_profile = horizontal_profile[:mid_point]
    right_profile = horizontal_profile[mid_point:]
    
    # Find peaks (lane markings) with minimum distance
    min_distance = frame_width // 20  # Minimum distance between lanes
    
    left_peaks, _ = signal.find_peaks(left_profile, distance=min_distance, height=50)
    right_peaks, _ = signal.find_peaks(right_profile, distance=min_distance, height=50)
    
    return len(left_peaks), len(right_peaks)

def draw_lane_count(frame, left_lanes, right_lanes):
    """
    Draw lane count information on frame
    """
    cv2.putText(frame, f'Left Lanes: {left_lanes}', 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Right Lanes: {right_lanes}', 
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def process_video():
    cap = cv2.VideoCapture(opt.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {opt.video}")
        return

    window_name = 'Lane Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    device = select_device(opt.device)
    model = torch.jit.load(opt.weights)
    model = model.to(device)
    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()

    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    # Print initial frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video dimensions: {frame_width}x{frame_height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file")
            break

        orig_frame = frame.copy()
        frame_count += 1
        t1 = time_synchronized()

        # Model inference with aspect ratio preservation
        img, ratio, padding = preprocess_image(frame, device, opt.img_size, half)
        [pred, anchor_grid], seg, ll = model(torch.zeros(1, 3, img, img).to(device).type_as(next(model.parameters())))  # run once

        pred = split_for_trace_model(pred, anchor_grid)
        
        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=False
        )
        t2 = time_synchronized()
        # Process detections and segmentation
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        
        # Debug print for mask shapes and types
        if frame_count == 1:
            print(f"DA mask shape: {da_seg_mask.shape}, type: {type(da_seg_mask)}")
            print(f"LL mask shape: {ll_seg_mask.shape}, type: {type(ll_seg_mask)}")
        
        # Count lanes
        h, w = frame.shape[:2]
        left_lanes, right_lanes = count_lanes(ll_seg_mask, h, w)

        try:
            # Show segmentation result with proper alignment
            frame = show_seg_result(orig_frame, (da_seg_mask, ll_seg_mask), ratio, padding)
            frame = draw_lane_count(frame, left_lanes, right_lanes)
        except Exception as e:
            print(f"Error in mask processing: {e}")
            frame = orig_frame  # Use original frame if mask processing fails
        
        # Calculate and display FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
            
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback stopped by user")
            break

       
        if frame_count % 30 == 0:
            print(f'Frame {frame_count}, Time: {(t2-t1):.3f}s, FPS: {fps:.1f}, '
                  f'Lanes: Left={left_lanes}, Right={right_lanes}')

    cap.release()
    cv2.destroyAllWindows()
    print(f'\nProcessing completed: {frame_count} frames processed')   
if __name__ == '__main__':
    opt = make_parser().parse_args()
    with torch.no_grad():
        process_video()