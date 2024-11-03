import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np

from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter, LoadImages
)

def count_lanes(lane_mask, min_points=100, min_confidence=0.3):
    """
    Count the number of distinct lanes in the lane mask
    Args:
        lane_mask: Binary mask of lane lines
        min_points: Minimum number of points to consider as a valid lane
        min_confidence: Minimum confidence threshold for lane detection
    Returns:
        num_lanes: Number of detected lanes
        processed_mask: Processed lane mask with labeled lanes
    """
    # Convert to numpy array if it's a tensor
    if torch.is_tensor(lane_mask):
        lane_mask = lane_mask.cpu().numpy()
    
    # Ensure the mask is in the correct format (single channel, 8-bit)
    if len(lane_mask.shape) > 2:
        # If we have multiple channels, take the first one
        lane_mask = lane_mask[0] if lane_mask.shape[0] == 1 else lane_mask[:,:,0]
    
    # Normalize to 0-255 range if not already
    if lane_mask.max() <= 1.0:
        lane_mask = (lane_mask * 255).astype(np.uint8)
    else:
        lane_mask = lane_mask.astype(np.uint8)
    
    # Apply threshold
    _, binary_mask = cv2.threshold(lane_mask, int(min_confidence * 255), 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Filter out small components (noise)
    valid_lanes = 0
    processed_mask = np.zeros_like(binary_mask)
    
    # Start from 1 to skip background
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_points:
            valid_lanes += 1
            processed_mask[labels == i] = 255
            
    return valid_lanes, processed_mask

def show_seg_result_with_lanes(img, result, is_demo=False, lane_count=0):
    """
    Modified show_seg_result function to display lane count
    """
    # Get original show_seg_result implementation
    img = show_seg_result(img, result, is_demo=is_demo)
    
    # Add lane count text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'Lanes Detected: {lane_count}'
    cv2.putText(img, text, (20, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return img

def detect():
    # Setting and directories
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')
    
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)

    if half:
        model.half()
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    if source.isnumeric():
        source = int(source)
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # Split for trace model
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        
        # Count lanes
        num_lanes, processed_lane_mask = count_lanes(ll_seg_mask)

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.realtime:
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Show results
            if opt.realtime:
                # Use modified show_seg_result function with lane count
                im0 = show_seg_result_with_lanes(im0, (da_seg_mask, processed_lane_mask), is_demo=True, lane_count=num_lanes)
                cv2.imshow('Real-time Detection', im0)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    cv2.destroyAllWindows()
                    return
            else:
                im0 = show_seg_result_with_lanes(im0, (da_seg_mask, processed_lane_mask), is_demo=True, lane_count=num_lanes)

            # Save results
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            # Print frame info
            print(f'{s}Done. ({t2 - t1:.3f}s) - Detected Lanes: {num_lanes}')

        inf_time.update(t2-t1, img.size(0))
        nms_time.update(t4-t3, img.size(0))
        waste_time.update(tw2-tw1, img.size(0))

    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')
    
    # Cleanup
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()
    cv2.destroyAllWindows()

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--realtime', action='store_true', help='enable real-time display')
    return parser

if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)
    
    with torch.no_grad():
        detect()