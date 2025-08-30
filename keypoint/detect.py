# this python file is used to import, not for running directly
import os
import time

import cv2
import numpy as np
import torch

from .models.experimental import attempt_load
from .utils.datasets import LoadImages, LoadStreams
from .utils.general import (check_img_size, check_requirements,
                           non_max_suppression, save_one_box, scale_coords,
                           xyxy2xywh)
from .utils.plots import colors, plot_one_box
from .utils.torch_utils import select_device


def letterbox(
    im, 
    new_shape=(640, 640), 
    color=(114, 114, 114), 
    auto=True, 
    scaleFill=False, 
    scaleup=True, 
    stride=64
):
    """
    im: 传入的原图(BGR格式)
    new_shape: 想要缩放到的大小 (w, h)
    color: 填充的颜色
    auto: 是否自动根据 stride 对缩放后的图像进行适配
    scaleFill: 是否强行拉伸图片到 new_shape（不保持原始宽高比）
    scaleup: 是否允许放大
    stride: 步长，用于保证输出的宽高是 stride 的整数倍
    """
    # 当前图像的原始大小
    shape = im.shape[:2]  # (h, w)

    # 想要得到的(w, h)
    new_w, new_h = new_shape

    # 计算缩放比例
    r = min(new_w / shape[1], new_h / shape[0])
    if not scaleup:
        # 如果不允许放大，那么当原图比 target 小就不缩放
        r = min(r, 1.0)
    
    # 计算缩放后图像的尺寸
    ratio = r, r  # w, h 同比缩放
    unpad_w, unpad_h = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 缩放
    im = cv2.resize(im, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    # 计算需要填充的像素 (dw, dh)
    dw, dh = new_w - unpad_w, new_h - unpad_h
    
    # auto=True时，会进一步根据stride对(dw, dh)进行修正，使输出的wh为stride的整数倍
    if auto:
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)

    # 分成一半一半填充（若 dw, dh 为奇数，可能会有 1 像素的差异）
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    # 用指定颜色进行填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    im = np.transpose(im, (2, 0, 1))

    return im, ratio, (dw, dh)


def key_detect(img, weights, save_dir):
    # convert the image to the shape of (304, 304, 3) and the type is np.ndarray
    img = cv2.resize(img, (304, 304))
    # if the channel of the image is 1, convert it to 3 channels, forexample, (304, 304) -> (304, 304, 3)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # img np array,   weights_path,  save_path
    # get current time and convert to YYYYMMDD_HHMMSS format
    now = time.localtime()

    now_time = time.strftime('%Y%m%d_%H%M%S', now)

    with torch.no_grad():
        # source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
        # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        kpt_label = True
        imgsz = 640
        line_thickness = 2
        hide_labels = False
        hide_conf = False
        conf_thres = 0.65
        iou_thres = 0.25
        # save_dir = './results'
        save_dir_label = os.path.join(save_dir, 'labels')
        # save_dir_label is not exist, create it
        if not os.path.exists(save_dir_label):
            os.makedirs(save_dir_label)
        # if not os.path.exists(save_dir_label)

        # Initialize
        device = select_device('')
        half = True  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half()  # to FP16

        # dataset = LoadImages(source, img_size=imgsz, stride=stride)
        im0 = img
        
        img, ratio, (d, dh) = letterbox(im0, new_shape=(imgsz, imgsz), stride=stride, auto=True)

        # Run inference
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        # for path, img, im0s, vid_cap in dataset:
        # im0s = img
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]
        print(pred[...,4].max())
        # Apply NMS
        # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label,nc=model.yaml['nc'], nkpt=model.yaml['nkpt'])
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, kpt_label=kpt_label,nc=model.yaml['nc'], nkpt=model.yaml['nkpt'])
        det = pred[0]

        save_path = os.path.join(save_dir, now_time + ".jpg")  # img.jpg
        txt_path = os.path.join(save_dir_label, now_time + ".txt")  # img.txt 
        # s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        results = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(det[:,:6]):
                # if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                kpts = det[det_index, 6:] # keypoints
                kpts_coords = kpts.view(-1, 3)[:, :2] / gn[0]  # extract keypoints coordinates and normalize
                kpts_coords = kpts_coords.view(-1).tolist()  # flatten to list
                line = (cls.item(), *xywh, *kpts_coords)  # label format
                # print(line)
                results.append(line)
                with open(txt_path, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # if save_img or opt.save_crop or view_img:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                # kpts = det[det_index, 6:]
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
        # print(results)
        cv2.imwrite(save_path, im0)
        return results



if __name__ == '__main__':

    source = 'test_STM_image/440_15.png'
    weights = 'runs/train/exp2/weights/best.pt'
    save_dir = './results'
    img = cv2.imread(source) 

    key_detect(img, weights, save_dir) # the shape of img is (h, w, c) and the type is np.ndarray
