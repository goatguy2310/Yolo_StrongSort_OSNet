import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms, models
from numpy import random


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, xywh2xyxy, box_iou, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov7.utils.plots import plot_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

def classify(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (256, 256))  # BGR
                cv2.imwrite('test%i.jpg' % j, im)

                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # BGR to RGB
                im = cv2.transpose(im, (2, 0, 1)) # transpose to 3x256x256
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(np.array(ims)).to(d.device)).argmax(1)  # classifier prediction
            real = [3.0, 4.0, 6.0, 7.0]
            for k, j in enumerate(x[i]):
              j[-1] = pred_cls2[k] = real[pred_cls2[k]]
            print('\nPRED:', pred_cls2)
    return x

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        classify_weights=None, # second stage classifier model.pt path,
        classify_name=None, # classifier model's name
        fall_thres=30, # threshold for fall detection
        heartatt_thres=100, # threshold for heart attack detection
        heartatt_max_thres=150, # maximum frame threshold for heart attack detection
        lying_thres=150, # frame threshold for lying detection
        lying_iou_thres=0.5, # iou between bed and sofa threshold for lying detection
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        classesnotrack=None, # filter out classes that don't need to be tracked: --classesnotrack 0, or --classesnotrack 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = Path(save_dir)
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    # model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max()  # model stride
    imgsz = check_img_size(imgsz[0], s=stride.cpu().numpy())  # check image size

    if classify_weights is not None and classify_name is not None:
        modelc = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        modelc.classifier[-1] = nn.Linear(1280, 4)
        modelc.to(device)
        modelc.load_state_dict(torch.load(str(classify_weights), map_location=device))
        modelc.eval()

    # Stable pose life cycle
    # |Stable lb1, stab = lb1, state = stable| -> |First lb2 detection, stab = -1, state = unstable| -> |After x frames and y same lb2 labels, stab = lb2, state = stable|
    #                                               |                                                           |
    #                        |More than z detection != label, stab = -1, state = unstable| <---------------------
    #
    

    st = [-1] * 100 # Stable states of objects
    lst = [[-1, -1]] * 100 # Last state of stable (cls, framecnt (-1 = never))
    ft = [[]] * 100 # Last 300 frames of whether a person is faint or not
    stab_q = {} # Class queue of ids
    frame_thres, stable_thres, other_thres = 30, 15, 5
    fall_thres = int(fall_thres)
    heartatt_thres = int(heartatt_thres)
    heartatt_max_thres = int(heartatt_max_thres)
    lying_thres = int(lying_thres)
    lying_iou_thres = float(lying_iou_thres)
    print('fall_thres: ', fall_thres)
    print('heartatt_thres: ', heartatt_thres)
    print('heartatt_max_thres: ', heartatt_max_thres)
    print('lying_thres: ', lying_thres)
    print('lying iou_thres: ', lying_iou_thres)

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride.cpu().numpy())
        nr_sources = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run tracking
    dt, seen, frame_cnt = [0.0, 0.0, 0.0, 0.0, 0.0], 0, 0
    start_time = time_synchronized()
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        frame_cnt += 1
        s = ''
        t1 = time_synchronized()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_synchronized()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im)
        t3 = time_synchronized()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms)
        dt[2] += time_synchronized() - t3
        
        # Filtering out classes that don't need to be tracked
        x = []
        if classesnotrack is not None:
            for i, p in enumerate(pred):
              trackingclasses = ~(p[:, 5:6] == torch.tensor(classesnotrack, device=p.device)).any(1)
              x.append(p[~trackingclasses])
              pred[i] = p[trackingclasses]

        # Applying second stage classifier
        if classify_weights is not None and classify_name is not None:
            t4 = time_synchronized()
            pred = classify(pred, modelc, im, im0s)
            dt[3] += time_synchronized() - t4

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            x[i][:, :4] = scale_coords(im.shape[2:], x[i][:, :4], im0.shape).round()
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_synchronized()
                dt[4] += t5 - t4

                # processing stableness of existing tracks
                de = {}
                sbboxes = {}
                for j in outputs[i]:
                    de[int(j[4])] = int(j[5])
                    sbboxes[int(j[4])] = j[0:4]
                print(de)

                for j in dict(stab_q):
                    if j in de:
                        targ = de[j]
                        stab_q[j].append(targ)
                    else:
                        stab_q[j].append(-1)
                    if len(stab_q[j]) > heartatt_max_thres: stab_q[j].pop(0)

                    ok = False
                    stab = 0
                    oth = 0
                    for it, k in enumerate(reversed(stab_q[j])):
                        if it >= frame_thres: break
                        if k != -1:
                            ok = True

                        if k == targ:
                            stab += 1
                            if stab >= stable_thres:
                                break
                        elif k != -1 and k != targ:
                            oth += 1
                            if oth >= other_thres:
                                stab = -1
                                break
                    
                    print(lst[j])
                    if not ok and len(stab_q[j]) >= frame_thres: 
                        stab_q.pop(j, None)
                        st[j] = -1
                        lst[j] = [-1, -1]
                        continue
                    
                    if stab >= stable_thres: 
                        st[j] = targ
                    else:
                        if st[j] != -1: lst[j] = [st[j], 0]
                        if lst[j][1] != -1: lst[j][1] += 1
                        st[j] = -1

                    # Checking if patient is lying
                    if st[j] == 3 and j in sbboxes:
                        ok = False
                        for k in x[i]:
                            if k[5] == 0 or k[5] == 5:
                                box1 = k[:4].cpu().numpy().astype(np.int32)
                                box2 = sbboxes[j]
                                print(box1, box2)
                                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

                                # Intersection area
                                a = min(b1_x2, b2_x2) - max(b1_x1, b2_x1)
                                if a < 0: a = 0
                                b = min(b1_y2, b2_y2) - max(b1_y1, b2_y1)
                                if b < 0: b = 0
                                inter = a * b

                                # Union Area
                                # w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
                                w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
                                # union = w1 * h1 + w2 * h2 - inter

                                iou = inter / (w2 * h2)
                                if iou > lying_iou_thres:
                                    print('MAYBE FAINT: ', iou)
                                    with open(txt_path + '_lie.txt', 'a') as f:
                                        f.write(f'MAYBE FAINT: {iou} | Thres: {lying_iou_thres}\n')
                                    ok = True
                        if not ok: ft[j].append(1)
                        else: ft[j].append(0)
                        if len(ft[j]) > lying_thres: ft[j].pop(0)
                        if sum(ft[j]) >= lying_thres * 95 / 100:
                            print('HUMAN FAINT')
                            with open('./lie.txt', 'a') as f:
                                f.write(txt_path + '\n')
                            with open(txt_path + '_lie.txt', 'a') as f:
                                f.write(f'{sum(ft[j])} | Thres: {lying_thres}\n')

                    # Checking if patient is falling
                    if st[j] == 3 and lst[j][0] == 6 and lst[j][1] <= fall_thres:
                        print('HUMAN HAS FALLEN')
                        with open('./fall.txt', 'a') as f:
                            f.write(txt_path + '\n')
                        with open(txt_path + '_fall.txt', 'a') as f:
                            f.write(f'{lst[j][1]} | Thres: {fall_thres}\n')
                    
                    # Checking if patient is having a heart attack
                    heart = 0
                    for k in reversed(stab_q[j]):
                        if k == 2: heart += 1
                        if heart >= heartatt_thres:
                            print('HUMAN IS HAVING A HEART ATTACK')
                            with open('./heart.txt', 'a') as f:
                                f.write(txt_path + '\n')
                            with open(txt_path + '_heart.txt', 'a') as f:
                                f.write(f'{heart} | Thres: {heartatt_thres}\n')
                            break

                # Print results
                pp = ''
                for j in outputs[i]:
                    pp += f"{int(j[4])} {names[int(j[5])]} ({'last stable: ' + str(lst[int(j[4])][1]) if st[int(j[4])] == -1 else 'stable'}), "  # add to string
                s += pp
                with open(txt_path + '_st.txt', 'a') as f:
                    f.write(pp + '\n')

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = int(output[4])
                        cls = int(output[5])

                        # processing stableness of new track
                        if id not in stab_q:
                            stab_q[id] = [cls]
                            st[id] = -1
                            lst[id] = [cls, -1]

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f"{id} ({'last stable: ' + str(lst[id][1]) if st[id] == -1 else 'stable'}) {names[c]}" if hide_conf else \
                                (f"{id} ({'last stable: ' + str(lst[id][1]) if st[id] == -1 else 'stable'}) {conf:.2f}" if hide_class else \
                                f"{id} ({'last stable: ' + str(lst[id][1]) if st[id] == -1 else 'stable'}) {names[c]} {conf:.2f}"))
                            plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s), Classifier:({dt[3]:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')
                
            if len(x) > 0 and x[i] is not None and len(x[i]):
                if save_vid or save_crop or show_vid:
                    # draw boxes for untracked objects
                    for j, (x1, y1, x2, y2, conf, cls) in enumerate(x[i]):
                        c = int(cls)
                        bboxes = [x1, y1, x2, y2]
                        label = None if hide_labels else (f'Object: {names[c]}')
                        plot_one_box(bboxes, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        if save_crop:
                            txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                            save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        t6 = time_synchronized()
        print(f'Total time: {t6 - t1:.3f}s, {(1 / (t6 - t1)):.3f} fps')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(t)
    avg = (time_synchronized() - start_time) / frame_cnt # average time per frame
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms second stage classifier, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    print(f'Average time: {avg:.3f}s, {(1 / avg):.3f} fps')
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--classify-weights', type=str, default=None, help='second stage classifier model.pt path')
    parser.add_argument('--classify-name', type=str, default=None, help='second stage classifier\'s name')
    parser.add_argument('--fall-thres', type=str, default=30, help='fall threshold')
    parser.add_argument('--heartatt-thres', type=str, default=100, help='heart attack threshold')
    parser.add_argument('--heartatt-max-thres', type=str, default=150, help='heart attack max frame threshold')
    parser.add_argument('--lying-thres', type=str, default=150, help='lying threshold')
    parser.add_argument('--lying-iou-thres', type=str, default=0.5, help='lying iou with bed or sofa threshold')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--classesnotrack', nargs='+', type=int, help='filter out classes that don\'t need to be tracked: --classesnotrack 0, or --classesnotrack 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)