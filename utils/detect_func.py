#%%
import argparse
from sys import platform

from utils.models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

#%%
def detectTable(opt):
    with torch.no_grad():
        img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
        #webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        webcam=False

        # Initialize
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Initialize model
        model = Darknet(opt.cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()
        # torch_utils.model_info(model, report='summary')  # 'full' or 'summary'

        # Eval mode
        model.to(device).eval()

        # Export mode
        if ONNX_EXPORT:
            model.fuse()
            img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
            f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
            torch.onnx.export(model, img, f, verbose=False, opset_version=11)

            # Validate exported model
            import onnx
            model = onnx.load(f)  # Load the ONNX model
            onnx.checker.check_model(model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
            return

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=img_size)
        else:
            save_img = False
            dataset = LoadImages(source, img_size=img_size)

        # Get names and colors
        names = load_classes(opt.names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        results=""
        for path, img, im0s, vid_cap in dataset:
            t = time.time()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img)[0].float() if half else model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            # with open(save_path + '.txt', 'a') as file:
                            #     file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                            results+=('%g ' * 6 + '\n') % (*xyxy, cls, conf)
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
        return results

#%%
class parameters:
    def __init__(self, img):
        self.cfg="utils/yolov3-tiny_table.cfg"
        self.names="utils/table.names"
        self.weights="utils/best_v2.weights"
        self.source=img
        self.output="outputs/"
        self.img_size=416
        self.conf_thres=0.2
        self.iou_thres=0.6
        self.fourcc='mp4v'
        self.half=False
        self.device="cpu"
        self.view_img=False
        self.save_txt=True
        self.classes=None
        self.agnostic_nms=False

