"""Low-level YOLOv3 inference runner.

This module is part of the vendored YOLOv3 engine (adapted from the
``ultralytics/yolov3`` reference implementation). It exposes a single entry
point, :func:`detect_table`, which loads the Darknet model, runs inference on a
single image and returns the raw detections as a whitespace-separated string of
``x1 y1 x2 y2 cls conf`` rows (one detection per line).

Higher level orchestration (PDF rendering, coordinate mapping, Camelot
extraction) lives in :mod:`pdf_table_extractor.pipeline`.
"""

from .models import *  # noqa: F401,F403  (set ONNX_EXPORT in models.py)
from .datasets import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403


def detect_table(opt):
    """Run YOLOv3 inference and return detections as a text block.

    ``opt`` is a duck-typed configuration object exposing the attributes
    produced by :class:`pdf_table_extractor.config.YoloConfig` (``cfg``,
    ``names``, ``weights``, ``source``, ``output``, ``img_size``,
    ``conf_thres``, ``iou_thres``, ``half``, ``device``, ``classes``,
    ``agnostic_nms``).
    """
    with torch.no_grad():
        img_size = opt.img_size
        out, source, weights, half = opt.output, opt.source, opt.weights, opt.half

        # Initialize
        device = torch_utils.select_device(device=opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Initialize model
        model = Darknet(opt.cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith(".pt"):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)["model"])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Eval mode
        model.to(device).eval()

        # Half precision (only supported on CUDA)
        half = half and device.type != "cpu"
        if half:
            model.half()

        # Dataloader for a single image / directory of images
        dataset = LoadImages(source, img_size=img_size)

        # Class names
        names = load_classes(opt.names)

        results = ""
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img)[0].float() if half else model(img)[0]

            # Non-max suppression
            pred = non_max_suppression(
                pred,
                opt.conf_thres,
                opt.iou_thres,
                classes=opt.classes,
                agnostic=opt.agnostic_nms,
            )

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = im0s
                if det is not None and len(det):
                    # Rescale boxes from img_size to original image size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det:
                        results += ("%g " * 6 + "\n") % (*xyxy, cls, conf)

        return results
