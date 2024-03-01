import os
import torch
import numpy as np
import yaml
from typing import Any
from torchvision.transforms.functional import normalize
from pathlib import Path
import copy


class Detector:
    #########   Factory method   ############
    _registry = {}

    def __init_subclass__(cls, model, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[model] = cls

    def __new__(cls, model: str):
        subclass = cls._registry[model]
        obj = object.__new__(subclass)
        return obj

    ######### Factory method end ############

    def __init__(self, model: str, **kwargs) -> None:
        """Create DetectModel with specific model

        Args:
            model (str): type of model
        """
        self.model: str = model
        config_path = os.environ.get('LM_CONFIG')
        with open(config_path, 'r') as f:
            local_config = yaml.load(f, Loader=yaml.FullLoader)
        extra = local_config.get(model, {})
        self.kwargs = kwargs | extra
        self.ckpt_dir = os.environ.get('CKPT_DIR', '/opt/ckpts')

    def infer(self, image: np.ndarray, prompt: Any, **kwargs) -> dict:
        # image shape H，W，C
        # SAM_prompt : numpy.ndarray  ,shape (1,2)
        raise NotImplementedError
        return {
            # list[np.ndarray(center_h, center_w, height, width)]
            "bbox": None,
            "text": None,  # str
            "score": None  # list[float | np.ndarray]
        }


class GroundingDino(Detector, model='grounding-dino'):

    def __init__(self, model, **kwargs: Any) -> None:
        super().__init__(model)
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict

        # 初始化GroundingDino模型
        self.args = SLConfig.fromfile(os.path.join(self.ckpt_dir, self.kwargs['config']))
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = build_model(self.args)
        checkpoint = torch.load(os.path.join(self.ckpt_dir, self.kwargs['ckpt']), map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        device = self.args.device
        self.model = self.model.to(device)

        # 初始化推理参数
        self.box_threshold = 0.2
        self.text_threshold = 0.2
        self.token_spans = None
        self.with_logits = False

    def infer(self, image: np.ndarray, prompt: str, **kwargs: Any) -> dict:
        # GroundingDino模型的推理逻辑
        image = np.transpose(np.array(image), (2, 0, 1))  # C，H，W
        self.image = normalize(torch.tensor(image, dtype=torch.float), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.prompt = prompt
        boxes_filt, pred_phrases, logits = self.get_grounding_output()
        boxes_filt = boxes_filt[:, [1, 0, 3, 2]]
        boxes_filt[:, 0] *= image.shape[1]
        boxes_filt[:, 1] *= image.shape[2]
        boxes_filt[:, 2] *= image.shape[1]
        boxes_filt[:, 3] *= image.shape[2]
        if len(logits) != 0:
            max_idx = np.argmax(logits)
            return {
                "bbox": boxes_filt[max_idx],  # list[np.ndarray(center_h, center_w, height, width)]
                "text": pred_phrases[max_idx],  # list[str]
                "score": logits[max_idx],  # list[float | np.ndarray]
            }
        else:
            return {"bbox": torch.tensor(np.array([0., 0., 0., 0.])), "score": 0.}

    def get_grounding_output(self):
        from groundingdino.util.utils import get_phrases_from_posmap
        from groundingdino.util.vl_utils import create_positive_map_from_span

        assert self.text_threshold is not None or self.token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
        caption = self.prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.image = self.image.to(self.args.device)
        with torch.no_grad():
            outputs = self.model(self.image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        score = []
        if self.token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = self.model.tokenizer
            tokenized = tokenlizer(caption)

            # build pred
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
                score.append(logits.max().item())
                if self.with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(self.model.tokenizer(self.prompt),
                                                          token_span=self.token_spans).to(
                                                              self.image.device)  # n_phrase, 256

            logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(self.token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > self.box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                logit_phr_num = logit_phr[filt_mask]
                score.append([logit for logit in logit_phr_num])
                if self.with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases

        return boxes_filt, pred_phrases, score


class YoloV7(Detector, model='yolo-v7'):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        from yolov7.models.experimental import attempt_load
        from yolov7.utils.general import check_img_size
        from yolov7.utils.torch_utils import select_device
        if torch.cuda.is_available():
            self.device = select_device("0")
        else:
            self.device = select_device("cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        path = os.path.join(self.ckpt_dir, self.kwargs['ckpt'])
        # self.model = torch.load(path,map_location=self.device)
        self.model = attempt_load(path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(640, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, self.imgsz,
                            self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
    def infer(self, image: np.array ,prompt: Any, **kwargs) -> dict:
        from yolov7.utils.datasets import LoadImages
        from yolov7.utils.general import non_max_suppression, scale_coords
        dict = {3: "mug", 2: "microwave_door", 1: "cabinet_handle", 0: "bowl"}
        bbox = []
        score = []
        # dataset = LoadImages(image, img_size=self.imgsz, stride=self.stride)
        # path, img, im0s, vid_cap = next(iter(dataset))
        img = torch.from_numpy(image).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)  # change dimension order to (batch_size, channels, height, width)
            import torch.nn.functional as F
            img = F.interpolate(img, size=(224, 224))
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.2, 0.45, classes=None, agnostic=False)
        det = pred[0]
        text = None
        max_conf = -1
        max_conf_bbox = []
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls = int(cls.item())
                if dict[cls] == prompt:
                    text = prompt
                    xyxy = torch.stack(xyxy).cpu().numpy()
                    bbox.append(xyxy)
                    conf = conf.cpu().numpy()
                    score.append(conf)
                    if conf > max_conf:
                        max_conf = conf
                        max_conf_bbox = xyxy
        _conf_bbox =torch.tensor(max_conf_bbox)
        return {
            "bbox": _conf_bbox,  # list[np.ndarray(center_h, center_w, height, width)]
            "text": text,  # str
            "score": max_conf  # list[float | np.ndarray]
        }


class Segmentor:
    #########   Factory method   ############
    _registry = {}

    def __init_subclass__(cls, model, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[model] = cls

    def __new__(cls, model: str):
        subclass = cls._registry[model]
        obj = object.__new__(subclass)
        return obj

    ######### Factory method end ############

    def __init__(self, model, **kwargs) -> None:
        """Create DetectModel with specific model

        Args:
            model (str): type of model
        """
        self.model: str = model
        config_path = os.environ.get('LM_CONFIG')
        with open(config_path, 'r') as f:
            local_config = yaml.load(f, Loader=yaml.FullLoader)
        extra = local_config.get(model, {})
        self.kwargs = kwargs | extra
        self.ckpt_dir = os.environ.get('CKPT_DIR', '/opt/ckpts')

    def infer(self, image: np.ndarray, prompt: Any, **kwargs) -> dict:
        # image shape H，W，C
        # SAM_prompt : numpy.ndarray  ,shape (1,2)
        raise NotImplementedError
        return {
            "mask": None,  # np.ndarray
            "score": None  # list[float | np.ndarray]
        }


class SegmentAnything(Segmentor, model='segment-anything'):

    def __init__(self, model, **kwargs: Any) -> None:
        super().__init__(model)

        from segment_anything_fast import sam_model_registry
        from segment_anything_fast import SamPredictor

        # 初始化 SAM 模型
        self.sam = sam_model_registry[self.kwargs['model_type']](
            checkpoint=os.path.join(self.ckpt_dir, self.kwargs['ckpt'])).to('cuda')
        self.predictor = SamPredictor(self.sam)
        # 初始化推理参数
        self.label = np.array([1])  # 标签1(前景点)或0(背景点)
        self.multimask_output = False

    def infer(self, image: np.ndarray, prompt: np.ndarray, **kwargs: Any) -> dict:
        # SAM模型的推理逻辑
        self.image = image
        self.predictor.set_image(image)
        self.prompt = copy.deepcopy(prompt)
        if isinstance(self.prompt, torch.Tensor):
            self.prompt = self.prompt.numpy()
        if self.prompt.shape[-1] == 2:
            masks, scores, logits = self.predictor.predict(
                point_coords=self.prompt,
                point_labels=self.label,
                multimask_output=self.multimask_output,
            )
        elif self.prompt.shape[-1] == 4:
            self.prompt[:2] -= self.prompt[2:] / 2
            self.prompt[2:] += self.prompt[:2]
            bbox = self.bbox_with_gap(self.prompt, gap=15)
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]]),
                point_labels=self.label,
                box=bbox,
                multimask_output=self.multimask_output,
            )
        else:
            raise NotImplementedError('Prompt type not supported')
        max_idx = np.argmax(scores)

        return {
            "mask": masks[max_idx].astype(bool),  # np.ndarray
            "score": scores[max_idx],  # str
        }

    def bbox_with_gap(self, bbox, gap):
        width = self.image.shape[1]
        height = self.image.shape[0]
        gapped_bbox = np.zeros(4)
        gapped_bbox[0] = np.maximum(0, bbox[0] - gap)
        gapped_bbox[1] = np.maximum(0, bbox[1] - gap)
        gapped_bbox[2] = np.minimum(width, bbox[2] + gap)
        gapped_bbox[3] = np.minimum(height, bbox[3] + gap)
        return bbox
