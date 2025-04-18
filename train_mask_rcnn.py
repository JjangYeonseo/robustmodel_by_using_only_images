import os
import torch
import psutil
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog

# === 경로 설정 ===
dataset_dir = r"C:\Users\dadab\Desktop\Sample"
image_dir = os.path.join(dataset_dir, "unified_images")
coco_dir = os.path.join(dataset_dir, "coco_output")
output_dir = os.path.join(dataset_dir, "output_model")
os.makedirs(output_dir, exist_ok=True)

# === COCO 데이터셋 등록 ===
register_coco_instances("my_train", {}, os.path.join(coco_dir, "train.json"), image_dir)
register_coco_instances("my_val", {}, os.path.join(coco_dir, "val.json"), image_dir)
register_coco_instances("my_test", {}, os.path.join(coco_dir, "test.json"), image_dir)

# === 클래스 이름 등록 (train.json의 categories 순서에 맞춤) ===
thing_classes = [
    "ground", "sky", "vegetation", "building", "road", "car", "sidewalk", "pole", "static", "cargroup",
    "person", "traffic sign", "truck", "wall", "guard rail", "terrain", "fence", "bridge", "tunnel",
    "bus", "traffic light", "dynamic", "parking", "rider", "bicycle", "motorcycle", "trailer"
]

# 메타데이터 등록
for d in ["my_train", "my_val", "my_test"]:
    MetadataCatalog.get(d).thing_classes = thing_classes

# === 배치 사이즈 자동 결정 함수 ===
def get_batch_size_by_vram(vram_gb):
    if vram_gb >= 10:
        return 6
    elif vram_gb >= 8:
        return 4
    elif vram_gb >= 6:
        return 2
    else:
        return 1

# GPU VRAM 감지
def detect_vram_gb():
    if torch.cuda.is_available():
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        return round(vram_bytes / 1024**3)
    return 0

# === Config 설정 ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_train",)
cfg.DATASETS.TEST = ("my_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.OUTPUT_DIR = output_dir

# 자동 VRAM 기반 배치 크기 설정
vram_gb = detect_vram_gb()
cfg.SOLVER.IMS_PER_BATCH = get_batch_size_by_vram(vram_gb)
print(f"✅ Detected VRAM: {vram_gb} GB → 배치 크기 자동 설정: {cfg.SOLVER.IMS_PER_BATCH}")

# 학습 하이퍼파라미터
cfg.SOLVER.BASE_LR = 0.0002
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (7000, 9000)
cfg.SOLVER.GAMMA = 0.1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
cfg.TEST.EVAL_PERIOD = 1000

# CUDA 설정
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 트레이너 정의 ===
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# === 학습 시작 ===
if __name__ == "__main__":
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
