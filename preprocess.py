# 라벨 통합 및 보정 (라벨에 있는 polygon이 닫혀 있지 않은 경우, 자동으로 첫 점과 끝 점을 이어서 닫아주고, 원본 라벨(label_root)과 증강 라벨(save_label_root)을 하나의 폴더(combined_label_dir)에 복사하면서 polygon이 3개 미만 좌표이거나 안 닫힌 것은 제거 또는 보정
# Labelme 형식 → COCO 포맷으로 변환
# 날씨별 클래스 개수 분석

import os
import json
import csv
import shutil
from tqdm import tqdm
from collections import defaultdict, Counter
import re
import glob
from pathlib import Path
from labelme2coco import convert
from sklearn.model_selection import train_test_split

# === 경로 설정 ===
original_img_dir = r"C:\Users\dadab\Desktop\Sample\01.\uC6D0\uCC9C\uB370\uC774\uD130"
augmented_img_dir = r"C:\Users\dadab\Desktop\Sample\augmented\images"
label_dirs = [
    r"C:\Users\dadab\Desktop\Sample\02.\uB77C\uBCA8\uB9C1\uB370\uC774\uD130",
    r"C:\Users\dadab\Desktop\Sample\augmented\labels"
]

unified_img_dir = r"C:\Users\dadab\Desktop\Sample\unified_images"
final_label_dir = r"C:\Users\dadab\Desktop\Sample\final_labels"
coco_output_dir = r"C:\Users\dadab\Desktop\Sample\coco_output"
csv_output_path = r"C:\Users\dadab\Desktop\Sample\weather_class_distribution.csv"

# === 기존 COCO 폴더 삭제 후 재생성 ===
if os.path.exists(coco_output_dir):
    shutil.rmtree(coco_output_dir)
os.makedirs(coco_output_dir, exist_ok=True)
os.makedirs(unified_img_dir, exist_ok=True)
os.makedirs(final_label_dir, exist_ok=True)

# === polygon 닫기 ===
def close_polygon_if_needed(points):
    if len(points) < 3:
        return None
    if points[0] != points[-1]:
        points.append(points[0])
    return points

# === 날씨 코드 추출 ===
def extract_weather_code(file_name):
    match = re.search(r'_(DD|DN|HD|ND|NN|NR|NS|RD|RN|SD)_', file_name)
    return match.group(1) if match else None

# === 라벨 통합 및 이미지 복사 ===
copied, fixed, skipped = 0, 0, 0

for label_root in label_dirs:
    for root, _, files in os.walk(label_root):
        for file in files:
            if not file.endswith(".json"):
                continue

            label_path = os.path.join(root, file)
            with open(label_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except:
                    continue

            img_name = os.path.basename(data.get("imagePath", ""))
            if not img_name:
                skipped += 1
                continue

            # 이미지 탐색
            candidates = list(Path(original_img_dir).rglob(img_name)) + list(Path(augmented_img_dir).rglob(img_name))
            if not candidates:
                skipped += 1
                continue

            img_path = str(candidates[0])
            shutil.copy2(img_path, os.path.join(unified_img_dir, img_name))
            data["imagePath"] = img_name

            valid_shapes = []
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "polygon":
                    continue
                pts = shape.get("points", [])
                closed = close_polygon_if_needed(pts.copy())
                if closed:
                    shape["points"] = closed
                    valid_shapes.append(shape)

            data["shapes"] = valid_shapes
            with open(os.path.join(final_label_dir, file), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            copied += 1
print(f"\n✅ 이미지 및 라벨 정리 완료: {copied}개")

# === labelme2coco 변환 ===
print("\n🚀 COCO 변환 중...")
convert(final_label_dir, coco_output_dir)
print("✅ COCO 변환 완료")

# === train/val/test 분할 ===
coco_path = os.path.join(coco_output_dir, "dataset.json")
with open(coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

image_id_to_annots = defaultdict(list)
for ann in annotations:
    image_id_to_annots[ann["image_id"]].append(ann)

train_imgs, temp_imgs = train_test_split(images, train_size=0.7, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)


def save_split(name, imgs):
    ids = {img["id"] for img in imgs}
    split = {
        "images": imgs,
        "annotations": [ann for ann in annotations if ann["image_id"] in ids],
        "categories": categories
    }
    with open(os.path.join(coco_output_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=4)

save_split("train", train_imgs)
save_split("val", val_imgs)
save_split("test", test_imgs)

print(f"\n✅ 분할 완료: train({len(train_imgs)}), val({len(val_imgs)}), test({len(test_imgs)})")
