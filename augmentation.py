# 날씨별 + 클래스별 불균형 해소를 위한 이미지 증대 
# albumentations 라이브러리를 활용해 이미지에 물리 기반 변환을 적용해 가상의 날씨 변화나 시점 변화를 흉내내는 방식 적용 (비, 눈, 회전, 확대 등)
# keypoint 기반으로 polygon 좌표 변환까지 동기화함
# ✅ 클래스 500 미만 → 정확히 500으로 맞춰 증강 (너무 적은 클래스의 경우 학습이 제대로 진행되지 않기 때문에 최소 500개 확보를 위함)
# ✅ 클래스 500 이상 → 10% 전략 증강으로 강건성 향상 (다양한 환경에서 적용 가능하도록 10% 증강)
# ✅ 날씨별 클래스 불균형 해소를 위해 개수 불균형 해소 후 기상 조건에 따른 증강 추가 실행하는 다차원 증강 적용
# ✅ 이미지 및 라벨 동기화 (이미지와 더불어 라벨도 함께 동기화함)
# ✅ 증강 시 매번 랜덤 방식 사용 (학습 이미지 자체의 부족으로 동일 이미지를 반복적으로 증강하는 중복 증강이 이루어질 수밖에 없음 -> 동일한 이미지를 여러 번 증강할 경우 전부 다른 방식으로 증강될 수 있도록 진행함)
# ✅ 날씨 추적 가능하도록 증강된 json 파일에 메타 데이터 기록  (weather_from: 원본 파일에서 추출, simulated_weather: 어떤 날씨 스타일을 적용했는지 기록)

import os
import json
import cv2
import numpy as np
import uuid
from tqdm import tqdm
from collections import defaultdict, Counter
import albumentations as A
import random
import re

# === 경로 설정 ===
image_root = r"C:\Users\dadab\Desktop\Sample\01.원천데이터"
label_root = r"C:\Users\dadab\Desktop\Sample\02.라벨링데이터"
save_img_root = r"C:\Users\dadab\Desktop\Sample\augmented\images"
save_label_root = r"C:\Users\dadab\Desktop\Sample\augmented\labels"

os.makedirs(save_img_root, exist_ok=True)
os.makedirs(save_label_root, exist_ok=True)

weather_codes = ['DD', 'DN', 'HD', 'ND', 'NN', 'NR', 'NS', 'RD', 'RN', 'SD']

# === 증강 정의 ===
def get_random_transform():
    simulated = []
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(blur_limit=5, p=0.3)
    ]
    if random.random() < 0.5:
        transforms.append(A.RandomRain(blur_value=1, p=1.0))
        simulated.append("rain")
    if random.random() < 0.5:
        transforms.append(A.RandomSnow(brightness_coeff=1.0, p=1.0))
        simulated.append("snow")
    transforms.append(A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-10, 10), p=0.5))

    transform = A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy'))
    return transform, simulated

# === 날씨 코드 추출 ===
def extract_weather_code(file_name):
    match = re.search(r'_(DD|DN|HD|ND|NN|NR|NS|RD|RN|SD)_', file_name)
    return match.group(1) if match else None

# === 원본 데이터 수집 ===
weather_class_counter = defaultdict(Counter)
weather_image_infos = defaultdict(list)

for clip_folder in os.listdir(label_root):
    label_clip_path = os.path.join(label_root, clip_folder, "Camera", "Camera_Front")
    image_clip_path = os.path.join(image_root, clip_folder, "Camera", "Camera_Front")
    if not os.path.isdir(label_clip_path):
        continue

    for file_name in os.listdir(label_clip_path):
        if not file_name.endswith(".json"):
            continue

        label_path = os.path.join(label_clip_path, file_name)
        image_path = os.path.join(image_clip_path, file_name.replace(".json", ".jpg"))

        weather = extract_weather_code(file_name)
        if not weather:
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            labels = [s["label"] for s in data.get("shapes", [])]
            weather_class_counter[weather].update(labels)

        weather_image_infos[weather].append({
            "labels": labels,
            "label_path": label_path,
            "image_path": image_path,
            "file_name": file_name,
            "weather": weather
        })

# === 증강 타겟 설정 ===
TARGET_PER_WEATHER = 700
aug_plan = defaultdict(lambda: defaultdict(int))

for weather, infos in weather_image_infos.items():
    current = len(infos)
    needed = max(0, TARGET_PER_WEATHER - current)
    if needed == 0:
        continue

    class_counts = weather_class_counter[weather]
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    class_list = [cls for cls, _ in sorted_classes[:7]]
    for i in range(needed):
        target_class = random.choice(class_list)
        aug_plan[weather][target_class] += 1

# === 증강 수행 ===
aug_idx = 0
aug_labels = []

for weather in aug_plan:
    print(f"\n🌦️ {weather} 조건 증강 중...")
    for cls in aug_plan[weather]:
        count = aug_plan[weather][cls]
        infos = [info for info in weather_image_infos[weather] if cls in info["labels"]]
        print(f"  ▶ {cls}: {count}장")
        for _ in range(count):
            info = random.choice(infos)
            transform, simulated_conditions = get_random_transform()

            image = cv2.imdecode(np.fromfile(info["image_path"], np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue

            with open(info["label_path"], 'r', encoding='utf-8') as f:
                data = json.load(f)
            original_shapes = data.get("shapes", [])

            keypoints = []
            point_map = {}
            pt_counter = 0
            for i, shape in enumerate(original_shapes):
                points = shape.get("points", [])
                point_map[i] = []
                for pt in points:
                    keypoints.append(tuple(pt))
                    point_map[i].append(pt_counter)
                    pt_counter += 1

            augmented = transform(image=image, keypoints=keypoints)
            aug_image = augmented["image"]
            aug_kps = augmented["keypoints"]

            aug_shapes = []
            for i, shape in enumerate(original_shapes):
                label = shape.get("label")
                if label != cls:
                    continue
                shape_type = shape.get("shape_type", "polygon")
                indices = point_map.get(i, [])
                aug_pts = [aug_kps[idx] for idx in indices if idx < len(aug_kps)]
                if not aug_pts:
                    continue
                aug_shapes.append({
                    "label": label,
                    "points": aug_pts,
                    "group_id": None,
                    "shape_type": shape_type,
                    "flags": {},
                    "object_id": None
                })

            if not aug_shapes:
                continue

            new_id = str(uuid.uuid4())[:8]
            new_img_name = f"aug_{aug_idx}_{new_id}.jpg"
            new_json_name = new_img_name.replace('.jpg', '.json')

            save_img_path = os.path.join(save_img_root, new_img_name)
            cv2.imencode('.jpg', aug_image)[1].tofile(save_img_path)

            aug_label = {
                "imagePath": new_img_name,
                "imageHeight": aug_image.shape[0],
                "imageWidth": aug_image.shape[1],
                "weather_from": info["weather"],
                "simulated_weather": simulated_conditions,
                "shapes": aug_shapes
            }
            save_json_path = os.path.join(save_label_root, new_json_name)
            with open(save_json_path, 'w', encoding='utf-8') as jf:
                json.dump(aug_label, jf, ensure_ascii=False, indent=4)

            aug_labels.extend([s["label"] for s in aug_shapes])
            aug_idx += 1

print(f"\n✅ 다차원 증강 완료! 총 생성 이미지 수: {aug_idx}개")

# === 날씨별 + 시뮬레이션 조건 포함한 클래스 분포 분석 ===
print("\n🌦️ 복합 날씨 조건별 클래스 분포 분석:")
combined_weather_counter = defaultdict(Counter)
label_dirs = [label_root, save_label_root]

for label_path in label_dirs:
    for root, _, files in os.walk(label_path):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                labels = [s["label"] for s in data.get("shapes", [])]
                weather = data.get("weather_from") or extract_weather_code(data.get("imagePath", file))
                sim = data.get("simulated_weather", [])
                weather_full = f"{weather}+{'+'.join(sim)}" if sim else weather
                combined_weather_counter[weather_full].update(labels)

for weather_full in sorted(combined_weather_counter):
    print(f"\n☁️ {weather_full}:")
    for cls, count in combined_weather_counter[weather_full].most_common():
        print(f"  {cls:15s}: {count}")
