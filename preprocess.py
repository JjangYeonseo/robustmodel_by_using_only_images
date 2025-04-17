# 라벨 통합 및 보정 (라벨에 있는 polygon이 닫혀 있지 않은 경우, 자동으로 첫 점과 끝 점을 이어서 닫아주고, 원본 라벨(label_root)과 증강 라벨(save_label_root)을 하나의 폴더(combined_label_dir)에 복사하면서 polygon이 3개 미만 좌표이거나 안 닫힌 것은 제거 또는 보정
# Labelme 형식 → COCO 포맷으로 변환
# 날씨별 클래스 개수 분석
# 분석 결과를 CSV로 저장 (라벨 파일을 읽고, 파일 이름 또는 메타정보에서 날씨 코드를 추출한 뒤 해당 날씨에서 어떤 클래스가 몇 번 등장했는지 계산)

import os
import json
import csv
import shutil
from tqdm import tqdm
from collections import defaultdict, Counter
import re
from labelme2coco import convert
from pathlib import Path

# === 경로 설정 ===
original_img_dir = r"C:\Users\dadab\Desktop\Sample\01.원천데이터"
augmented_img_dir = r"C:\Users\dadab\Desktop\Sample\augmented\images"
label_dirs = [r"C:\Users\dadab\Desktop\Sample\02.라벨링데이터", r"C:\Users\dadab\Desktop\Sample\augmented\labels"]

unified_img_dir = r"C:\Users\dadab\Desktop\Sample\unified_images"
final_label_dir = r"C:\Users\dadab\Desktop\Sample\final_labels"
coco_output_dir = r"C:\Users\dadab\Desktop\Sample\coco_output"
csv_output_path = r"C:\Users\dadab\Desktop\Sample\weather_class_distribution.csv"

os.makedirs(unified_img_dir, exist_ok=True)
os.makedirs(final_label_dir, exist_ok=True)
os.makedirs(coco_output_dir, exist_ok=True)

weather_codes = ['DD', 'DN', 'HD', 'ND', 'NN', 'NR', 'NS', 'RD', 'RN', 'SD']

# === 유틸: polygon 닫기 ===
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

            img_name = data.get("imagePath")
            if not img_name:
                skipped += 1
                continue

            # 이미지 탐색 (원본/증강 포함)
            candidates = list(Path(original_img_dir).rglob(img_name)) + list(Path(augmented_img_dir).rglob(img_name))
            if not candidates:
                skipped += 1
                continue

            img_path = str(candidates[0])
            shutil.copy2(img_path, os.path.join(unified_img_dir, img_name))
            data["imagePath"] = img_name  # 경로 정리

            modified = False
            valid_shapes = []
            for shape in data.get("shapes", []):
                if shape.get("shape_type") != "polygon":
                    continue
                pts = shape["points"]
                closed = close_polygon_if_needed(pts.copy())
                if not closed:
                    continue  # 잘못된 polygon 제거
                shape["points"] = closed
                valid_shapes.append(shape)
                modified = True

            data["shapes"] = valid_shapes

            with open(os.path.join(final_label_dir, file), 'w', encoding='utf-8') as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=4)

            if modified:
                fixed += 1
            copied += 1

print(f"\n✅ 총 이미지 복사 완료: {copied}개")
print(f"✅ polygon 보정된 라벨: {fixed}개")
print(f"⚠️ 스킵된 항목: {skipped}개")

# === COCO 변환 ===
print("\n🚀 Labelme → COCO 포맷 변환 중...")
convert(final_label_dir, coco_output_dir)
print("✅ COCO 변환 완료!")

# === 날씨 조건별 클래스 분포 분석 및 CSV 저장 ===
print("\n📊 날씨 조건별 클래스 분포 분석 중...")
weather_class_counter = defaultdict(Counter)

for root, _, files in os.walk(final_label_dir):
    for file in files:
        if not file.endswith(".json"):
            continue
        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            labels = [s["label"] for s in data.get("shapes", [])]
            weather = data.get("weather_from") or extract_weather_code(data.get("imagePath", file))
            if weather:
                weather_class_counter[weather].update(labels)

for weather in sorted(weather_class_counter):
    print(f"☁️ {weather}:")
    for cls, count in weather_class_counter[weather].most_common():
        print(f"  {cls:15s}: {count}")
    print()

# === CSV 저장 ===
with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    all_classes = sorted({cls for c in weather_class_counter.values() for cls in c})
    writer.writerow(["Weather"] + all_classes)
    for weather in sorted(weather_class_counter):
        row = [weather] + [weather_class_counter[weather].get(cls, 0) for cls in all_classes]
        writer.writerow(row)

print(f"\n📁 CSV 저장 완료: {csv_output_path}")
