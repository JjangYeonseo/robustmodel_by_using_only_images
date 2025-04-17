# 라벨 통합 및 보정 (라벨에 있는 polygon이 닫혀 있지 않은 경우, 자동으로 첫 점과 끝 점을 이어서 닫아주고, 원본 라벨(label_root)과 증강 라벨(save_label_root)을 하나의 폴더(combined_label_dir)에 복사하면서 polygon이 3개 미만 좌표이거나 안 닫힌 것은 제거 또는 보정
# Labelme 형식 → COCO 포맷으로 변환
# 날씨별 클래스 개수 분석
# 분석 결과를 CSV로 저장 (라벨 파일을 읽고, 파일 이름 또는 메타정보에서 날씨 코드를 추출한 뒤 해당 날씨에서 어떤 클래스가 몇 번 등장했는지 계산)

import os
import json
import csv
import cv2
import numpy as np
import uuid
from tqdm import tqdm
from collections import defaultdict, Counter
import albumentations as A
import random
import re
from labelme2coco import convert

# === 경로 설정 ===
image_root = r"C:\Users\dadab\Desktop\Sample\01.원천데이터"
label_root = r"C:\Users\dadab\Desktop\Sample\02.라벨링데이터"
save_img_root = r"C:\Users\dadab\Desktop\Sample\augmented\images"
save_label_root = r"C:\Users\dadab\Desktop\Sample\augmented\labels"
combined_label_dir = r"C:\Users\dadab\Desktop\Sample\all_labels"
coco_output_dir = r"C:\Users\dadab\Desktop\Sample\coco_output"
csv_output_path = r"C:\Users\dadab\Desktop\Sample\weather_class_distribution.csv"

os.makedirs(save_img_root, exist_ok=True)
os.makedirs(save_label_root, exist_ok=True)
os.makedirs(combined_label_dir, exist_ok=True)
os.makedirs(coco_output_dir, exist_ok=True)

weather_codes = ['DD', 'DN', 'HD', 'ND', 'NN', 'NR', 'NS', 'RD', 'RN', 'SD']

# === 유틸: polygon 닫기 ===
def close_polygon_if_needed(points):
    if len(points) < 3:
        return points
    if points[0] != points[-1]:
        points.append(points[0])
    return points

# === 전처리 및 통합: 라벨 복사 + polygon 보정 ===
def collect_and_fix_labels():
    fixed = 0
    for root_dir in [label_root, save_label_root]:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue
                src_path = os.path.join(root, file)
                dst_path = os.path.join(combined_label_dir, file)

                with open(src_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except:
                        continue

                modified = False
                new_shapes = []
                for shape in data.get("shapes", []):
                    if shape.get("shape_type") == "polygon":
                        pts = shape["points"]
                        if len(pts) < 3:
                            continue  # 유효하지 않은 polygon 제거
                        shape["points"] = close_polygon_if_needed(pts.copy())
                        modified = True
                    new_shapes.append(shape)

                data["shapes"] = new_shapes

                with open(dst_path, 'w', encoding='utf-8') as out_f:
                    json.dump(data, out_f, ensure_ascii=False, indent=4)
                if modified:
                    fixed += 1
    print(f"\n✅ 총 라벨 파일 복사 완료 및 polygon 보정: {fixed}개 수정됨")

# === 변환 수행 ===
def convert_to_coco():
    print("\n🚀 Labelme → COCO 포맷 변환 중...")
    convert(combined_label_dir, coco_output_dir)
    print("✅ COCO 변환 완료!")

# === 날씨 조건별 클래스 분포 분석 및 저장 ===
def analyze_weather_class_distribution():
    weather_class_counter = defaultdict(Counter)
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
                    if weather in weather_codes:
                        weather_class_counter[weather].update(labels)

    print("\n🌦️ 날씨 조건별 클래스 분포:")
    for weather in sorted(weather_class_counter):
        print(f"☁️ {weather}:")
        for cls, count in weather_class_counter[weather].most_common():
            print(f"  {cls:15s}: {count}")
        print()

    with open(csv_output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        all_classes = sorted({cls for counter in weather_class_counter.values() for cls in counter})
        writer.writerow(["Weather"] + all_classes)
        for weather in sorted(weather_class_counter):
            row = [weather] + [weather_class_counter[weather].get(cls, 0) for cls in all_classes]
            writer.writerow(row)
    print(f"\n📁 CSV 저장 완료: {csv_output_path}")

# === 날씨 코드 추출 ===
def extract_weather_code(file_name):
    match = re.search(r'_(DD|DN|HD|ND|NN|NR|NS|RD|RN|SD)_', file_name)
    return match.group(1) if match else None

# === 실행 ===
if __name__ == '__main__':
    collect_and_fix_labels()
    convert_to_coco()
    analyze_weather_class_distribution()
