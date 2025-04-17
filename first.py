#날씨 조건 목록과 클래스별 분포 확인

import os
import json
from collections import Counter

# 루트 경로
label_root = r"C:\Users\dadab\Desktop\Sample\02.라벨링데이터"

weather_set = set()
class_counter = Counter()

# 모든 Clip 폴더 순회
for clip_folder in os.listdir(label_root):
    label_clip_path = os.path.join(label_root, clip_folder, "Camera", "Camera_Front")
    if not os.path.isdir(label_clip_path):
        continue

    for file_name in os.listdir(label_clip_path):
        if not file_name.endswith(".json"):
            continue

        # ➤ 날씨 조건 추출
        parts = file_name.split('_')
        if len(parts) > 1:
            weather_set.add(parts[1])

        # ➤ 클래스 수집
        label_path = os.path.join(label_clip_path, file_name)
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label:
                    class_counter[label] += 1

# 출력
print("✅ 날씨 조건 목록:", sorted(weather_set))
print("✅ 클래스 목록 및 빈도수:")
for cls, count in class_counter.most_common():
    print(f"  {cls}: {count}개")
