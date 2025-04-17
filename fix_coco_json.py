# file_name 경로 정리 (os.path.basename)
# bbox가 누락된 경우, segmentation(polygon)으로부터 자동 계산
# bbox_mode가 없으면 자동으로 1로 설정 (COCO는 XYWH 기준 → BoxMode.XYWH_ABS == 1)
# bbox 유효성 검사 (len == 4이고 숫자 타입) -> 유효하지 않으면 제거

import json
import os

jsons = [
    r"C:\Users\dadab\Desktop\Sample\coco_output\train.json",
    r"C:\Users\dadab\Desktop\Sample\coco_output\val.json",
    r"C:\Users\dadab\Desktop\Sample\coco_output\test.json",
]

for path in jsons:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 파일명만 남기기
    for img in data["images"]:
        img["file_name"] = os.path.basename(img["file_name"])

    valid_anns = []
    removed_count = 0

    for ann in data["annotations"]:
        bbox = ann.get("bbox", [])

        # bbox 없으면 segmentation으로부터 계산
        if (not bbox or len(bbox) != 4) and "segmentation" in ann:
            seg = ann["segmentation"]
            if isinstance(seg, list) and all(isinstance(p, list) for p in seg):
                xs, ys = [], []
                for poly in seg:
                    xs.extend(poly[::2])
                    ys.extend(poly[1::2])
                if xs and ys:
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    ann["bbox"] = bbox

        # bbox_mode 없으면 기본값 1(COCO 포맷)로 설정
        if "bbox_mode" not in ann:
            ann["bbox_mode"] = 1  # XYWH_ABS

        # 유효성 검사
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
            valid_anns.append(ann)
        else:
            removed_count += 1
            print(f"❌ 잘못된 bbox 제거됨: {bbox}")

    data["annotations"] = valid_anns

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Cleaned & Validated: {os.path.basename(path)} | 유효 bbox: {len(valid_anns)}, 이미지: {len(data['images'])}")
