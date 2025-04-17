import json
#✅ 아래 코드 수행하여 나온 분할 결과: train(5168), val(1476), test(739)


import os
import random
from sklearn.model_selection import train_test_split

# === 경로 설정 ===
coco_path = r"C:\Users\dadab\Desktop\Sample\coco_output\dataset.json"
output_dir = r"C:\Users\dadab\Desktop\Sample\coco_output"
os.makedirs(output_dir, exist_ok=True)

# === 비율 설정 ===
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

with open(coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

# 이미지 ID → annotation 연결
image_id_to_annots = {}
for ann in annotations:
    image_id_to_annots.setdefault(ann["image_id"], []).append(ann)

# 셔플 및 분할
random.shuffle(images)
train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

def save_split(name, split_imgs):
    split_img_ids = {img["id"] for img in split_imgs}
    split_anns = [ann for ann in annotations if ann["image_id"] in split_img_ids]

    split_data = {
        "images": split_imgs,
        "annotations": split_anns,
        "categories": coco["categories"]
    }

    with open(os.path.join(output_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=4)

save_split("train", train_imgs)
save_split("val", val_imgs)
save_split("test", test_imgs)

print(f"✅ 분할 완료: train({len(train_imgs)}), val({len(val_imgs)}), test({len(test_imgs)})")
