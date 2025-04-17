### robustmodel_by_using_only_images

---

## 원본 데이터

---

#✅ 날씨 조건 목록: ['DD', 'DN', 'HD', 'ND', 'NN', 'NR', 'NS', 'RD', 'RN', 'SD']
#✅ 클래스 목록 및 빈도수:
  static: 4525개
  car: 1454개
  vegetation: 1410개
  guard rail: 793개
  dynamic: 715개
  traffic sign: 707개
  road: 542개
  ground: 487개
  sky: 475개
  pole: 451개
  building: 414개
  truck: 384개
  terrain: 321개
  wall: 244개
  sidewalk: 170개
  traffic light: 163개
  fence: 137개
  bus: 64개
  tunnel: 49개
  bridge: 39개
  person: 34개
  cargroup: 21개
  parking: 8개
  rider: 2개
  bicycle: 2개
  trailer: 1개
  motorcycle: 1개

---

## 증강 적용한 데이터

---

# 🎯 균형 증강 대상:
  sky: +25개 필요
  truck: +116개 필요
  ground: +13개 필요
  building: +86개 필요
  sidewalk: +330개 필요
  terrain: +179개 필요
  wall: +256개 필요
  traffic light: +337개 필요
  bridge: +461개 필요
  fence: +363개 필요
  pole: +49개 필요
  tunnel: +451개 필요
  bus: +436개 필요
  person: +466개 필요
  parking: +492개 필요
  cargroup: +479개 필요
  rider: +498개 필요
  bicycle: +498개 필요
  trailer: +499개 필요
  motorcycle: +499개 필요

---

# 💪 전략 증강 대상 (강건성 강화):
  static: +452개 추가
  road: +54개 추가
  vegetation: +141개 추가
  guard rail: +79개 추가
  car: +145개 추가
  traffic sign: +70개 추가
  dynamic: +71개 추가

---

# ✅ 총 증강 이미지 수: 5390개

---

# 📊 최종 클래스 분포 (원본 + 증강):
  static: 4947개
  car: 1566개
  vegetation: 1515개
  guard rail: 849개
  dynamic: 758개
  traffic sign: 752개
  road: 581개
  sky: 539개
  ground: 538개
  building: 531개
  pole: 531개
  terrain: 526개
  wall: 524개
  traffic light: 520개
  sidewalk: 516개
  truck: 515개
  fence: 510개
  bus: 508개
  person: 503개
  cargroup: 502개
  tunnel: 501개
  bridge: 500개
  parking: 500개
  rider: 500개
  bicycle: 500개
  trailer: 500개
  motorcycle: 500개
