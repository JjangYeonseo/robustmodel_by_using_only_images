# robustmodel_by_using_only_images

---

## ✅ 원본 데이터

- **날씨 조건 목록**:

| 약어 | 의미 (Weather Condition)        |
|------|----------------------------------|
| DD   | Daytime – Clear                 |
| DN   | Daytime – Night Transition      |
| HD   | Hazy Day                        |
| ND   | Night – Dry                     |
| NN   | Night – No Light / Dark         |
| NR   | Night – Rainy                   |
| NS   | Night – Snowy                   |
| RD   | Rainy Day                       |
| RN   | Rain – Night                    |
| SD   | Snowy Day                       |

- **클래스 목록 및 빈도수:**

| 클래스           | 개수 |   | 클래스           | 개수 |
|------------------|------|---|------------------|------|
| static           | 4525 |   | sky              | 475  |
| car              | 1454 |   | pole             | 451  |
| vegetation       | 1410 |   | building         | 414  |
| guard rail       | 793  |   | truck            | 384  |
| dynamic          | 715  |   | terrain          | 321  |
| traffic sign     | 707  |   | wall             | 244  |
| road             | 542  |   | sidewalk         | 170  |
| ground           | 487  |   | traffic light    | 163  |
| fence            | 137  |   | bus              | 64   |
| tunnel           | 49   |   | bridge           | 39   |
| person           | 34   |   | cargroup         | 21   |
| parking          | 8    |   | rider            | 2    |
| bicycle          | 2    |   | trailer          | 1    |
| motorcycle       | 1    |   |                  |      |

---

## 📈 증강 데이터 설정

### 🎯 균형 증강 대상 (500개 미만 클래스 → 500까지 보충)

- `sky`: +25  
- `truck`: +116  
- `ground`: +13  
- `building`: +86  
- `sidewalk`: +330  
- `terrain`: +179  
- `wall`: +256  
- `traffic light`: +337  
- `bridge`: +461  
- `fence`: +363  
- `pole`: +49  
- `tunnel`: +451  
- `bus`: +436  
- `person`: +466  
- `parking`: +492  
- `cargroup`: +479  
- `rider`: +498  
- `bicycle`: +498  
- `trailer`: +499  
- `motorcycle`: +499  

### 💪 전략 증강 대상 (강건성 향상 목적, 500 이상 클래스 중 일부 10% 추가 증강)

- `static`: +452  
- `road`: +54  
- `vegetation`: +141  
- `guard rail`: +79  
- `car`: +145  
- `traffic sign`: +70  
- `dynamic`: +71  

---

## ✅ 최종 결과

- 총 **증강 이미지 수**: `5390`개

- **최종 클래스 분포 (원본 + 증강 포함)**:

| 클래스           | 개수 |   | 클래스           | 개수 |
|------------------|------|---|------------------|------|
| static           | 4947 |   | traffic light    | 520  |
| car              | 1566 |   | sidewalk         | 516  |
| vegetation       | 1515 |   | truck            | 515  |
| guard rail       | 849  |   | fence            | 510  |
| dynamic          | 758  |   | bus              | 508  |
| traffic sign     | 752  |   | person           | 503  |
| road             | 581  |   | cargroup         | 502  |
| sky              | 539  |   | tunnel           | 501  |
| ground           | 538  |   | bridge           | 500  |
| building         | 531  |   | parking          | 500  |
| pole             | 531  |   | rider            | 500  |
| terrain          | 526  |   | bicycle          | 500  |
| wall             | 524  |   | trailer          | 500  |
|                  |      |   | motorcycle       | 500  |

---

