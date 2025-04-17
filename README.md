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

## ✅ 증강 요약 결과

- 총 증강 이미지 수: **6883장**

---

### 📊 최종 클래스 분포 (원본 + 증강 포함)

| 클래스         | 총 개수 |
|----------------|--------:|
| static         | 5032개  |
| car            | 1646개  |
| vegetation     | 1571개  |
| dynamic        | 1023개  |
| traffic light  | 952개   |
| guard rail     | 884개   |
| traffic sign   | 822개   |
| person         | 749개   |
| sidewalk       | 712개   |
| cargroup       | 705개   |
| fence          | 651개   |
| wall           | 642개   |
| building       | 620개   |
| terrain        | 597개   |
| truck          | 595개   |
| road           | 593개   |
| bridge         | 584개   |
| tunnel         | 579개   |
| bus            | 569개   |
| pole           | 568개   |
| ground         | 501개   |
| sky            | 500개   |
| parking        | 500개   |
| rider          | 500개   |
| bicycle        | 500개   |
| trailer        | 500개   |
| motorcycle     | 500개   |
