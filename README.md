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

---

## ✅ 증강 요약 결과

- 총 증강 이미지 수: **6003장**

---



### 📊 최종 클래스 분포 (원본 + 증강)

| 클래스         | 총 개수 |
|----------------|--------:|
| static         | 4905개 |
| car            | 1559개 |
| vegetation     | 1525개 |
| guard rail     | 873개  |
| traffic sign   | 766개  |
| dynamic        | 747개  |
| road           | 591개  |
| sky            | 544개  |
| ground         | 543개  |
| pole           | 541개  |
| terrain        | 536개  |
| truck          | 530개  |
| building       | 524개  |
| sidewalk       | 515개  |
| wall           | 512개  |
| traffic light  | 512개  |
| fence          | 511개  |
| bridge         | 505개  |
| tunnel         | 503개  |
| person         | 503개  |
| cargroup       | 502개  |
| bus            | 501개  |
| parking        | 500개  |
| rider          | 500개  |
| bicycle        | 500개  |
| trailer        | 500개  |
| motorcycle     | 500개  |

---

### 📊 전체 클래스 분포 (원본 + 증강 포함) | 클래스 | 총 개수 | |----------------|--------:| | static | 4525개 | | car | 1454개 | | vegetation | 1410개 | | person | 1195개 | | dynamic | 1009개 | | tunnel | 946개 | | bus | 838개 | | cargroup | 799개 | | guard rail | 793개 | | pole | 787개 | | bridge | 736개 | | traffic sign | 707개 | | wall | 699개 | | building | 689개 | | ground | 580개 | | sky | 559개 | | truck | 551개 | | road | 542개 | | fence | 466개 | | parking | 453개 | | sidewalk | 452개 | | terrain | 394개 | | traffic light | 241개 | | bicycle | 186개 | | rider | 154개 | | motorcycle | 101개 | | trailer | 95개 |
