# Model Card: 3-Class Shape Image Classification

## 1. 모델 개요

  항목                      내용
  ------------------------- ------------------------------------------------
  작업                      이미지 분류(Image Classification)
  분류 클래스               `circle`, `square`, `triangle`
  클래스 인덱스             `circle=0`, `square=1`, `triangle=2`
  주요 평가 데이터          검증 데이터(Validation)
  테스트 데이터 사용 여부   사용하지 않음
  최종 선택 모델(주 실험)   소형 모델의 `head + sampler` 후보
  별도 비교 실험            ImageNet 사전학습 ResNet18의 `head`, `partial`

이 문서는 제공된 실행 결과와 코드 동작 정보를 바탕으로 작성했다.
확인되지 않은 데이터 출처나 실험 조건은 사실처럼 단정하지 않고 별도로
표시했다.

## 2. 데이터 및 출처

### 2.1 데이터 분할 및 클래스별 이미지 수

  분할            circle   square   triangle      합계
  ------------ --------- -------- ---------- ---------
  Train              120       40         20       180
  Validation          30       30         30        90
  Test                10       10         10        30
  **전체**       **160**   **80**     **60**   **300**

-   Train 데이터는 클래스별 이미지 수가 다르다. `circle`이 120장으로
    가장 많고 `triangle`은 20장으로 가장 적다.
-   Train 데이터의 클래스 비율은 `circle:square:triangle = 6:2:1`이다.
-   Validation과 Test는 각 클래스가 30장, 10장으로 동일한 수량이다.
-   파일 개수는 사용자가 확인한 PNG 파일 수를 기준으로 한다.

### 2.2 레이블 매핑

  클래스 이름     숫자 레이블
  ------------- -------------
  circle                    0
  square                    1
  triangle                  2

체크포인트에 저장된 `class_to_idx`에서도 위 매핑을 확인했다.

### 2.3 데이터 출처 및 중복 관리

-   이미지의 원본 생성 방식, 수집 경로 및 라이선스 정보는 제공된
    결과만으로 확인할 수 없다.
-   따라서 데이터가 실제 촬영 이미지인지, 프로그램으로 생성한 합성
    이미지인지 이 문서에서는 확정하지 않는다.
-   Train/Validation/Test 사이에 동일 이미지 또는 거의 같은 이미지가
    중복되었는지, 같은 원본에서 파생된 이미지가 분할을 넘나드는지 확인한
    기록은 제공되지 않았다.
-   향후 실제 데이터로 평가할 때는 원본 단위 또는 그룹 단위로 먼저
    분리한 뒤 Train/Validation/Test를 구성하는 것이 바람직하다. 같은
    원본에서 변형한 이미지가 서로 다른 분할에 들어가면 데이터 누수로
    인해 성능이 실제보다 높게 측정될 수 있다.

## 3. 실험 환경

  항목                  버전 또는 값
  --------------------- -------------------------------------
  운영체제              Windows 11 (사용자 제공 표기: 26H2)
  Python                3.13.16
  PyTorch               2.14.0+cu130
  Torchvision           0.29.0+cu130
  NumPy                 2.5.4
  Pandas                3.1.1
  scikit-learn          1.9.1
  OpenCV                5.0.0
  CUDA 사용 가능 여부   `True`

PyTorch에서 `torch.cuda.is_available()`가 `True`로 확인되었다. 따라서
CUDA를 사용할 수 있는 환경에서 실행한 결과다. 다만 이 기록에는 GPU
모델명과 실제 실행 중 선택된 장치를 별도로 출력한 로그가 없으므로 GPU
모델명은 기재하지 않는다.

## 4. 실험 설계 및 학습 조건

### 4.1 주 실험: 소형 모델 후보 비교

다음 네 후보를 비교했다.

  후보   학습 전략   클래스 불균형 처리
  ------ ----------- --------------------------
  1      `head`      기본 데이터 순서 섞기
  2      `partial`   기본 데이터 순서 섞기
  3      `head`      클래스 가중치(`weight`)
  4      `head`      가중치 샘플러(`sampler`)

확인된 공통 조건:

-   Seed: 14
-   Epoch 예산: 후보별 최대 8 Epoch
-   Train Batch Size: 16
-   Validation Batch Size: 32
-   Optimizer: AdamW
-   Weight Decay: 0.0001
-   Head Learning Rate: 0.01
-   Backbone Learning Rate: 0.001
-   Batch Normalization: 학습 중 고정(`freeze_bn=True`)
-   Train 데이터: 기본 후보는 셔플, `sampler` 후보는
    `WeightedRandomSampler` 사용
-   Validation 데이터: 셔플하지 않음
-   입력 변환: `Resize((32, 32))` 후 `ToTensor()`

`head`는 분류기 부분을 학습하는 전략이고, `partial`은 분류기뿐 아니라
일부 특징 추출 계층도 학습하는 전략이다. `sampler`는 Train에서 적은
클래스가 학습 배치에 더 자주 포함되도록 샘플링하는 방식이다.

### 4.2 별도 실험: ImageNet 사전학습 ResNet18

공식 Torchvision ResNet18을 사용한 별도 실험이다.

  항목                 `head`                     `partial`
  -------------------- -------------------------- -----------------------------
  초기 가중치          ImageNet `IMAGENET1K_V1`   ImageNet `IMAGENET1K_V1`
  학습 대상            최종 분류기 `fc`           `layer4`와 최종 분류기 `fc`
  Epoch                2                          2
  Batch Size           8                          8
  분류 클래스 수       3                          3
  입력 전처리          공식 가중치의 transforms   공식 가중치의 transforms
  테스트 데이터 사용   아니오                     아니오

ResNet18의 공식 전처리는 Resize 256, Center Crop 224, ImageNet
평균·표준편차 정규화로 구성된다. 이는 주 실험 소형 모델의 32×32 입력
전처리와 다르다.

공통 학습 조건으로 Seed 14, AdamW, Weight Decay 0.0001, 분류기 학습률
0.001을 사용했다. `partial`에서는 `layer4` 학습률 0.0001을 추가로
사용했다. Batch Normalization 계층은 학습 중 평가 모드로 고정했다.

이 ResNet18 결과는 소형 모델 네 후보와 입력 크기, 모델 구조, 학습률,
Batch Size 및 Epoch 예산이 다르므로 동일 조건의 직접 비교로 해석하면 안
된다.

## 5. 평가 결과

### 5.1 주 실험: 소형 모델 후보 결과

  ----------------------------------------------------------------------------------------------
  후보           Accuracy    Macro F1      Circle      Square    Triangle   Validation      Best
                                           Recall      Recall      Recall         Loss     Epoch
  ----------- ----------- ----------- ----------- ----------- ----------- ------------ ---------
  head + none       1.000       1.000       1.000       1.000       1.000       0.0126         8

  partial +         1.000       1.000       1.000       1.000       1.000       0.0081         8
  none                                                                                 

  head +            1.000       1.000       1.000       1.000       1.000       0.0066         8
  weight                                                                               

  **head +      **1.000**   **1.000**   **1.000**   **1.000**   **1.000**   **0.0065**     **8**
  sampler**                                                                            
  ----------------------------------------------------------------------------------------------

모든 후보가 Validation 90장에 대해 Accuracy 1.0, Macro F1 1.0을
기록했다. 클래스별 Recall도 모두 1.0이다.

### 5.2 최종 후보 선택 근거

선택 기준은 다음과 같다.

1.  Macro F1이 높은 후보를 우선한다.
2.  Macro F1이 같으면 Validation Loss가 낮은 후보를 선택한다.

네 후보의 Macro F1은 모두 1.0으로 같았다. 그중 `head + sampler`의
Validation Loss가 0.0065로 가장 낮아 주 실험의 최종 후보로 선택했다.

Validation Loss는 정답 여부뿐 아니라 모델이 정답 클래스에 부여한 확률의
크기에도 영향을 받는다. 따라서 Accuracy와 Macro F1이 같더라도 Loss는
서로 다를 수 있다.

`best_epoch=8`은 8회 학습 중 Validation Loss가 가장 낮았던 Epoch가
8이라는 뜻이다. 정확도가 처음으로 100%가 된 시점을 의미하지는 않는다.

### 5.3 ResNet18 별도 실험 결과

  -----------------------------------------------------------------------------------
  전략             Epoch 1    Epoch 1    Epoch 1        Epoch 2    Epoch 2    Epoch 2
                Validation   Accuracy   Macro F1     Validation   Accuracy   Macro F1
                      Loss                                 Loss            
  --------- -------------- ---------- ---------- -------------- ---------- ----------
  head          0.38928818     0.7778     0.7500     0.12204314     1.0000     1.0000

  partial     0.0000022451     1.0000     1.0000   0.0000003126     1.0000     1.0000
  -----------------------------------------------------------------------------------

-   `head`는 1 Epoch에서 Accuracy 0.7778, Macro F1 0.75였고 2 Epoch에서
    두 지표 모두 1.0이 되었다.
-   `partial`은 1 Epoch부터 Accuracy와 Macro F1이 1.0이었고 2 Epoch에서
    Validation Loss가 더 낮아졌다.
-   두 전략 모두 Test 데이터는 사용하지 않았다.
-   현재 제공된 ResNet18 로그에는 클래스별 Recall, Confusion Matrix,
    예측별 확률, 학습 소요 시간 및 best epoch 번호가 포함되어 있지 않다.
    따라서 해당 항목은 추정하지 않는다.

## 6. 오류 분석 및 설명 가능성

### 6.1 Validation 오분류

제공된 `predictions.csv`에는 Validation 이미지 90장의 실제 클래스와 예측
클래스가 기록되어 있다. 모든 행에서 `actual`과 `predicted`가 일치한다.

Confusion Matrix:

  실제  예측     circle   square   triangle
  ------------ -------- -------- ----------
  circle             30        0          0
  square              0       30          0
  triangle            0        0         30

-   Validation 오분류 수: 0건
-   Validation 정분류 수: 90건
-   클래스별 Recall: 모두 1.0
-   제공된 오류 목록에서 확인되는 실제 오분류 사례: 없음

따라서 실제 오분류 이미지의 ID, 오분류 대상 클래스 또는 오류 원인을
임의로 작성하지 않았다.

### 6.2 예측 확률 해석

`predictions.csv`에는 각 이미지에 대해 예측 클래스의
확률(`confidence`)과 클래스별 확률(`p0`, `p1`, `p2`)이 저장되어 있다.

예를 들어 `circle_002.png`는 circle로 올바르게 분류되었지만 circle
확률은 약 0.8388이다. 이는 다른 Validation 이미지보다 모델의 확신이
낮았다는 뜻이지, 오분류가 발생했다는 뜻은 아니다.

모델이 출력한 확률은 모델의 확신 정도를 나타내는 값이다. 실제 정답
가능성과 항상 일치하는 보정된 확률이라고 단정할 수는 없다.

### 6.3 Grad-CAM

제공된 `cam_index.json`에는 다음 항목이 기록되어 있다.

-   종류: `correct`
-   이미지: `valid\circle\circle_000.png`
-   정답 클래스: 0 (`circle`)

이는 정답으로 분류된 이미지에 대한 Grad-CAM 대상이 기록되어 있다는
뜻이다.

Grad-CAM은 모델 예측에 영향을 준 특징 영역을 시각적으로 살펴보는
방법이다. 모델이 실제로 어떤 논리로 판단했는지 완전히 증명하거나, 특정
영역이 예측의 원인임을 인과적으로 입증하지는 않는다.

현재 제공된 `cam_index.json`에는 위의 정답 사례 한 건만 확인된다. 별도의
`cam_wrong` 이미지나 오류 이미지가 제공되지 않았으므로 오분류 Grad-CAM
사례는 기재하지 않는다.

## 7. 체크포인트 및 재현

### 7.1 주 실험 체크포인트

주 실험은 선택된 `head + sampler` 모델을 `best_model.pt`로 저장하고,
저장된 모델을 다시 불러와 Validation 예측을 재현하는 절차를 포함한다.

복원 확인 결과:

  확인 항목                    결과
  ---------------------------- ------
  검증 이미지 경로 순서 일치   True
  예측 확률 일치               True
  예측 클래스 일치             True

따라서 제공된 복원 확인 결과에서는 저장된 체크포인트를 다시 불러온 뒤
이미지 경로 순서, 예측 확률 및 예측 클래스가 일치했다.

### 7.2 ResNet18 체크포인트

ResNet18 실행 결과 파일:

-   `outputs/imagenet/head.pt`
-   `outputs/imagenet/head.json`
-   `outputs/imagenet/partial.pt`
-   `outputs/imagenet/partial.json`

업로드된 `head.pt` 체크포인트의 메타데이터에서 다음 항목을 확인했다.

-   가중치 출처: `IMAGENET1K_V1`
-   전략: `head`
-   클래스 매핑: `circle=0`, `square=1`, `triangle=2`
-   입력 전처리: Resize 256, Crop 224, ImageNet 정규화

### 7.3 실행 명령

주 실험 및 체크포인트 복원:

``` bash
python main.py
python restore.py
```

ResNet18 별도 실험:

``` bash
python resnet18_official.py --strategy head --epochs 2
python resnet18_official.py --strategy partial --epochs 2
```

환경 확인:

``` bash
python --version
python -c "import torch, torchvision, numpy, pandas, sklearn, cv2; print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('numpy:',numpy.__version__); print('pandas:',pandas.__version__); print('sklearn:',sklearn.__version__); print('opencv:',cv2.__version__); print('CUDA:',torch.cuda.is_available())"
```

## 8. 한계 및 주의사항

1.  **작은 데이터셋:** 전체 데이터는 300장이다. 현재 Validation에서 높은
    점수를 얻었더라도 다양한 실제 환경의 이미지에 일반화된다고 단정할 수
    없다.
2.  **학습 데이터 불균형:** Train은 circle 120장, square 40장, triangle
    20장으로 클래스 수가 불균형하다. Sampler를 적용한 실험은 이 문제를
    완화하려는 시도지만, 데이터 자체의 다양성 부족까지 해결하지는
    않는다.
3.  **데이터 출처 미확인:** 이미지의 실제 출처와 생성 방식, 중복·그룹
    분할 정책이 확인되지 않았다.
4.  **Validation 기반 선택:** 후보 선택에 Validation 성능을 사용했다.
    따라서 Validation 결과는 완전히 독립적인 최종 성능 평가가 아니다.
5.  **Test 미사용:** Test 이미지 30장은 존재하지만 이번 실험에서
    평가하지 않았다. 최종 모델과 설정을 확정한 뒤 Test를 한 번 평가해
    별도의 최종 성능으로 보고해야 한다.
6.  **ResNet18과 소형 모델의 조건 차이:** 두 실험은 입력 크기, 모델
    구조, 학습률, Batch Size 및 Epoch 예산이 다르므로 결과를 동일 조건의
    우열로 단정할 수 없다.
7.  **Grad-CAM 해석 한계:** Grad-CAM은 시각적 설명 도구이며 모델 판단의
    인과적 근거를 증명하지 않는다.
8.  **실서비스 검증 필요:** 실제 사용 환경의 조명, 배경, 크기, 회전,
    가림, 카메라 및 이미지 품질 변화에 대한 추가 검증이 필요하다.

## 9. 종합 결론

주 실험의 네 후보는 모두 Validation에서 Accuracy 1.0, Macro F1 1.0 및
클래스별 Recall 1.0을 기록했다. 사전에 정한 선택 기준인 Macro F1 우선,
동점 시 Validation Loss 최소 기준에 따라 `head + sampler`를 최종 후보로
선택했다.

ImageNet 사전학습 ResNet18의 별도 실험에서도 `head`와 `partial` 모두 2
Epoch 종료 시 Validation Accuracy와 Macro F1이 1.0이었다. `partial`은 1
Epoch부터 두 지표가 1.0이었으며 Validation Loss도 더 낮았다. 다만 두
실험 트랙은 학습 조건이 다르므로 직접적인 성능 순위로 해석하지 않는다.

체크포인트 복원 결과에서는 이미지 경로 순서, 예측 확률 및 예측 클래스가
모두 일치했다.

**현재 결과는 제한된 데이터셋의 Validation 성능과 복원 가능성을 확인한
실험 결과다. 실제 환경에서의 일반화 성능이 검증되었다는 의미는 아니며,
Test 데이터 평가와 다양한 실제 이미지에 대한 추가 검증이 남아 있다.**
