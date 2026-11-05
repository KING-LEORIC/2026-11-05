# 📋 프로젝트 논의 및 결과 보고서

---

## 1. 💬 서로의 논의 내용

### 📌 심화 관련 내용

> **Q. 1·2위 Loss 격차(0.000075)가 노이즈 수준인데, 무작위 샘플링(1위) 대신 재현성이 보장되는 2위(head_weight)를 채택하는 게 타당하지 않을까요?**

#### 🤝 논의 결과
1. `0.000075` 차이는 단일 시드 노이즈에 불과해 제기된 안정성 우려가 타당함.
2. 실무 환경의 재현성과 시스템 안정성을 최우선해 2위(`head_weight`)를 최종 선정함.

* 0.000075 차이는 단일 시드 노이즈에 불과해 제기된 안정성 우려가 타당함.
* 실무 환경의 재현성과 시스템 안정성을 최우선해 2위(`head_weight`)를 최종 선정함.

---

## 2. 📊 진행한 프로젝트에 대한 결과 정리

### 🔹 a. 기본필수

* **i. 요약**  
  3개 도형 클래스 총 36장의 소표본 환경에서 사전학습 백본을 동결하고 신규 Linear Head(771개 파라미터)만 학습시켜 8 에포크 수렴 성능과 Grad-CAM 설명력을 검증한 프로젝트입니다.
* **ii. 결론**  
  Train셋의 불균형(4:2:1)에도 불구하고 8 에포크 만에 검증 손실 0.0905 및 Accuracy/Macro-F1 1.0(오답 0건)을 달성하여, 사전학습 특징을 활용한 초경량 Head 전이학습의 유효성을 실증했습니다.
* **iii. 한계**  
  검증 데이터가 9장에 불과해 1건의 오분류로도 지표가 11.1%p 급변하는 통계적 불안정성이 크며, 일부 샘플의 낮은 확신도(76-80%)와 4x4 특징 맵 기반 Grad-CAM의 공간 해상도 한계로 인해 실환경 노이즈에 대한 강건성은 미검증 상태입니다.

🔗 **관련 링크**  
- [링크#1 : MODEL_CARD.md](https://github.com/kihoon808/AI-Core-Engineering-Course/blob/main/20261105/01_%EA%B8%B0%EB%B3%B8%ED%95%84%EC%88%98/MODEL_CARD.md)

---

### 🔹 b. 응용

* **i. 요약**  
  합성 도형으로 사전학습한 TinyShapeCNN Backbone을 고정(BN 통계 포함)하고, 불균형 train(120·40·20)으로 3클래스 Head를 학습하는 전이학습 프로젝트다. 동일한 seed·epoch(8)·valid 기준에서 head·none(기준선), partial·none(마지막 Conv 추가 해제), head·weight(클래스 가중 손실) 세 후보를 비교했다.
* **ii. 결론**  
  head·none은 Macro F1 0.8038, triangle Recall 0.5로 소수 클래스를 circle로 오분류했고, partial·none은 0.9562로 개선됐다. 학습 파라미터를 771개로 유지한 head·weight가 Macro F1 1.0, valid 오답 0건으로 가장 좋았으며, 재실행 시 metrics·predictions가 동일하게 재현됐다.
* **iii. 한계**  
  합성 데이터, 클래스당 30장인 valid, 단일 seed에서 epoch와 후보를 함께 고른 결과라 valid 성능은 낙관적 추정이며, 세 후보 모두 best epoch가 8로 학습 부족 가능성도 있다. test 평가와 sampler 등 다른 불균형 대응은 실행하지 않았다.

🔗 **관련 링크**  
- [링크#1 : day14_cnn_modeling_report.md](https://github.com/Minseok-conv/20261105/blob/master/day14_cnn_modeling_report.md)

---

## 3. 🔬 심화

* **i. 요약**  
  3개 클래스(circle, square, triangle) 총 300장의 이미지를 활용하여 소형 분류 모델과 ResNet18 사전학습 모델의 검증 성능 및 복원 가능성을 평가한 실험 결과 보고서입니다.
* **ii. 결론**  
  주 실험의 모든 후보와 ResNet18 실험에서 검증(Validation) Accuracy 및 Macro F1이 1.0을 기록했으며, 그중 Validation Loss가 가장 낮은 head + sampler 조합이 최종 모델로 선택되었습니다.
* **iii. 한계**  
  데이터셋이 300장으로 매우 적고 학습 데이터의 클래스 불균형이 존재하며, 검증 데이터 기반 모델 선택, Test 세트 미평가, 데이터 원출처 미확인 및 실서비스 환경 검증의 부재 등 일반화 성능 검증에 한계가 있습니다.

🔗 **관련 링크**  
- [링크#1 : MODEL_CARD.md (KING-LEORIC)](https://github.com/KING-LEORIC/2026-11-05/blob/main/03_%EC%8B%AC%ED%99%94/day14_vision/reports/model_card.md)  
- [링크#2 : MODEL_CARD.md (doorinetwork)](https://github.com/doorinetwork/ai-core-engineer-course/blob/main/20261105_day14/03_%EC%8B%AC%ED%99%94/day14_vision/reports/model_card.md)