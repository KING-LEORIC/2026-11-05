# 파일 및 폴더 경로를 운영체제에 맞게 처리하기 위한 기능
from pathlib import Path
# JSON 형식의 데이터 읽기 및 저장 기능
import json
# 배열 및 수치 연산을 위한 라이브러리
import numpy as np
# 표 형태의 데이터 처리를 위한 라이브러리
import pandas as pd
# 딥러닝 연산을 위한 PyTorch
import torch
# 분류 성능을 상세하게 평가하고 혼동 행렬을 계산하는 함수
from sklearn.metrics import classification_report,confusion_matrix
# 모델 학습 및 평가를 수행하는 함수
from engine import experiment
# 오답 이미지와 Grad-CAM 이미지를 생성하는 함수
from explain import save_explanations

# 현재 main.py 파일이 있는 폴더를 기준으로 프로젝트 경로 설정
# - __file__ : 현재 실행 중인 Python 파일의 경로
# - resolve() : 절대 경로로 변환
# - parent : 현재 파일이 들어 있는 폴더
# - DATA : 이미지 데이터가 저장된 폴더
# - OUT : 실행 결과를 저장할 폴더
BASE=Path(__file__).resolve().parent;DATA=BASE/"data";OUT=BASE/"outputs"
# 결과 저장 폴더 생성 및 CPU 연산 스레드 수 설정
# - mkdir(exist_ok=True) : outputs 폴더가 없으면 생성
# - torch.set_num_threads(1) : CPU 연산 스레드를 1개로 제한
OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
# TODO D07: 같은 8회 예산에서 네 후보를 명시하세요. Scratch는 비교표의 별도 보조 후보입니다.
# BEGIN D07

# 동일한 8 epoch 예산으로 비교할 네 가지 실험 후보 지정
# - ("head","none") : 분류 Head만 학습하고 클래스 불균형 처리는 하지 않음
# - ("partial","none") : 마지막 Conv 계층과 Head를 학습하고 불균형 처리는 하지 않음
# - ("head","weight") : Head만 학습하고 손실 함수에 클래스별 가중치 적용
# - ("head","sampler") : Head만 학습하고 샘플 추출 시 클래스별 가중치 적용
# - 모든 후보는 experiment()에서 epochs=8로 실행
experiments=[("head","none"),("partial","none"),("head","weight"),("head","sampler")]  # 나머지 세 후보를 추가하세요.
# END D07
# 실험 결과와 학습 기록, 최적 후보 선택을 위한 변수 초기화
# - rows : 후보별 최종 평가 결과를 저장할 목록
# - histories : 후보별 epoch 학습 기록을 저장할 목록
# - best_key : 현재까지 가장 좋은 후보를 판정하는 기준값
rows=[];histories=[];best_key=None
# 지정한 네 가지 실험 후보를 하나씩 실행
# - strategy : 학습할 모델 영역
# - balance : 클래스 불균형 처리 방식
for strategy,balance in experiments:
    # 현재 후보의 모델 학습 및 검증 수행
    # - DATA : 이미지 데이터 폴더
    # - strategy : 현재 후보의 학습 전략
    # - balance : 현재 후보의 불균형 처리 방식
    # - epochs=8 : 후보마다 8 epoch 학습
    # - model : 학습이 완료된 모델
    # - summary : 최종 검증 평가 지표
    # - history : epoch별 학습 기록
    # - prediction : 검증 이미지별 예측 결과
    model,summary,history,prediction=experiment(DATA,strategy=strategy,balance=balance,epochs=8)
    # 현재 후보의 최종 결과와 epoch별 기록을 목록에 추가
    # - rows : 후보별 최종 평가 지표 저장
    # - history["candidate"] : 후보를 구분하기 위한 이름
    # - strategy+"_"+balance : 예를 들어 head_none과 같은 후보 이름 생성
    # - histories : 후보별 학습 기록 저장
    rows.append(summary);history["candidate"]=strategy+"_"+balance;histories.append(history)
    # TODO D08: Macro F1을 최대화하고 동률이면 loss가 작은 후보를 고르세요. test는 보지 않습니다.
    # BEGIN D08
    
    # 현재 후보의 성능을 비교하기 위한 기준값 생성
    # - summary["macro_f1"] : 세 클래스의 F1 점수를 동일한 비중으로 평균한 값
    # - summary["loss"] : 검증 데이터의 평균 손실
    # - -summary["loss"] : loss가 작을수록 큰 비교값이 되도록 부호 반전
    # - key : Macro F1을 우선 비교하고, 동률이면 loss를 비교하는 튜플
    key=(summary["macro_f1"],-summary["loss"])  # Macro F1, loss 순서로 바꾸세요.
    # END D08
    # 첫 번째 후보이거나 현재 후보가 기존 최적 후보보다 좋은지 확인
    # - best_key is None : 아직 최적 후보가 선택되지 않은 경우
    # - key>best_key : 현재 후보의 비교 기준값이 더 높은 경우
    if best_key is None or key>best_key:
        # 현재 후보를 최적 후보로 저장
        # - best_key : 최적 후보의 비교 기준값
        # - chosen : 최적 후보의 평가 지표
        # - chosen_pred : 최적 후보의 이미지별 예측 결과
        # - chosen_state : 최적 후보의 모델 가중치 복사본
        # - detach() : 계산 그래프에서 분리
        # - cpu() : CPU 메모리로 이동
        # - clone() : 독립적인 Tensor 복사본 생성
        best_key=key;chosen=summary;chosen_pred=prediction;chosen_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
# 후보별 최종 평가 결과를 CSV 파일로 저장
# - pd.DataFrame(rows) : 후보별 결과 목록을 표로 변환
# - to_csv() : CSV 파일로 저장
# - index=False : 행 번호를 별도 열로 저장하지 않음
pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False)
# 모든 후보의 epoch별 학습 기록을 하나의 CSV 파일로 저장
# - pd.concat() : 여러 DataFrame을 하나로 연결
# - ignore_index=True : 연결 후 행 번호를 새로 지정
pd.concat(histories,ignore_index=True).to_csv(OUT/"history.csv",index=False)
# 최적 후보의 가중치를 현재 모델에 복원하고 평가 모드로 전환
# - load_state_dict() : 저장한 모델 가중치를 모델에 적용
# - model.eval() : 평가 모드로 전환
model.load_state_dict(chosen_state);model.eval()
# 최적 후보의 이미지별 예측 결과를 CSV 파일로 저장
chosen_pred.to_csv(OUT/"predictions.csv",index=False)
# 최적 후보의 예측 결과를 바탕으로 클래스별 상세 평가 보고서 계산
# - chosen_pred.actual : 실제 정답 라벨
# - chosen_pred.predicted : 모델의 예측 라벨
# - labels=[0,1,2] : 세 클래스의 라벨 번호
# - target_names : 각 라벨에 대응하는 클래스 이름
# - output_dict=True : 결과를 딕셔너리로 반환
# - zero_division=0 : 0으로 나누는 경우 해당 지표를 0으로 처리
report=classification_report(chosen_pred.actual,chosen_pred.predicted,labels=[0,1,2],target_names=["circle","square","triangle"],output_dict=True,zero_division=0)
# 최적 후보의 평가 결과와 상세 보고서를 JSON 파일로 저장
# - selected : 선택된 후보의 평가 지표
# - report : 클래스별 정밀도·재현율·F1 점수
# - test_used=False : test 데이터를 사용하지 않았음을 기록
# - json.dumps() : Python 딕셔너리를 JSON 문자열로 변환
# - indent=2 : JSON 내용을 들여쓰기하여 보기 좋게 저장
# - ensure_ascii=False : 한글 등 유니코드 문자를 그대로 저장
# - encoding="utf-8" : UTF-8 인코딩 사용
(OUT/"metrics.json").write_text(json.dumps({"selected":chosen,"report":report,"test_used":False},indent=2,ensure_ascii=False),encoding="utf-8")
# 최적 후보의 혼동 행렬을 계산하고 CSV 파일로 저장
# - confusion_matrix() : 실제 라벨과 예측 라벨을 비교하여 혼동 행렬 계산
# - labels=[0,1,2] : 세 클래스의 순서를 지정
# - pd.DataFrame() : 혼동 행렬을 표 형태로 변환
pd.DataFrame(confusion_matrix(chosen_pred.actual,chosen_pred.predicted,labels=[0,1,2])).to_csv(OUT/"confusion.csv",index=False)
# 최적 후보의 예측 결과를 이용해 오류 이미지와 Grad-CAM 이미지 생성
# - DATA : 원본 이미지 폴더
# - OUT : 결과 이미지 저장 폴더
# - model : 최적 가중치가 적용된 모델
# - chosen_pred : 최적 후보의 이미지별 예측 결과
# - explanations : 생성한 Grad-CAM 결과 정보 목록
explanations=save_explanations(DATA,OUT,model,chosen_pred)
# Grad-CAM 생성 결과 목록을 JSON 파일로 저장
# - json.dumps() : Python 목록을 JSON 문자열로 변환
# - ensure_ascii=False : 유니코드 문자를 그대로 저장
# - indent=2 : 들여쓰기를 적용하여 저장
(OUT/"cam_index.json").write_text(json.dumps(explanations,ensure_ascii=False,indent=2),encoding="utf-8")
# TODO D09: 선택 가중치와 클래스 매핑·입력 계약을 함께 저장하세요.
# BEGIN D09
# pass  # torch.save로 필요한 상태·계약을 저장하세요.

# 학습 데이터의 클래스 번호와 클래스 이름 대응 정보를 가져오기
# - datasets() : engine.py에 정의된 데이터셋 생성 함수
# - train : 학습 데이터셋
# - train.class_to_idx : 클래스 이름을 정수 라벨로 연결하는 딕셔너리
from engine import datasets
train,_ = datasets(DATA)
# 최적 모델의 가중치와 입력 데이터 규칙을 함께 저장
# - model_state_dict : 최적 후보의 모델 가중치
# - class_to_idx : 학습 데이터의 클래스 이름과 라벨 번호 대응 정보
# - input_contract : 모델을 사용할 때 지켜야 하는 입력 이미지 규칙
# - image_size : 입력 이미지의 높이와 너비
# - color_mode : 입력 이미지의 색상 모드
# - tensor_layout : Tensor의 차원 순서
# - dtype : Tensor의 데이터 형식
# - value_range : 픽셀값의 범위
# - transform : 학습 시 사용한 이미지 전처리 과정
# - torch.save() : 모델 가중치와 관련 정보를 파일로 저장
torch.save({
    "model_state_dict":chosen_state,
    "class_to_idx":train.class_to_idx,
    "input_contract":{
        "image_size":(32,32),
        "color_mode":"RGB",
        "tensor_layout":"CHW",
        "dtype":"float32",
        "value_range":(0.0,1.0),
        "transform":"Resize((32,32)) + ToTensor()"
    }
},OUT/"best_model.pt")

# END D09
# 후보별 평가 결과를 화면에 출력
# - pd.DataFrame(rows) : 후보별 결과를 표로 변환
# - drop(columns="seconds") : 학습 시간 열을 출력에서 제외
# - round(4) : 숫자를 소수점 아래 네 자리까지 표시
# - to_string(index=False) : 행 번호 없이 문자열로 출력
print(pd.DataFrame(rows).drop(columns="seconds").round(4).to_string(index=False))
# 최종 선택된 후보와 test 미사용 여부 출력
print("선택:",chosen["strategy"],chosen["balance"],"test 사용: False")
