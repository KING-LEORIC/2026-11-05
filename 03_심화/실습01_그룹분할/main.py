from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
# 현재 실행 중인 파일이 위치한 디렉터리 경로 객체 생성
BASE=Path(__file__).resolve().parent
# day14_vision 패키지를 import할 수 있도록 모듈 탐색 경로(sys.path)에 추가
sys.path.insert(0,str(BASE.parent/"day14_vision"))
# 데이터 폴더 경로
DATA=BASE.parent/"data"
# 출력 결과 저장 폴더 경로
OUT=BASE/"outputs"
# 출력 폴더가 없으면 생성
OUT.mkdir(exist_ok=True)
# CPU 스레드 수를 1개로 제한하여 실행 결과 재현성 향상
torch.set_num_threads(1)
# 실험 실행, 데이터셋, 평가 지표 모듈 로드
from engine import experiment,datasets,metrics
# 모델 생성 및 설정 함수 로드
from models import build_model,configure
# 데이터 분할 유틸리티 로드
from sklearn.model_selection import GroupShuffleSplit,train_test_split
# 데이터 메타정보(manifest) 로드
manifest=pd.read_csv(DATA/"manifest.csv")
# 실제 배포 분할은 이미 그룹 독립. 비교 시연에는 train 행만 사용합니다.
# train 세트만 추출하여 비교 실험에 사용
rows=manifest[manifest.split=="train"].reset_index(drop=True)
# 일반 무작위 분할 수행
# - 그룹 정보를 고려하지 않음
# - 전체 샘플 인덱스를 train/valid로 분할
train_i,valid_i=train_test_split(np.arange(len(rows)),test_size=.25,random_state=14)
# 무작위 분할 후 train과 valid에 동시에 등장한 group_id 확인
# - 값이 존재하면 데이터 누수(leakage) 가능성 존재
random_overlap=set(rows.iloc[train_i].group_id)&set(rows.iloc[valid_i].group_id)

"""랜덤 분할은 그룹 누수 발생 가능성이 있으며, GroupShuffleSplit로 이를 방지함"""
# 그룹 단위 분할 객체 생성
# - 동일 group_id는 반드시 같은 분할에만 배치
splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=14)
# TODO D01: groups에 group_id를 전달하고 두 인덱스를 꺼내세요.
# BEGIN D01

# 그룹 정보를 기준으로 train/valid 분할 수행
# - 동일한 group_id는 train과 valid에 동시에 포함되지 않음
# - 반환값은 train 인덱스와 valid 인덱스
group_train, group_valid = next(splitter.split(rows, groups=rows["group_id"]))  # GroupShuffleSplit의 인덱스로 바꾸세요.
# END D01
# 그룹 단위 분할 후 train과 valid에 중복 등장한 group_id 확인
# - 정상 동작 시 빈 집합이 됨
group_overlap=set(rows.iloc[group_train].group_id)&set(rows.iloc[group_valid].group_id)
# 무작위 분할 시 그룹 중복 개수 출력
print("이미지 무작위 분할 그룹 중복:",len(random_overlap))
# 그룹 단위 분할 시 그룹 중복 개수 출력
print("그룹 단위 분할 그룹 중복:",len(group_overlap))
print("주의: GroupShuffleSplit이 클래스 비율까지 보장하지 않습니다.")
# 분할 결과를 demo_split 컬럼으로 기록
# - group_train에 속하면 "train"
# - 나머지는 "valid"
# - 결과를 CSV 파일로 저장
rows.assign(demo_split=np.where(rows.index.isin(group_train),"train","valid")).to_csv(OUT/"group_split.csv",index=False)
