"""제공되는 학습·평가 골격. train/valid만 모델에 전달하고 test는 열지 않습니다."""
# Path : 운영체제에 따라 다른 파일 경로 표기 방식을 통일해서 처리
from pathlib import Path
# time : 학습에 걸린 시간 측정
# copy : 모델 가중치의 복사본 생성
import time,copy
# numpy : 배열 및 수치 연산 처리
import numpy as np
# pandas : 예측 결과 및 학습 기록을 표 형태로 관리
import pandas as pd
# torch : 딥러닝 연산을 위한 PyTorch 핵심 라이브러리
import torch
# nn : 신경망 계층과 손실 함수 사용
from torch import nn
# DataLoader : 데이터를 배치 단위로 모델에 전달
# WeightedRandomSampler : 클래스별 가중치에 따라 데이터를 추출 //클래스별 선택 확률을 조절하는 샘플러
from torch.utils.data import DataLoader,WeightedRandomSampler
# transforms : 이미지 크기 변경, 텐서 변환 등의 전처리
from torchvision import transforms as T
# ImageFolder : 폴더 이름을 클래스 이름으로 인식하여 이미지 데이터셋 생성
from torchvision.datasets import ImageFolder
# 분류 모델의 평가 지표 함수
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
# 모델 생성, 학습 파라미터 설정, 학습 모드 설정 함수
from models import build_model,configure,training_mode


# 학습 이미지와 검증 이미지를 불러오는 함수
# - data : 이미지 데이터가 저장된 상위 폴더 경로
def datasets(data):
    # 이미지 전처리 과정
    # - Resize((32,32)) : 모든 이미지를 높이 32, 너비 32 크기로 변경
    # - ToTensor() : 이미지를 PyTorch Tensor로 변환하고 픽셀값을 0~1 범위의 실수로 변환
    # - T.Compose() : 여러 전처리 작업을 지정한 순서대로 연결
    transform=T.Compose([T.Resize((32,32)),T.ToTensor()])
    # 학습 데이터셋과 검증 데이터셋 생성
    # - Path(data)/'train' : 학습 이미지가 저장된 폴더
    # - Path(data)/'valid' : 검증 이미지가 저장된 폴더
    # - ImageFolder() : 폴더 이름을 클래스 이름으로 인식하고 각 이미지에 정답 라벨 부여
    # - transform : 이미지를 불러올 때 위에서 지정한 전처리 적용
    # - return : 학습 데이터셋과 검증 데이터셋을 순서대로 반환
    return (ImageFolder(Path(data)/'train',transform),ImageFolder(Path(data)/'valid',transform))


# 실제 정답과 모델의 예측 확률을 이용해 평가 지표를 계산하는 함수
# - y : 이미지별 실제 정답 라벨
# - prob : 이미지별 클래스 예측 확률 배열
def metrics(y,prob):
    # 각 이미지에서 확률이 가장 높은 클래스의 인덱스를 예측 라벨로 선택
    # - prob : 이미지별 클래스 예측 확률
    # - argmax(1) : 클래스 방향으로 가장 큰 확률의 위치를 선택
    # - pred : 모델이 예측한 클래스 번호
    pred=prob.argmax(1)
    # 클래스별 정밀도·재현율·F1 점수 등 상세 평가 결과 계산
    # - labels=[0,1,2] : 평가할 클래스 번호
    # - output_dict=True : 결과를 딕셔너리 형태로 반환
    # - zero_division=0 : 0으로 나누는 상황이 발생하면 해당 지표를 0으로 처리
    report=classification_report(y,pred,labels=[0,1,2],output_dict=True,zero_division=0)
    # 전체 평가 지표를 딕셔너리로 반환
    # - accuracy : 전체 예측 중 정답을 맞힌 비율
    # - macro_f1 : 세 클래스의 F1 점수를 동일한 비중으로 평균한 값
    # - circle_recall : circle 클래스의 재현율
    # - square_recall : square 클래스의 재현율
    # - triangle_recall : triangle 클래스의 재현율
    # - float() : NumPy 등의 숫자 값을 Python 기본 실수형으로 변환
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'circle_recall':report['0']['recall'],'square_recall':report['1']['recall'],'triangle_recall':report['2']['recall']}


# 검증 데이터로 모델의 성능을 평가하는 함수
# - model : 평가할 신경망 모델
# - loader : 검증 데이터를 배치 단위로 제공하는 DataLoader
def evaluate(model,loader):
    """평가 손실은 가중치 없이 계산합니다. 모든 행을 한 번씩 평가하고 표본 수로 집계합니다."""
    # 평가 전 각 모듈의 학습 모드 상태를 저장
    # - model.modules() : 모델을 구성하는 모든 계층을 순회
    # - m.training : 해당 계층이 학습 모드인지 평가 모드인지 나타내는 값
    flags=[(m,m.training) for m in model.modules()]
    # 검증을 위한 변수 초기화 및 모델을 평가 모드로 전환
    # model.eval() : Dropout과 BatchNorm 등을 평가 방식으로 동작시킴
    # ys : 실제 정답 라벨을 모아 둘 목록
    # ps : 모델의 예측 확률을 모아 둘 목록
    # total : 검증 데이터 전체의 손실 합계
    # n : 지금까지 평가한 이미지 수
    model.eval();ys=[];ps=[];total=0.;n=0
    # 검증 단계에서는 기울기를 계산하지 않도록 설정
    # - torch.no_grad() : 역전파에 필요한 기울기 계산을 비활성화 //학습에 필요한 역전파가 없으므로 메모리 사용량 절약
    with torch.no_grad():
        # 검증 데이터를 배치 단위로 순회
        # - x : 현재 배치의 이미지 Tensor
        # - y : 현재 배치 이미지의 실제 정답 라벨
        for x,y in loader:
            # 현재 배치의 예측 점수와 손실 계산
            # 모델이 계산한 클래스별 점수(logits)
            # - model(x) : 이미지 배치를 모델에 입력하여 클래스별 점수(logits) 계산
            # CrossEntropyLoss의 배치별 손실을 합산
            # - cross_entropy() : 예측 점수와 실제 정답의 차이를 손실로 계산
            # - reduction='sum' : 배치 내 이미지들의 손실을 합계로 반환
            # - item() : Tensor의 숫자 값을 Python 숫자로 변환
            # 평가한 이미지 개수 누적
            # - len(y) : 현재 배치에 포함된 이미지 수
            logits=model(x);total+=nn.functional.cross_entropy(logits,y,reduction='sum').item();n+=len(y)
            # 실제 라벨과 예측 확률을 목록에 누적
            # - y.tolist() : 실제 라벨 Tensor를 Python 목록으로 변환
            # - logits.softmax(1) : 클래스별 점수를 확률로 변환
            # - cpu() : 예측 확률 Tensor를 CPU 메모리로 이동
            ys.extend(y.tolist());ps.append(logits.softmax(1).cpu())
    # 평가 전에 저장한 각 계층의 학습·평가 모드를 복구
    # - m : 모델을 구성하는 각 계층
    # - flag : 평가 전에 저장한 해당 계층의 모드
    for m,flag in flags:m.training=flag
    # 검증 결과를 계산하고 반환
    # - torch.cat(ps) : 배치별 예측 확률을 하나의 Tensor로 연결
    # - numpy() : Tensor를 NumPy 배열로 변환
    # - np.array(ys) : 실제 정답 목록을 NumPy 배열로 변환
    # - metrics() : 정확도, Macro F1, 클래스별 Recall 계산
    # - total/n : 전체 손실 합계를 이미지 수로 나누어 평균 손실 계산
    prob=torch.cat(ps).numpy();summary=metrics(np.array(ys),prob);summary['loss']=total/n
    # 평가 지표, 실제 라벨, 예측 확률 반환
    return summary,np.array(ys),prob


# 학습 전략과 데이터 불균형 처리 방식을 지정하여 모델을 학습하는 함수
# - data : 학습 및 검증 이미지가 저장된 폴더 경로
# - strategy : head, partial, full 중 학습할 모델 영역
# - balance : none, weight, sampler 중 클래스 불균형 처리 방식
# - epochs : 전체 학습 반복 횟수
# - seed : 난수 생성 기준값으로 실험 재현성을 높임
# - freeze_bn : BatchNorm의 통계값을 고정할지 여부
# - pretrained : 제공된 사전 학습 가중치를 사용할지 여부
# - switch_epoch : 지정한 epoch부터 partial 전략으로 변경하는 선택적 설정
# - return : 학습된 모델, 최종 검증 지표, epoch별 기록, 샘플별 예측 결과
def experiment(data, strategy='head', balance='none', epochs=8, seed=14, freeze_bn=True, pretrained=True, switch_epoch=None):
    """같은 분할·예산을 비교합니다. 모든 후보의 valid는 동일한 비가중 평가입니다."""
    # 클래스 불균형 처리 방식이 허용된 값인지 확인
    # - none : 별도 불균형 처리 없음
    # - weight : 손실 함수에 클래스별 가중치 적용
    # - sampler : 데이터를 추출할 때 클래스별 가중치 적용
    # - 허용되지 않은 값이면 ValueError 발생
    if balance not in {'none','weight','sampler'}:raise ValueError('unknown balance')
    # 학습·검증 데이터셋을 만들고 CPU 연산 스레드 수 지정
    # - torch.set_num_threads(1) : CPU 연산 스레드를 1개로 제한
    # - datasets(data) : 학습 데이터와 검증 데이터 불러오기
    # - train : 학습 데이터셋
    # - valid : 검증 데이터셋
    torch.set_num_threads(1);train,valid=datasets(data)
    # 학습 모델과 옵티마이저 생성
    # - build_model() : 모델 구조를 만들고 필요하면 사전 학습 가중치 적용
    # - configure() : strategy에 따라 학습 가능한 파라미터와 옵티마이저 설정
    # - head_lr=.01 : 분류 Head의 학습률
    # - backbone_lr=.001 : Backbone의 학습률
    model=build_model(data,pretrained,seed);optimizer=configure(model,strategy,head_lr=.01,backbone_lr=.001)
    # 데이터 추출에 사용할 난수 생성기 생성
    # - manual_seed(seed) : 난수 생성 기준값 지정
    # - 같은 seed를 사용하면 난수 기반 작업의 재현성을 높일 수 있음
    g=torch.Generator().manual_seed(seed)
    # 학습 데이터에서 클래스별 이미지 개수 계산
    # - train.targets : 학습 이미지 전체의 정수형 클래스 라벨 목록
    # - torch.tensor() : 라벨 목록을 Tensor로 변환
    # - torch.bincount() : 클래스별 데이터 개수 계산
    # - minlength=3 : 클래스가 3개이므로 결과 길이를 최소 3으로 지정
    # - float() : 이후 나눗셈을 위해 실수형으로 변환
    counts=torch.bincount(torch.tensor(train.targets),minlength=3).float()
    # 클래스별 데이터 개수의 역수를 가중치로 계산
    # - 1/counts : 데이터가 적은 클래스일수록 큰 가중치 부여
    # - class_weights : 클래스별 가중치
    # - /= class_weights.mean() : 가중치 평균이 1이 되도록 정규화
    class_weights=(1/counts);class_weights/=class_weights.mean()
    # 클래스별 가중치를 이용하는 샘플러 생성
    # - balance=='sampler' : sampler 방식일 때만 샘플러 사용
    # - (1/counts)[train.targets] : 각 이미지의 클래스에 해당하는 역빈도 가중치 부여
    # - len(train) : 한 epoch에서 추출할 이미지 수
    # - replacement=True : 같은 이미지가 중복 선택되는 것을 허용
    # - generator=g : 위에서 만든 난수 생성기 사용
    # - sampler가 필요하지 않으면 None 지정
    sampler=WeightedRandomSampler((1/counts)[train.targets],len(train),replacement=True,generator=g) if balance=='sampler' else None
    # 학습 데이터를 배치 단위로 제공하는 DataLoader 생성
    # - batch_size=16 : 한 번에 이미지 16개씩 모델에 전달
    # - shuffle=sampler is None : 샘플러가 없을 때만 데이터 순서를 무작위로 섞음
    # - sampler=sampler : 샘플러가 있으면 해당 가중치에 따라 데이터 추출
    # - generator=g : 난수 생성기 지정
    # - num_workers=0 : 별도의 데이터 로딩 프로세스를 사용하지 않음
    loader=DataLoader(train,batch_size=16,shuffle=sampler is None,sampler=sampler,generator=g,num_workers=0)
    # 검증 데이터를 배치 단위로 제공하는 DataLoader 생성
    # - batch_size=32 : 한 번에 이미지 32개씩 평가
    # - shuffle=False : 검증 데이터 순서를 섞지 않음
    # - num_workers=0 : 별도의 데이터 로딩 프로세스를 사용하지 않음
    vl=DataLoader(valid,batch_size=32,shuffle=False,num_workers=0)
    # 학습에 사용할 손실 함수 생성
    # - CrossEntropyLoss() : 다중 클래스 분류에 사용하는 손실 함수
    # - weight=class_weights : balance가 weight일 때 클래스별 가중치 적용
    # - 그 외에는 None을 전달하여 가중치 없이 계산
    loss_fn=nn.CrossEntropyLoss(weight=class_weights if balance=='weight' else None)
    # 학습 중 최적의 모델과 epoch별 기록을 저장할 변수 초기화
    # - best : 지금까지 가장 낮았던 검증 손실
    # - history : epoch별 학습 및 검증 기록
    # - best_state : 가장 좋은 검증 손실을 기록한 시점의 모델 가중치
    # - seen : 원본 코드에서 생성하지만 이후 사용하지 않는 목록
    best=float('inf');history=[];best_state=None;seen=[]
    # 전체 학습에 걸리는 시간 측정 시작
    start=time.perf_counter()
    # 지정된 epoch 수만큼 학습 반복
    # - range(1,epochs+1) : 1부터 epochs까지 반복
    # - epoch : 현재 학습 반복 번호
    for epoch in range(1,epochs+1):
        # 지정된 epoch에 도달하면 학습 전략을 partial로 변경
        # - switch_epoch is not None : 전략 변경 시점이 지정되었는지 확인
        # - epoch==switch_epoch : 현재 epoch가 변경 시점과 같은지 확인
        if switch_epoch is not None and epoch==switch_epoch:
            # 교육용 순차 비교: 옵티마이저 재생성 → 기존 momentum이 초기화됨을 보고서에 명시.
            # 기존 옵티마이저를 partial 전략에 맞는 새 옵티마이저로 교체
            optimizer=configure(model,'partial',head_lr=.01,backbone_lr=.001)
        # 모델을 학습 모드로 설정하고 BatchNorm 고정 여부 적용
        # - training_mode() : 모델을 학습 모드로 설정
        # - freeze_bn : BatchNorm 통계값을 고정할지 결정
        training_mode(model,freeze_bn)
        # 한 epoch의 학습 손실과 클래스별 추출 수를 누적할 변수 초기화
        # - numerator : 학습 손실의 가중 합계
        # - denominator : 학습 손실 평균 계산에 사용할 분모
        # - sample_counts : 클래스별로 추출된 이미지 수
        numerator=0.;denominator=0.;sample_counts=torch.zeros(3,dtype=torch.long)
        # 학습 데이터를 배치 단위로 순회
        # - x : 현재 배치의 이미지 Tensor
        # - y : 현재 배치의 실제 정답 라벨
        for x,y in loader:
            # 현재 배치의 기울기를 초기화하고 모델의 예측 점수 계산
            # - zero_grad(set_to_none=True) : 이전 배치의 기울기를 None으로 초기화
            # - model(x) : 이미지 배치를 모델에 입력하여 logits 계산
            optimizer.zero_grad(set_to_none=True);logits=model(x)
            # 예측값과 실제 정답의 차이를 계산하고 모델 가중치 갱신
            # - loss_fn(logits,y) : 예측 점수와 실제 정답으로 손실 계산
            # - backward() : 손실에 대한 각 파라미터의 기울기 계산
            # - optimizer.step() : 기울기를 이용해 모델 파라미터 갱신
            loss=loss_fn(logits,y);loss.backward();optimizer.step()
            # 현재 배치의 손실을 평균 계산에 사용할 가중치로 환산 //가중치 손실을 사용하면 가중치 합계를 분모로 사용, 그렇지 않으면 배치의 이미지 수를 사용
            # - balance=='weight' : 클래스 가중치를 사용한 경우
            # - class_weights[y].sum() : 배치에 포함된 각 이미지의 클래스 가중치 합계
            # - len(y) : 일반적인 경우 배치에 포함된 이미지 수
            denom=float(class_weights[y].sum()) if balance=='weight' else len(y)
            # 학습 손실과 클래스별 추출 이미지 수 누적
            # - loss.item()*denom : 배치 손실을 해당 분모 기준으로 환산
            # - denominator : 전체 분모 누적
            # - torch.bincount() : 현재 배치의 클래스별 이미지 수 계산
            numerator+=loss.item()*denom;denominator+=denom;sample_counts+=torch.bincount(y,minlength=3)
        # 한 epoch의 학습이 끝난 뒤 검증 데이터로 성능 평가
        # - summary : 검증 손실과 평가 지표
        # - 나머지 두 반환값은 이 위치에서 사용하지 않음
        summary,_,_=evaluate(model,vl)
        # 현재 epoch의 학습 및 검증 결과를 기록
        # - train_objective : 해당 epoch의 평균 학습 손실
        # - valid_loss : 검증 데이터의 평균 손실
        # - valid_macro_f1 : 검증 데이터의 Macro F1
        # - draw_circle/square/triangle : 학습 중 추출된 클래스별 이미지 수
        history.append({'epoch':epoch,'strategy':strategy,'balance':balance,'train_objective':numerator/denominator,'valid_loss':summary['loss'],'valid_macro_f1':summary['macro_f1'],'draw_circle':int(sample_counts[0]),'draw_square':int(sample_counts[1]),'draw_triangle':int(sample_counts[2])})
        # 현재 검증 손실이 지금까지의 최저 손실보다 낮으면 최적 모델 갱신
        # - best : 가장 낮은 검증 손실 갱신
        # - best_epoch : 최저 손실을 기록한 epoch 번호 기록
        # - copy.deepcopy() : 현재 모델 가중치를 독립된 복사본으로 저장
        if summary['loss']<best:
            best=summary['loss'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
    # 학습 종료 시각을 이용해 전체 학습 시간 계산
    # - perf_counter() : 고정밀 시간 측정
    # - seconds : 전체 학습에 걸린 시간(초)
    seconds=time.perf_counter()-start
    # 가장 좋은 검증 손실을 기록한 시점의 모델 가중치 복원
    # - load_state_dict() : 저장한 가중치를 현재 모델에 적용
    # - model.eval() : 모델을 평가 모드로 전환
    model.load_state_dict(best_state);model.eval()
    # 복원된 최적 모델을 이용해 검증 데이터 다시 평가
    # - summary : 최종 평가 지표
    # - y : 검증 데이터의 실제 정답 라벨
    # - prob : 검증 데이터의 클래스별 예측 확률
    summary,y,prob=evaluate(model,vl)
    # 최종 평가 지표에 학습 전략과 학습 관련 정보 추가
    # - strategy : 사용한 학습 전략
    # - balance : 사용한 클래스 불균형 처리 방식
    # - best_epoch : 가장 좋은 검증 손실을 기록한 epoch
    # - trainable : requires_grad=True인 파라미터의 전체 개수
    # - epochs : 설정한 전체 학습 epoch 수
    # - seconds : 실제 학습 소요 시간
    # - source : 사전 학습 가중치 사용 여부
    summary.update({'strategy':strategy,'balance':balance,'best_epoch':best_epoch,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'epochs':epochs,'seconds':seconds,'source':'synthetic_source' if pretrained else 'scratch'})
    # 검증 데이터 이미지별 예측 결과를 DataFrame으로 구성
    # - path : 데이터 폴더를 기준으로 한 이미지의 상대 경로
    # - actual : 이미지의 실제 정답 라벨
    # - predicted : 예측 확률이 가장 높은 클래스 번호
    # - confidence : 가장 높은 예측 확률
    prediction=pd.DataFrame({'path':[str(Path(p).relative_to(data)) for p,_ in valid.samples], 'actual':y,'predicted':prob.argmax(1),'confidence':prob.max(1)})
    # 각 이미지의 클래스별 예측 확률을 DataFrame에 열로 추가
    # - range(3) : 클래스 번호 0, 1, 2 순회
    # - prob[:,i] : 모든 이미지에 대한 i번째 클래스의 예측 확률
    # - p0 : circle 확률
    # - p1 : square 확률
    # - p2 : triangle 확률
    for i in range(3):prediction[f'p{i}']=prob[:,i]
    # 학습된 모델과 평가 결과 반환
    # - model : 최적 가중치가 적용된 모델
    # - summary : 최종 검증 평가 지표
    # - pd.DataFrame(history) : epoch별 학습 기록
    # - prediction : 이미지별 실제 라벨, 예측 라벨, 예측 확률
    return model,summary,pd.DataFrame(history),prediction
