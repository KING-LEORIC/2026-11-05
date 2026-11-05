"""Grad-CAM과 오류 이미지 저장. 임의 예시 열지도를 성능 증거로 만들지 않습니다."""
# Path : 파일 및 폴더 경로 처리
from pathlib import Path
# NumPy : 이미지 배열과 수치 연산
import numpy as np
# PIL : 이미지 읽기, 이미지 위에 글자 쓰기
from PIL import Image,ImageDraw
# OpenCV : 이미지 색상 변환 및 열지도 합성
import cv2
# PyTorch : 딥러닝 연산
import torch


# Grad-CAM을 계산하는 함수
# - model : CAM을 계산할 학습된 신경망
# - x : CAM을 생성할 입력 이미지 Tensor
# - target : CAM을 계산할 대상 클래스 번호
def cam_for(model,x,target):
    """소형 Sequential 전용. Parameter의 기존 .grad를 건드리지 않습니다."""
    # 계산 전 각 계층의 학습·평가 모드 저장 및 모델을 평가 모드로 전환
    # - model.modules() : 모델을 구성하는 모든 계층 순회
    # - m.training : 각 계층의 현재 학습 모드 여부
    flags=[(m,m.training) for m in model.modules()];model.eval()
    # CAM 계산 중 오류가 발생하더라도 원래 모드를 복구하도록 try/finally 사용
    try:
        # 평가 모드에서도 CAM 계산에 필요한 기울기는 활성화
        # - torch.enable_grad() : 기울기 계산 허용
        with torch.enable_grad():
            # 입력 이미지의 복사본을 만들고 기울기 계산이 가능하도록 설정
            # - detach() : 기존 계산 그래프와 분리
            # - clone() : Tensor 복사본 생성
            # - requires_grad_(True) : 입력 이미지에 대한 기울기 계산 허용
            image=x.detach().clone().requires_grad_(True)
            # 모델의 앞부분과 뒷부분을 나누어 실행
            # - model[:7] : 앞쪽 합성곱 계층을 실행하여 특징 지도 생성
            # - activation : 합성곱 계층에서 추출한 특징 지도
            # - model[7:] : 나머지 계층을 실행하여 최종 클래스 점수 계산
            # - logits : 클래스별 예측 점수
            activation=model[:7](image);logits=model[7:](activation)
            # 지정한 클래스 점수에 대한 특징 지도의 기울기 계산 //기존 Parameter.grad를 변경하지 않음
            # - logits[0,target] : 첫 번째 이미지에서 target 클래스의 점수
            # - torch.autograd.grad() : 지정한 출력값에 대한 입력값의 기울기 계산
            # - grad : 특징 지도 각 위치의 기울기
            grad=torch.autograd.grad(logits[0,target],activation)[0]
            # Grad-CAM의 기본 가중합 계산
            # - grad.mean((2,3),keepdim=True) : 높이와 너비 방향으로 기울기 평균 계산
            # - activation : 합성곱 계층에서 추출한 특징 지도
            # - 곱한 뒤 sum(1) : 채널별 결과를 합산
            # - relu() : 음수 값을 0으로 처리
            cam=(grad.mean((2,3),keepdim=True)*activation).sum(1).relu()
            # CAM 크기를 입력 이미지 크기로 확대
            # - unsqueeze(1) : 채널 차원 추가
            # - interpolate() : 특징 지도를 지정한 크기로 보간
            # - mode='bilinear' : 양선형 보간 사용
            # - align_corners=False : 모서리 정렬 방식을 비활성화
            # - [0,0] : 배치와 채널 차원 제거
            cam=torch.nn.functional.interpolate(cam.unsqueeze(1),x.shape[-2:],mode='bilinear',align_corners=False)[0,0]
            # CAM 값을 0~1 범위로 정규화한 NumPy 배열 반환
            # - cam.max() : CAM의 최댓값
            # - clamp_min(1e-8) : 0으로 나누는 상황 방지
            # - detach() : 계산 그래프에서 분리
            # - cpu() : CPU 메모리로 이동
            # - numpy() : NumPy 배열로 변환
            return (cam/cam.max().clamp_min(1e-8)).detach().cpu().numpy()
    # CAM 계산이 끝나면 각 계층의 학습·평가 모드를 원래 상태로 복구
    finally:
        for m,flag in flags:m.training=flag


# 오답 이미지와 Grad-CAM 이미지를 저장하는 함수
# - data : 이미지 데이터가 저장된 폴더 경로
# - out : 결과 이미지 파일을 저장할 폴더 경로
# - model : 학습이 완료된 모델
# - prediction : 이미지별 실제 정답과 예측 결과가 담긴 DataFrame
def save_explanations(data,out,model,prediction):
    """정답·오답 각각 최대 1장 CAM, 실제 오답 최대 8장. 없는 오류를 조작하지 않습니다."""
    # 학습·검증 데이터셋을 불러오는 함수
    # - datasets : engine.py에 정의된 데이터셋 생성 함수
    from engine import datasets
    # 검증 데이터셋을 불러오고 결과 폴더를 준비
    # - datasets(data) : 학습 및 검증 데이터셋 생성
    # - valid : 검증 데이터셋
    # - Path(out) : 결과 폴더 경로를 Path 객체로 변환
    # - mkdir(exist_ok=True) : 폴더가 없으면 생성하고 이미 있으면 유지
    _,valid=datasets(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # 이전 실행에서 생성된 오류 이미지 파일 삭제
    # - glob('error_*.png') : error_로 시작하는 PNG 파일 검색
    # - unlink() : 해당 파일 삭제
    for stale in out.glob('error_*.png'):stale.unlink()
    # 이전 실행에서 생성된 Grad-CAM 이미지 파일 삭제
    # - glob('cam_*.png') : cam_으로 시작하는 PNG 파일 검색
    # - unlink() : 해당 파일 삭제
    for stale in out.glob('cam_*.png'):stale.unlink()
    # 실제 오답 이미지와 정답 이미지의 인덱스 분리
    # - prediction.actual : 실제 정답 라벨
    # - prediction.predicted : 모델의 예측 라벨
    # - actual!=predicted : 실제 정답과 예측이 다른 행
    # - actual==predicted : 실제 정답과 예측이 같은 행
    # - tolist() : 인덱스를 Python 목록으로 변환
    wrong=prediction.index[prediction.actual!=prediction.predicted].tolist()
    correct=prediction.index[prediction.actual==prediction.predicted].tolist()
    # 실제 오답 중 최대 8장의 이미지 저장
    # - wrong[:8] : 오답 인덱스 중 앞에서부터 최대 8개 선택
    for index in wrong[:8]:
        # 현재 오답 이미지의 예측 정보를 가져오고 원본 이미지 열기
        # - prediction.loc[index] : 해당 인덱스의 예측 정보
        # - Image.open() : 이미지 파일 열기
        # - convert('RGB') : RGB 색상 이미지로 변환
        # - resize((192,192)) : 이미지 크기를 192×192로 변경
        row=prediction.loc[index];image=Image.open(Path(data)/row.path).convert('RGB').resize((192,192))
        # 이미지와 정답·예측 정보를 하나의 이미지로 구성
        # - Image.new() : 흰색 배경 이미지 생성
        # - tile.paste() : 원본 이미지를 배경에 붙여넣기
        # - ImageDraw.Draw() : 이미지 위에 글자를 그리는 객체 생성
        tile=Image.new('RGB',(192,240),'white');tile.paste(image,(0,0));d=ImageDraw.Draw(tile)
        # 실제 정답 클래스 이름을 이미지 아래쪽에 표시
        # - valid.classes : 클래스 번호와 클래스 이름의 대응 목록
        # - row.actual : 실제 정답 클래스 번호
        # - fill='black' : 글자 색상을 검정색으로 지정
        d.text((4,195),f'true: {valid.classes[int(row.actual)]}',fill='black')
        # 모델이 예측한 클래스 이름을 이미지 아래쪽에 표시
        # - row.predicted : 모델이 예측한 클래스 번호
        d.text((4,210),f'pred: {valid.classes[int(row.predicted)]}',fill='black')
        # 모델의 예측 확률을 이미지 아래쪽에 표시하고 파일 저장
        # - row.confidence : 가장 높은 예측 확률
        # - :.3f : 소수점 아래 세 자리까지 표시
        # - tile.save() : 구성한 이미지 파일 저장
        d.text((4,225),f'p={row.confidence:.3f}',fill='black');tile.save(out/f'error_{index:03d}.png')
    # Grad-CAM 결과 정보를 저장할 목록 생성
    records=[]
    # 정답 이미지와 오답 이미지를 각각 확인
    # - correct : 정답으로 예측한 이미지 인덱스
    # - wrong : 오답으로 예측한 이미지 인덱스
    for kind,indices in [('correct',correct),('wrong',wrong)]:
        # 해당 종류의 이미지가 없으면 다음 종류로 이동
        if not indices:continue
        # 해당 종류에서 첫 번째 이미지 선택 후 Grad-CAM 계산
        # - index : 선택한 이미지의 인덱스
        # - valid[index] : 이미지 Tensor와 실제 라벨
        # - target : CAM을 계산할 예측 클래스 번호
        # - cam_for() : 선택한 이미지의 Grad-CAM 생성
        index=indices[0];x,_=valid[index];target=int(prediction.loc[index,'predicted']);cam=cam_for(model,x[None],target)
        # 이미지 Tensor를 OpenCV에서 사용할 수 있는 RGB 배열로 변환
        # - permute(1,2,0) : CHW 배열을 HWC 배열로 변경
        # - *255 : 0~1 범위의 픽셀값을 0~255 범위로 변환
        # - np.uint8() : 8비트 정수형 배열로 변환
        rgb=np.uint8(x.permute(1,2,0).numpy()*255)
        # 원본 이미지와 Grad-CAM 색상 지도를 합성
        # - cv2.cvtColor() : RGB 이미지를 BGR 이미지로 변환
        # - cv2.applyColorMap() : CAM에 색상 지도 적용
        # - COLORMAP_JET : 파랑·초록·빨강 계열 색상 지도
        # - addWeighted() : 두 이미지를 지정한 비율로 합성
        # - .6 : 원본 이미지 비율
        # - .4 : Grad-CAM 이미지 비율
        overlay=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.6,cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET),.4,0)
        # 합성 이미지를 256×256 크기로 변경하여 파일 저장
        # - cv2.resize() : 이미지 크기 변경
        # - cv2.imwrite() : 이미지 파일 저장
        cv2.imwrite(str(out/f'cam_{kind}.png'),cv2.resize(overlay,(256,256)))
        # 생성한 Grad-CAM 이미지의 정보를 목록에 기록
        # - kind : 정답 이미지인지 오답 이미지인지 구분
        # - path : 이미지 파일의 상대 경로
        # - target_class : Grad-CAM을 계산한 대상 클래스
        records.append({'kind':kind,'path':prediction.loc[index,'path'],'target_class':target})
    # 정답·오답 Grad-CAM 생성 결과 목록 반환
    return records
