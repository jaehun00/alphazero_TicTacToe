# 프로젝트
----
Pytorch 와 CUDA를 활용하여 Tic Tac Toe 모델을 직접 훈련시키고 시켜 자동 대국 AI 설계


# 프로젝트 개발 환경
-----
모델 훈련 환경

  python  : 3.8.19
  CUDA    : 11.7
  torch   : 2.0.0

모델 사용 환경

  Jetson Orin NX 16gb

# 파일 구성

  game.py
  - State : 틱택토 게임의 상태를 정의하는 클래스 (패배 및 무승부 확인하는 과정을 포함)
  - random_action : 주어진 상태에 대해 무작위 액션을 선택하는 함수
  - alpha_beta_action : 알파-베타 미니 맥스 알고리즘을 기반으로 액션을 선택하는 함수
  - mcts_action : 몬테카를로 트리 탐색 알고리즘을 기반으로 액션을 선택하는 함수

  dual_network.py
 - ResidualBlock : Resnet의 기반이 되는 ResidualBlock클래스
 - DualNetwork : 틱택토 게임의 상태를 평가하는 신경망 정책과 가치를 계산
