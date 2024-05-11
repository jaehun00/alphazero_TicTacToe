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
----
  game.py
  - State : 틱택토 게임의 상태를 정의하는 클래스 (패배 및 무승부 확인하는 과정을 포함)
  - random_action : 주어진 상태에 대해 무작위 액션을 선택하는 함수
  - alpha_beta_action : 알파-베타 미니 맥스 알고리즘을 기반으로 액션을 선택하는 함수
  - mcts_action : 몬테카를로 트리 탐색 알고리즘을 기반으로 액션을 선택하는 함수

  dual_network.py
  - ResidualBlock : Resnet의 기반이 되는 ResidualBlock클래스
  - DualNetwork : 틱택토 게임의 상태를 평가하는 신경망 정책과 가치를 계산

   mcts.py    
  - predict : 신경망 모델을 사용하여 주어진 게임상태에 대한 정책과 가치를 예측
  - Node : 현재상태, 신경망, 해당 노드의 방문횟수, 승리횟수, 자식노드
  - pv_mcts_scores : 주어진 게임상태에 대해 MCTS 행동에 대한 점수 계산
  - pv_mcts_action : 다음수를 결정하는 함수를 반환

   data.py
  - TicTacToeDataset: 입력 : (2,3,3)의 보드 상태, 출력 : 정책(policies),가치(value)
  
   self_play.py
  - self_play : 각 게임의 결과 history 리스트에 저장
  
  train_network.py
  - DataLoader : Tic Tac Toe Dataset 로드
  - LossFunction : MSELoss <= CrossEntropy, Value 예측
  - Optimizer : Adam
 
  evaluate_network.py
  - evaluate_network : 모델 평가
  - update_best_player : 가장 좋은 모델 가중치 업데이트
    
  train.py
  - train을 위한 함수

# 모델 학습
