![header](https://capsule-render.vercel.app/api?type=slice&color=auto&height=200&section=header&text=Hello&rotate=13&fontAlign=70&fontAlignY=27&desc=I'm%20JaeHun&descAlign=75&descAlignY=45&fontSize=65)

<br/>
:point_down: 프로젝트 명

Pytorch 와 CUDA를 활용하여 Tic Tac Toe 모델을 직접 훈련시키고 시켜 자동 대국 AI 설계

<br/>
:point_down: 프로젝트 명 프로젝트 개발 환경

모델 훈련 환경

    GPU     : RTX 3060 12gb
    python  : 3.8.19
    CUDA    : 11.7
    torch   : 2.0.0

모델 사용 환경

    Jetson Orin NX 16gb

<br/>
:point_down: 개발 기간

2023-09-01 ~ 2024-12-24

<br/>
:point_down: 개발인원

:blush: 3명

<br/>
:point_down:  프로젝트 담당

Tensorflow 코드 해석, 분석<br/>
evaluate part Pytorch 설계<br/>
TicTacToe GUI 설계<br/>

<br/>
:point_down:  개발 tool

Python, PyTorch

<br/>
:point_down: 진행 사유

전공수업 텀프로젝트

<br/>
:point_down:  관련 전공

AI, 임베디드

<br/>
:point_down:  프로젝트 소개

강화학습을 기반으로 한 구글에서 개발한 Alpha-Zero 모델의 구조를 분석하고 이를 PyTorch로 구현 및 직접학습

<br/>
:point_down:  프로젝트 내용

<details>
<summary>
  :hash: game.py
</summary>
  - State : 틱택토 게임의 상태를 정의하는 클래스 (패배 및 무승부 확인하는 과정을 포함)<br/>
  - random_action : 주어진 상태에 대해 무작위 액션을 선택하는 함수<br/>
  - alpha_beta_action : 알파-베타 미니 맥스 알고리즘을 기반으로 액션을 선택하는 함수<br/>
  - mcts_action : 몬테카를로 트리 탐색 알고리즘을 기반으로 액션을 선택하는 함수<br/>
</details>

<details>
<summary>
  :hash: dual_network.py
</summary>
  - ResidualBlock : Resnet의 기반이 되는 ResidualBlock클래스<br/>
  - DualNetwork : 틱택토 게임의 상태를 평가하는 신경망 정책과 가치를 계산<br/>
</details>

<details>
<summary>
  :hash: mcts.py 
</summary>
  - predict : 신경망 모델을 사용하여 주어진 게임상태에 대한 정책과 가치를 예측<br/>
  - Node : 현재상태, 신경망, 해당 노드의 방문횟수, 승리횟수, 자식노드<br/>
  - pv_mcts_scores : 주어진 게임상태에 대해 MCTS 행동에 대한 점수 계산<br/>
  - pv_mcts_action : 다음수를 결정하는 함수를 반환<br/>
</details>

<details>
<summary>
  :hash: data.py
</summary>
  - TicTacToeDataset: 입력 : (2,3,3)의 보드 상태, 출력 : 정책(policies),가치(value)<br/>
</details>

<details>
<summary>
  :hash: self_play.py
</summary>
  - self_play : 각 게임의 결과 history 리스트에 저장<br/>
</details>

<details>
<summary>
  :hash: train.py
</summary>
  - DataLoader : Tic Tac Toe Dataset 로드<br/>
  - LossFunction : MSELoss <= CrossEntropy, Value 예측<br/>
  - Optimizer : Adam<br/>
</details>

<details>
<summary>
  :hash: evaluate_network.py
</summary>
  - evaluate_network : 모델 평가<br/>
  - update_best_player : 가장 좋은 모델 가중치 업데이트<br/>
</details>

<details>
<summary>
  :hash: train_cycle.py
</summary>
  - train을 위한 함수
</details>

<details>
<summary>
  :hash: human_play.py
</summary>
  - tic tac toe GUI, tkinter 사용<br/>
</details>

모델학습

     python train_cycle.py

Play TicTacToe

     python humna_play.py

Submit 버튼을 누른 뒤 터미널 창에 좌표값 입력(1~9)
