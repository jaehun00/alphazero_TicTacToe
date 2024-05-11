import matplotlib.pyplot as plt
import torch
import argparse
import os

from dual_network import DualNetwork
from mcts import pv_mcts_action
from self_play import self_play
from train import train_network
from evaluate_network import *

parser = argparse.ArgumentParser(description="Reinforcement Train")

# model hyper parameter
parser.add_argument('--pv_eval_count', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--num_residual_block', type=int, default=16)
parser.add_argument('--num_filters', type=int, default=128)

# train hyper parameter
parser.add_argument('--self_count', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=4)

# evaluate hyper parameter
parser.add_argument('--eval_epochs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)

def plot_evaluate(Num, Random, AlphaBeta, mcts):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Num+1),Random, label='Random_average', marker='o')
    plt.plot(range(1, Num+1),AlphaBeta, label='AlphaBeta_average', marker='o')
    plt.plot(range(1, Num+1),mcts, label='mcts_average', marker='o')
    plt.xlabel('Update_Num')
    plt.ylabel('average_point')
    plt.xlim([0, Num+1])
    plt.ylim([0, 1])
    plt.title(f'evaluate_best_player')
    plt.legend()
    plt.grid()
    plt.savefig('./plot/train_cycle.png')
    plt.show()

def train_cycle(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice : ', device)
    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)
    net.device = device
    print(net.device)
    if not os.path.isfile('./model/best.pth'):
        torch.save(net.state_dict(), './model/best.pth')
        print('best file')

    Num = 0

    for epoch in range(args.epochs):
        print(f'\nTrain Cycle [{epoch} / {args.epochs}]')
        self_play(args, net)
        print('Done self_play')

        train_network(args, net)
        print('Done train_network')
        # evaluate_network
        update_best_player = evaluate_network(args, net)

        # 베스트 플레이어 평가
        if update_best_player:
            Num += 1
            evaluate_best_player(args, net)
        
        directory = './data'

        # 디렉토리 내의 모든 파일 삭제
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            os.unlink(filepath)

    #plot_evaluate(Num, Random, AlphaBeta, mcts)

if __name__ == '__main__':
    if not os.path.exists('./model'):
        os.mkdir('./model')
    args = parser.parse_args()
    train_cycle(args)