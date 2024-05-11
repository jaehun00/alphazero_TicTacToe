import torch
import numpy as np
import os

from game import State
from pv_mcts import pv_mcts_action
from dual_network import DualNetwork
from pathlib import Path
from shutil import copy
import matplotlib.pyplot as plt


def visualize_board(board_state, step):
    player_pieces, enemy_pieces = board_state

    for i in range(3):
        row_visual = '    '

        for j in range(3):
            index = i * 3 + j
            if step % 2 != 0:
                if player_pieces[index] == 1:
                    row_visual += ' X '
                elif enemy_pieces[index] == 1:
                    row_visual += ' O '
                else:
                    row_visual += ' · '
            else:
                if player_pieces[index] == 1:
                    row_visual += ' O '
                elif enemy_pieces[index] == 1:
                    row_visual += ' X '
                else:
                    row_visual += ' · '

        print(row_visual)

# point of first player (1 : win / 0.5 : draw / 0 : lose)
def first_player_point(ended_state):
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

# play game once
def play(next_actions):
    state = State()

    while True:
        if state.is_done():
            break

        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        state = state.next(action)

    return first_player_point(state)
        
# change model
def update_best_player():
    copy('./model/latest.pth', './model/best.pth')
    print('Change BestPlayer')

def evaluate_network(args, device):
    game_count = args.game_count
    # load latest model
    net0 = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)
    net0.load_state_dict(torch.load('./model/latest.pth'))
    net0.eval()

    # load best model
    net1 = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)
    net1.load_state_dict(torch.load('./model/best.pth'))
    net1.eval()
    
    # using pv_mcts make function of action
    next_action0 = pv_mcts_action(net0, temperature=1.0, pv_eval_count=50, device = device)
    next_action1 = pv_mcts_action(net1, temperature=1.0, pv_eval_count=50, device = device)
    next_actions = (next_action0, next_action1)

    # Game loop
    total_point = 0
    points = []
    for i in range(game_count):
        if i % 2 == 0:
            point = play(next_actions)
        else:
            point = 1 - play(list(reversed(next_actions)))
        total_point += point
        points.append(point)
        print('\rEvaluate {}/{}'.format(i + 1, game_count), end='')
    print('')

    # calculate average point
    average_point = total_point / game_count
    print('AveragePoint', average_point)