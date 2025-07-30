#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
from datetime import datetime

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def get_config():
    parser = argparse.ArgumentParser()

    # Logging & model folders
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_dir', type=str, default=None)

    # Planning horizon
    parser.add_argument('--TLow', type=int, default=100)
    parser.add_argument('--TUp', type=int, default=100)
    parser.add_argument('--Ttest', type=int, default=100)

    # Demand distribution
    parser.add_argument('--demandDistribution', type=int, default=1)
    parser.add_argument('--demandLow', type=int, default=1)
    parser.add_argument('--demandUp', type=int, default=8)
    parser.add_argument('--demandMu', type=float, default=4.0)
    parser.add_argument('--demandSigma', type=float, default=2.0)

    # Cost parameters
    parser.add_argument('--ch', type=float, default=0.5)
    parser.add_argument('--cp', type=float, default=1.0)

    # Lead times
    parser.add_argument('--leadRecItem', type=int, default=2)
    parser.add_argument('--leadRecOrder', type=int, default=2)

    # Initial inventory
    parser.add_argument('--ILInit', type=int, default=12)
    parser.add_argument('--AOInit', type=int, default=4)
    parser.add_argument('--ASInit', type=int, default=4)

    # Early-Warning parameters
    parser.add_argument('--pd', type=float, default=0.9)
    parser.add_argument('--pf', type=float, default=0.1)

    # State & action space
    parser.add_argument('--stateDim', type=int, default=6)  # Standard DQN
    parser.add_argument('--stateDimEWA', type=int, default=7)  # EWA has 2 extra features and deletes average demand tracking
    parser.add_argument('--actionLow', type=int, default=0)
    parser.add_argument('--actionUp', type=int, default=20)
    parser.add_argument('--action_step', type=int, default=1)
    parser.add_argument('--fixedAction', type=str2bool, default=False)

    # DQN hyperparameters
    parser.add_argument('--maxEpisodesTrain', type=int, default=15000)
    parser.add_argument('--batchSize', type=int, default=32,
                        help='Smaller batch for more frequent updates')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='Back to original learning rate')
    parser.add_argument('--epsilonBeg', type=float, default=0.99,
                        help='Very high initial exploration')
    parser.add_argument('--epsilonEnd', type=float, default=0.01,
                        help='Lower final epsilon')
    parser.add_argument('--multPerdInpt', type=int, default=10)
    parser.add_argument('--seed_run', type=int, default=0)
    parser.add_argument('--NoHiLayer', type=int, default=2)
    parser.add_argument('--node1', type=int, default=64,
                        help='Back to original size')
    parser.add_argument('--node2', type=int, default=64)
    parser.add_argument('--node3', type=int, default=0)
    parser.add_argument('--iftl', type=str2bool, default=False)
    parser.add_argument('--NoFixedLayer', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Fewer warmup steps')
    parser.add_argument('--ifUsePreviousModel', type=str2bool, default=False)
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.5)
    parser.add_argument('--number_cpu_active', type=int, default=4)

    # DQN memory and update parameters
    parser.add_argument('--minReplayMem', type=int, default=1000,
                        help='Start training sooner')
    parser.add_argument('--maxReplayMem', type=int, default=50000,
                        help='Smaller memory for recent experiences')
    parser.add_argument('--saveInterval', type=int, default=5000)
    parser.add_argument('--dnnUpCnt', type=int, default=1000,
                        help='Less frequent target updates')

    # Sterman parameters
    parser.add_argument('--alpha_b', type=float, default=0.36)
    parser.add_argument('--betta_b', type=float, default=0.26)

    # Agent types
    parser.add_argument('--agentTypes', nargs='+',
                        default=['Strm', 'srdqn', 'ewa'])

    # Testing
    parser.add_argument('--testRepeatMid', type=int, default=50)
    parser.add_argument('--testInterval', type=int, default=100)

    # Early stopping
    parser.add_argument('--patience', type=int, default=2000)
    parser.add_argument('--min_episodes', type=int, default=3000)

    # Plotting
    parser.add_argument('--if_titled_figure', type=str2bool, default=False)

    # EWA-specific training parameters
    parser.add_argument('--ewa_lr0', type=float, default=0.0008,
                        help='Slightly lower learning rate for EWA')
    parser.add_argument('--ewa_warmup', type=int, default=1500,
                        help='More warmup for EWA to understand warnings')

    args = parser.parse_args()

    # Directories
    os.makedirs(args.log_dir, exist_ok=True)
    if args.model_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_dir = os.path.join(args.log_dir, f"run_{ts}")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.join(args.model_dir, 'saved_figures'), exist_ok=True)

    # Build action list
    args.actionList = list(range(args.actionLow,
                                 args.actionUp + 1,
                                 args.action_step))
    args.actionListLen = len(args.actionList)
    args.baseActionSize = args.actionListLen
    args.actionListOpt = args.actionList[:]
    args.actionListLenOpt = len(args.actionListOpt)

    # Network structure
    args.nodes = [args.stateDim * args.multPerdInpt,
                  args.node1,
                  args.node2,
                  args.node3]

    # Pack scalars
    args.c_h = args.ch
    args.c_p = args.cp
    args.eta = 1.0

    return args