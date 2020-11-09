import wandb
import torch
import os
from src.MF_env import MultiFidelityEnv
from src.MSE_numba import MSE_backend
from src.iDQN.policy import NN_policy
from src.iDQN.model import Qnetwork
from src.MF_env.paser import  parse_senario

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser(description='Display DQN policy')
    parser.add_argument('--dir', type=str, help='dir of DQN policy')
    args = parser.parse_args()
    return args


def get_env(scenario_name, step_t, sim_t, use_gui=False, sync_step = False):
    scenario = parse_senario(scenario_name)
    backend = MSE_backend.MSE_backend(scenario=scenario, step_t=step_t, sim_t=sim_t)
    backend.use_gui = use_gui
    env = MultiFidelityEnv.MultiFidelityEnv(scenario,backend,sync_step)
    return env

 
if __name__ == '__main__':
    api = wandb.Api()
    args = parse_arg()
    run = api.run(args.dir)
    args_dict = run.config
    eval_env = get_env(args_dict['eval_env'], step_t=args_dict['step_t'], sim_t=0.1, use_gui = True)
    eval_env.reset_rollout()


    
    Q_network = Qnetwork(nb_pos=args_dict['nb_pos'],
                nb_laser=args_dict['nb_laser'],
                nb_actions=args_dict['nb_actions'],
                hidden1 = args_dict['hidden1'],
                hidden2 = args_dict['hidden2'] ,
                layer_norm = args_dict['layer_norm'])
    run.file("Qnetwork.pkl").download(root='./tmp')
    Q_network_dict = torch.load('./tmp/Qnetwork.pkl')

    Q_network.load_state_dict(Q_network_dict.state_dict(), strict=True)
    policy = NN_policy(Q_network,0)
    eval_env.rollout_sync(policy.inference)
    results = eval_env.get_result()