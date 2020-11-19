import wandb
import torch
import os
from src.MF_env import MultiFidelityEnv
from src.MSE_numba import MSE_backend
from src.DDPG.policy import NN_policy
from src.DDPG.model import Actor
from src.MF_env.paser import  parse_senario

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser(description='Display DDPG policy')
    parser.add_argument('--dir', type=str, help='dir of DDPG policy')
    args = parser.parse_args()
    return args


def get_env(scenario_name, step_t, sim_t, use_gui=False, sync_step = False):
    scenario = parse_senario(scenario_name)
    backend = MSE_backend.MSE_backend(scenario=scenario, step_t=step_t, sim_t=sim_t)
    backend.use_gui = use_gui
    env = MultiFidelityEnv.MultiFidelityEnv(scenario,backend,sync_step)
    return env

def get_actor(dir,args_dict):
    actor  = Actor( nb_pos=args_dict['nb_pos'],
                    nb_laser=args_dict['nb_laser'],
                    nb_actions=args_dict['nb_actions'],
                    hidden1 = args_dict['hidden1'],
                    hidden2 = args_dict['hidden2'] ,
                    layer_norm = args_dict['layer_norm'])
    actor_dict = torch.load('{}/files/actor.pkl'.format(dir) )
    actor.load_state_dict(actor_dict.state_dict(), strict=True)
    return actor
 
if __name__ == '__main__':
    api = wandb.Api()
    args = parse_arg()
    run = api.run(args.dir)
    run_id = args.dir.split('/')[1]
    args_dict = run.config
#    eval_env = get_env(args_dict['eval_env'], step_t=args_dict['step_t'], sim_t=args_dict['eval_dt'], use_gui = True)
    eval_env = get_env(args_dict['eval_env'], step_t=args_dict['step_t'], sim_t=0.1, use_gui = True)
    eval_env.reset_rollout()
    all_files = [f for f in os.listdir('./wandb/' )]
    for dir_name in all_files:
        if run_id in dir_name:
            path = './wandb/'+dir_name
            break
    actor = get_actor(path,args_dict)
    policy = NN_policy(actor,0)
    eval_env.rollout_sync(policy.inference)
    results = eval_env.get_result()
