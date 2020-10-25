import time
from .MF_env import MultiFidelityEnv
from .MSE import MSE_backend
from .DDPG.trainer import DDPG_trainer
from .MF_env.paser import  parse_senario
import numpy as np
import torch
import random

def get_env(scenario_name, step_t, sim_t, use_gui=False):
    scenario = parse_senario(scenario_name)
    backend = MSE_backend.MSE_backend(scenario=scenario, step_t=step_t, sim_t=sim_t)
    backend.use_gui = use_gui
    env = MultiFidelityEnv.MultiFidelityEnv(scenario,backend)
    return env

#def run_ddpg(env_args, train_args, agent_args, model_args, run_instance = None):
def run_ddpg(args_dict, run_instance = None):
    # set seed for reproducibility
    manualSeed = args_dict['seed']
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    train_env = get_env(args_dict['train_env'], step_t=args_dict['step_t'], sim_t=args_dict['train_dt'])
    search_env = get_env(args_dict['search_env'], step_t=args_dict['step_t'], sim_t=args_dict['search_dt'])
    eval_env = get_env(args_dict['eval_env'], step_t=args_dict['step_t'], sim_t=args_dict['eval_dt'])

    trainer = DDPG_trainer(args_dict)
    trainer.setup(train_env, search_env, eval_env)
    trainer.cuda()
    total_cycle = 0
    for epoch in range(args_dict['nb_epoch']):
        epsilon = 0.1**((epoch/args_dict['nb_epoch'])/args_dict['decay_coef'])
        for cycle in range(args_dict['nb_cycles_per_epoch']):
            log_info = trainer.cycle(epsilon = epsilon, train_actor = epoch>0)
            if run_instance is not None:
                run_instance.log(log_info,step=total_cycle)
            log_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +\
                       'epoch: %d/%d, '%(epoch+1, args_dict['nb_epoch']) +\
                       'cycle: %d/%d, '%(total_cycle+1,args_dict['nb_epoch']*args_dict['nb_cycles_per_epoch']) +\
                       'total_reward: %f, '%(log_info['total_reward']) +\
                       'crash_time: %f, '%(log_info['crash_time']) +\
                       'reach_time: %f, '%(log_info['reach_time'])
            print(log_info)
            total_cycle += 1



if __name__ == '__main__':
    agent_args = {'actor_lr': 0.001,
                  'critic_lr': 0.01,
                  'lr_decay': 10.0,
                  'l2_critic': 0.01,
                  'discount': 0.99,
                  'tau': 0.001,
                  'batch_size': 256,
                  'buffer_size': 1e6}
    
    train_args = {'nb_epoch': 400,
                  'nb_cycles_per_epoch': 10,
                  'nb_rollout_steps': 100,
                  'nb_train_steps': 20,
                  'nb_warmup_steps': 100,
                  'train_mode': 1,
                  'decay_coef': 0.2,
                  'search_method': 0,
                  'back_step': 7,
                  'seed': 0,
                  'expert_file': './expert.pkl'}
    
    model_args = {'hidden1': 128,
                  'hidden2': 128,
                  'layer_norm': False,
                  'nb_pos': 2,
                  'nb_laser': 128,
                  'nb_actions': 2}
    
    env_args = {'train_env': './src/scenario/scenario_train.yaml',
                'search_env': './src/scenario/scenario_train.yaml',
                'eval_env': './src/scenario/scenario_train.yaml',
                'step_t': 1.0,
                'train_dt': 0.001,
                'search_dt': 1.0,
                'eval_dt': 0.001}
    args_dict = {}
    args_dict.update(agent_args)
    args_dict.update(train_args)
    args_dict.update(model_args)
    args_dict.update(env_args)
    
    run_ddpg(args_dict, model_args, run_instance = None)