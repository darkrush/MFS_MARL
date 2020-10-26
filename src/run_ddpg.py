import time
from .MF_env import MultiFidelityEnv
from .MSE import MSE_backend
from .DDPG.trainer import DDPG_trainer
from .MF_env.paser import  parse_senario
from .utils import process_bar,float2time
import numpy as np
import torch
import random

def get_env(scenario_name, step_t, sim_t, use_gui=False):
    scenario = parse_senario(scenario_name)
    backend = MSE_backend.MSE_backend(scenario=scenario, step_t=step_t, sim_t=sim_t)
    backend.use_gui = use_gui
    env = MultiFidelityEnv.MultiFidelityEnv(scenario,backend)
    return env

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

    # setup environment instance for trainning, searching policy and evaluation(Eval not used yet)
    train_env = get_env(args_dict['train_env'], step_t=args_dict['step_t'], sim_t=args_dict['train_dt'])
    search_env = get_env(args_dict['search_env'], step_t=args_dict['step_t'], sim_t=args_dict['search_dt'])
    eval_env = get_env(args_dict['eval_env'], step_t=args_dict['step_t'], sim_t=args_dict['eval_dt'])

    # setup DDPG trainer.
    trainer = DDPG_trainer(args_dict)
    trainer.setup(train_env, search_env, eval_env)
    trainer.cuda()
    
    # Trainning DDPG
    cycle_count = 0
    total_search_sample = 0
    total_train_sample = 0
    total_cycle = args_dict['nb_cycles_per_epoch']*args_dict['nb_epoch']

    # Init process_bar
    PB = process_bar(total_cycle)
    PB.start()
    for epoch in range(args_dict['nb_epoch']):
        # Calculate the epsilon decayed by epoch.
        # Which used for epsilon-greedy exploration and policy search exploration.
        epsilon = 0.1**((epoch/args_dict['nb_epoch'])/args_dict['decay_coef'])
        for cycle in range(args_dict['nb_cycles_per_epoch']):
            # Do training in cycle-way. In each cycle, rollout one trajectory and update critic and actor.
            log_info = trainer.cycle(epsilon = epsilon, train_actor = epoch>0)

            #Calculate total serach step in trainning
            total_search_sample += log_info['train_search_step']
            log_info['train_total_search_sample'] = total_search_sample

            #Calculate total train step in trainning
            total_train_sample += log_info['train_train_step']
            log_info['train_total_train_sample'] = total_train_sample

            # Do log recording by wandb.
            if run_instance is not None:
                run_instance.log(log_info)
            
            # process bar take a tik
            process_past, time_total, time_left = PB.tik()
            str_time_total = float2time(time_total)
            str_time_left = float2time(time_left)
            # print log info, epoch, cycle, total_reward, crash_time, reach_time.
            str_log_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +\
                       ' epoch: %d/%d, '%(epoch+1, args_dict['nb_epoch']) +\
                       'cycle: %d/%d, '%(cycle_count+1, total_cycle) +\
                       'process: %d%%, '%(process_past) +\
                       'time: %s/%s, '%(str_time_left,str_time_total) +\
                       'train_total_reward: %f, '%(log_info['train_total_reward']) +\
                       'train_crash_time: %f, '%(log_info['train_crash_time']) +\
                       'train_reach_time: %f, '%(log_info['train_reach_time'])
            print(str_log_info)
            
            
            # count the total cycle number 
            cycle_count += 1
            
        # save model
        trainer.save_model(run_instance.dir)

        # eval_task
        result = trainer.eval()
        if run_instance is not None:
            run_instance.log(result)