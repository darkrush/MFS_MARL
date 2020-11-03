import torch
import random
from copy import deepcopy
from .DQN import DQN
from .policy import NN_policy, Mix_policy, Agent_Mix_policy

#ACTION_TABLE = np.array(((-1,-1),(-1,0),(-1,1),(0,0),(1,-1),(1,0),(1,1)))

def conti2dis(conti_action):
    dis_action = int(conti_action[0]*2+conti_action[1]+3)
    return dis_action

class Crash_Checker(object):
    def __init__(self):
        self.mask = []
        self.p = 1
    def set_mask(self,mask):
        self.mask = mask
    def set_p(self,p):
        self.p = p
    def check_crash(self,state_list):
        if random.random() < self.p:
            for idx,state in enumerate(state_list):
                if state.crash :
                    if idx in self.mask:
                        self.mask.remove(idx)
                        continue
                    else:
                        return True
        return False

class DQN_trainer(object):
#    def __init__(self, train_args, agent_args, model_args):
    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.nb_epoch = args_dict['nb_epoch']
        self.nb_cycles_per_epoch = args_dict['nb_cycles_per_epoch']
        self.nb_rollout_steps = args_dict['nb_rollout_steps']
        self.nb_train_steps = args_dict['nb_train_steps']
        self.nb_warmup_steps = args_dict['nb_warmup_steps']
        self.train_mode = args_dict['train_mode']
        self.search_method= args_dict['search_method']
        self.back_step = args_dict['back_step']
        self.expert_file = args_dict['expert_file']

        self.agent = DQN(args_dict)

    def setup(self, env_instance, search_env_instance, eval_env_instance):
        self.env = env_instance
        self.search_env = search_env_instance
        self.eval_env = eval_env_instance
        self.agent.setup()
        if self.search_method is 2 :
            self.expert_Qnetwork = torch.load(self.expert_file)

    def cuda(self):
        self.agent.cuda()
    
    def cycle(self, epsilon):
        self.env.reset_rollout()
        crash_checker = Crash_Checker()
        crash_checker.set_p(epsilon)
        #get trajection
        search_policy = None
        replace_table = None
        search_state_index= 0
        last_back_index = 0
        state_history = []
        obs_history = []
        time_history = []
        action_history = []
        search_step = 0
        while True:
            if self.search_method is 1 :
                rollout_policy = Mix_policy(self.agent.Qnetwork,epsilon,search_policy,replace_table)
                finish = self.env.rollout_sync(rollout_policy.inference, pause_call_back = crash_checker.check_crash)
            elif self.search_method is 2 :
                rollout_policy = Agent_Mix_policy(self.agent.Qnetwork,epsilon,self.expert_Qnetwork,replace_table)
                finish = self.env.rollout_sync(rollout_policy.inference, pause_call_back = crash_checker.check_crash)
            else:
                rollout_policy = Mix_policy(self.agent.Qnetwork,epsilon,None,None)
                finish = self.env.rollout_sync(rollout_policy.inference)
            if finish is 'finish' :
                break
            elif finish is 'pause':
                state_history,obs_history,time_history,action_history = self.env.get_history()
                crash_list = [state.crash for state in state_history[-1]]
                last_back_index= search_state_index
                search_state_index = max(len(state_history)-self.back_step,1)
                if self.search_method is 1 :
                    if search_state_index > last_back_index:
                        backtrack_state = deepcopy(state_history[search_state_index])
                        for b_State,crash in zip(backtrack_state,crash_list):
                            b_State.enable = crash
                        self.search_env.set_state(backtrack_state)
                        search_policy,replace_table,search_step = self.search_env.search_policy(multi_step = 1,back_number = 2)
                    else:
                        search_policy = None
                        replace_table = None

                elif self.search_method is 2:
                    if search_state_index > last_back_index:
                        replace_table = []
                        for idx,crash in enumerate(crash_list):
                            if crash:
                                replace_table.append(idx)
                    else:
                        replace_table = None
                mask = []
                if replace_table is None:
                    for idx,state in enumerate(state_history[-1]):
                        if state.crash:
                            mask.append(idx)
                    crash_checker.set_mask(mask)
                    state_history = state_history[:-1]
                    obs_history = obs_history[:-1]
                    time_history = time_history[:-1]
                else :
                    self.env.set_state(state_history[search_state_index],total_time = time_history[search_state_index])
                    state_history = state_history[:search_state_index]
                    obs_history = obs_history[:search_state_index]
                    time_history = time_history[:search_state_index]
                    action_history = action_history[:search_state_index]
                self.env.set_history(state_history,obs_history,time_history,action_history)

        trajectoy = self.env.get_trajectoy()
        train_sample = len(trajectoy[0])
        results = self.env.get_result()
        for agent_traj in (trajectoy):
            for trans in agent_traj:
                self.agent.memory.append( [trans['obs'].pos,trans['obs'].laser_data], 
                                    conti2dis([trans['action'].ctrl_vel,trans['action'].ctrl_phi]),
                                    trans['reward'],
                                    [trans['obs_next'].pos,trans['obs_next'].laser_data],
                                    trans['done'])
        Q_loss = self._apply_train()

        log_info = {'train_total_reward': results['total_reward'],
                    'train_crash_time': results['crash_time'],
                    'train_reach_time': results['reach_time'],
                    'train_mean_vel': results['mean_vel'],
                    'train_total_time': results['total_time'],
                    'train_Q_loss': Q_loss,
                    'train_search_step': search_step,
                    'train_train_step': train_sample}
        return log_info

    def save_model(self, model_dir):
        self.agent.save_model(model_dir)

    def apply_lr_decay(self, decay_sacle):
        self.agent.apply_lr_decay(decay_sacle)

    def eval(self):
        self.eval_env.reset_rollout()
        rollout_policy = NN_policy(self.agent.Qnetwork,0)
        self.eval_env.rollout_sync(rollout_policy.inference)
        results = self.eval_env.get_result()
        return results

    def _apply_train(self):
        #update agent for nb_train_steps times
        ql_list = []
        if self.train_mode == 0:
            for t_train in range(self.nb_train_steps):
                ql = self.agent.update_Qnetwork()
                self.agent.update_Qnetwork_target()
                ql_list.append(ql)
        elif self.train_mode == 1:
            for t_train in range(self.nb_train_steps):
                ql = self.agent.update_Qnetwork()
                ql_list.append(ql)
            self.agent.update_Qnetwork_target(soft_update = False)
        return sum(ql_list)/len(ql_list)
