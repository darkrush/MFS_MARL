#from  gym import Env
import copy
import numpy as np
import math
import time
from itertools import product
from .basic import AgentProp,Action,AgentState

def prefer_vel_list(vel,phi):
    if phi == 0:
        #return [(vel,0),(vel,-1),(vel,1),(-vel,0),(-vel,-1),(-vel,1)]
        return [(vel,0),(vel,-1),(vel,1)]
    else:
        #return [(vel,phi),(vel,-phi),(vel,0),(-vel,-phi),(-vel,phi),(-vel,0)]
        return [(vel,phi),(vel,-phi),(vel,0)]

def path(x,y,theta,target_x,target_y, max_phi = math.pi/6.0, l = 0.3, dist = 0.1):
    min_r = l/np.tan(max_phi)
    xt = target_x - x
    yt = target_y - y
    xt,yt = (xt*np.cos(theta)+yt*np.sin(theta),yt*np.cos(theta)-xt*np.sin(theta))
    if abs(yt) < dist*0.5:
        vel = np.sign(xt)
        phi = 0
    else:
        in_min_r = (xt**2+(abs(yt)-min_r)**2)< min_r**2
        vel = -1 if (bool(in_min_r) ^ bool(xt<0)) else 1
        phi = -1 if (bool(in_min_r) ^ bool(yt<0)) else 1
    return vel,phi


class SearchNode(object):
    def __init__(self,state_list,time,parent_action = None,parent_action_index = None):
        self.state_list = copy.deepcopy(state_list)
        self.time = time
        self.parent_action = parent_action
        self.parent_action_index = parent_action_index
        self.prefer_action_list = []
        self.enable_index_list = []
        if parent_action is None:
            self.parent_action =[]
            self.parent_action_index = []
            for state in self.state_list:
                self.parent_action.append([0,0])
                self.parent_action_index.append(0)
        action_index_list = []

        for index,state in enumerate(self.state_list):
            if state.enable and not state.reach:
                self.enable_index_list.append(index)
                vel,phi = path(state.x,state.y,state.theta,state.target_x,state.target_y)
                self.prefer_action_list.append(prefer_vel_list(vel,phi))
                action_index_list.append([i for i in range(len(self.prefer_action_list[-1]))])
        self.action_index_list = [i for i in product( *action_index_list)]
        #random.shuffle (self.action_index_list )
        self.action_index_list.sort(key = lambda l : l.count(0),reverse = True)

        if parent_action is not None and len(self.prefer_action_list)>0:
            inv_action_index = []
            have_inv = True
            for index,state in enumerate(self.state_list):
                if not have_inv : break
                if state.enable and not state.reach:
                    inv_action = (-parent_action[index].ctrl_vel,parent_action[index].ctrl_phi)
                    if have_inv and inv_action in self.prefer_action_list[-1]:
                        inv_action_index.append(self.prefer_action_list[-1].index(inv_action))
                    else:
                        have_inv = False
            if have_inv:
                self.action_index_list.remove(tuple(inv_action_index))

    def pruning(self, agent_id_list, action_index_list = None):
        if action_index_list is None:
            action_index_list = self.last_action
        enable_agent_idx_list = [self.enable_index_list.index(i) for i in agent_id_list if i in self.enable_index_list ]
        new_action_index_list = []
        for action_index in self.action_index_list:
            check = False
            for idx in enable_agent_idx_list:
                if action_index[idx] != action_index_list[idx]:
                    check = True
                    break
            if check :
                new_action_index_list.append(action_index)
        self.action_index_list = new_action_index_list
        
        

    def next_action_index(self):
        if len(self.action_index_list) > 0:
            self.last_action = self.action_index_list.pop(0)
            return self.last_action
        else:
            return None
        
    def decode_action(self,action_index_list):
        action_list = [Action() for _ in self.state_list]
        action_index_list_ = [-1 for _ in self.state_list]
        for enable_idx,all_idx in enumerate(self.enable_index_list):
            test_action = self.prefer_action_list[enable_idx][action_index_list[enable_idx]]
            action_list[all_idx].ctrl_vel = test_action[0]
            action_list[all_idx].ctrl_phi = test_action[1]
            action_index_list_[all_idx] = action_index_list[enable_idx]
        return action_list,action_index_list_


class MultiFidelityEnv(object):
    def __init__(self,senario_dict,backend):
        self.backend = backend
        self.senario_dict = senario_dict
        self.time_limit = senario_dict['common']['time_limit']
        self.reward_coef = senario_dict['common']['reward_coef']
        self.reset_mode = senario_dict['common']['reset_mode']
        self.field_range = senario_dict['common']['field_range']
        self.ref_state_list = []
        self.ref_agent_list = []
        self.agent_num = 0
        self.step_number = 0
        for (_,grop) in self.senario_dict['agent_groups'].items():
            for agent_prop in grop:
                agent = AgentProp(agent_prop)
                state = AgentState()
                state.x = agent.init_x
                state.y = agent.init_y
                state.theta = agent.init_theta
                state.vel_b = agent.init_vel_b
                state.movable = agent.init_movable
                state.phi = agent.init_phi
                state.target_x = agent.init_target_x
                state.target_y = agent.init_target_y
                state.enable = True
                state.crash = False
                state.reach = False
                self.ref_state_list.append(state)
                self.ref_agent_list.append(agent)
                self.agent_num+=1

    def _random_reset(self,new_state, all_reset = False, retry_time = 40):
        #state_list  = copy.deepcopy(new_state)
        state_list  = new_state
        enable_list = [ all_reset|state.crash|state.reach for  state in state_list]
        enable_tmp = True in enable_list
        crash_idx_list = []
        for idx ,state in enumerate(state_list):
            if state.crash or all_reset: crash_idx_list.append(idx)
                
        if len(crash_idx_list)>0:
            for idx in crash_idx_list:
                state_list[idx].crash = False
                state_list[idx].movable = True
            for try_time in range(retry_time):
                for idx in crash_idx_list:
                    state_list[idx].x = np.random.uniform(self.field_range[0],self.field_range[1])
                    state_list[idx].y = np.random.uniform(self.field_range[2],self.field_range[3])
                    state_list[idx].theta = np.random.uniform(0,3.1415926*2)
                no_conflict = True
                for idx_a,idx_b in product(range(self.agent_num),range(self.agent_num)):
                    if idx_a == idx_b: continue
                    state_a = state_list[idx_a]
                    state_b = state_list[idx_b]
                    agent_dist = ((state_a.x-state_b.x)**2+(state_a.y-state_b.y)**2)**0.5
                    agent_size = self.ref_agent_list[idx_a].R_safe+self.ref_agent_list[idx_b].R_safe
                    no_conflict = agent_dist > agent_size
                    if not no_conflict : break
                if no_conflict: break
            #if not no_conflict: print('failed to place agent with no confiliction')

        reach_idx_list = []
        for idx ,state in enumerate(state_list):
            if state.reach or all_reset: reach_idx_list.append(idx)
        if len(reach_idx_list)>0:
            for idx in reach_idx_list:
                state_list[idx].reach = False
                state_list[idx].movable = True
            for try_time in range(retry_time):
                for idx in reach_idx_list:
                    state_list[idx].target_x = np.random.uniform(self.field_range[0],self.field_range[1])
                    state_list[idx].target_y = np.random.uniform(self.field_range[2],self.field_range[3])
                no_conflict = True
                for idx_a,idx_b in product(range(self.agent_num),range(self.agent_num)):
                    if idx_a == idx_b: continue
                    state_a = state_list[idx_a]
                    state_b = state_list[idx_b]
                    agent_dist = ((state_a.target_x-state_b.target_x)**2+(state_a.target_y-state_b.target_y)**2)**0.5
                    agent_size = self.ref_agent_list[idx_a].R_safe+self.ref_agent_list[idx_b].R_safe
                    no_conflict = agent_dist > agent_size
                    if not no_conflict : break
                if no_conflict: break
            #if not no_conflict: print('failed to place target with no confiliction')
        return enable_tmp,state_list,enable_list

    def _calc_reward(self,new_state,old_state,delta_time):
        crash = self.reward_coef['crash'] if new_state.crash else 0
        reach = self.reward_coef['reach'] if new_state.reach else 0
        new_dist = ((new_state.x-new_state.target_x)**2+(new_state.y-new_state.target_y)**2)**0.5
        old_dist = ((old_state.x-old_state.target_x)**2+(old_state.y-old_state.target_y)**2)**0.5

        potential = self.reward_coef['potential'] * (old_dist-new_dist)
        time_penalty = self.reward_coef['time_penalty']*delta_time
        reward = crash + reach + potential + time_penalty
        #print(re , crash , reach , potential, time_penalty)
        return reward
    
    def get_state(self):
        return self.backend.get_state()

    def set_state(self,state,enable_list = None,total_time = None):
        if enable_list is None:
            enable_list = [True] * len(state)
        self.backend.set_state(state,enable_list,reset = False , total_time = total_time)   


    def reset_rollout(self):
        if self.reset_mode == 'random':
            _,state_list,enable_list = self._random_reset(self.ref_state_list,True)
        else:
            state_list = self.ref_state_list
            enable_list = [True]*len(state_list)
        self.backend.set_state(state_list,enable_list,True)
        self.state_history = []
        self.obs_history = []
        self.time_history = []
        self.action_history = []

    def get_history(self):
        return self.state_history,self.obs_history,self.time_history,self.action_history
    
    def set_history(self,state_history,obs_history,time_history,action_history):
        self.state_history = state_history
        self.obs_history = obs_history
        self.time_history = time_history
        self.action_history = action_history

    def rollout_sync(self, policy_call_back, finish_call_back = None, pause_call_back = None,delay = 0):
        result_flag = None
        while True:
            total_time,new_state = self.backend.get_state()
            new_obs = self.backend.get_obs()
            action = policy_call_back(new_obs,new_state)
            self.state_history.append(copy.deepcopy(new_state))
            self.obs_history.append(new_obs)
            self.time_history.append(total_time)
            # check whether we should pause rollout
            if pause_call_back is not None:
                if pause_call_back(new_state):
                    result_flag = 'pause'
                    break

            # check whether we should stop one rollout
            finish = False
            if finish_call_back is not None:
                finish = finish_call_back(new_state)
            finish = finish or (total_time > self.time_limit)
            if finish:
                result_flag = 'finish'
                break
            if self.reset_mode == 'random':
                enable_tmp,state_list,enable_list = self._random_reset(new_state)
                if enable_tmp:
                    self.backend.set_state(state_list,enable_list)
            else :
                change = False 
                for idx,state in enumerate(new_state):
                    if state.reach:
                        change = True
                        new_state[idx].enable = False
                if change:
                    self.backend.set_state(new_state)
                

            self.backend.set_action(action)
            self.action_history.append(action)
            self.backend.step()
            self.step_number+=1
            if delay>0:
                time.sleep(delay)
        return result_flag

    def search_policy(self, multi_step = 2, back_number = 1):
        search_step = 0
        MAX_STEP = 1e4
        start_time ,begin_state = self.backend.get_state()
        for state in begin_state:
            if not state.enable: continue
            if state.reach or state.crash:
                #print('init state for search not clean')
                return None,None
        search_stack = [SearchNode(begin_state,start_time),]
        search_finish = False
        step = 0
        while not search_finish:
            step = step + 1
            state = self.backend.get_state()
            if step >= MAX_STEP:
                return None, None

            if len(search_stack) == 0:
                return None,None
            next_action_index = search_stack[-1].next_action_index()
            if next_action_index is None:
                search_stack.pop()
                continue
            next_action,next_action_index = search_stack[-1].decode_action(next_action_index)
            self.backend.set_action(next_action)
            for _ in range(multi_step):
                self.backend.step()
                search_step += 1
            self.step_number+=1
            new_time , new_state = self.backend.get_state()
            have_crash = False
            all_reach = True
            crash_index = []
            for idx,state in enumerate(new_state):
                if state.enable:
                    if state.crash:
                        crash_index.append(idx)
                    all_reach = all_reach and state.reach
                    have_crash = have_crash or state.crash
            
            if have_crash:
                for _ in range(len(search_stack) - max(len(search_stack)-back_number,0) - 1):
                    search_stack.pop()

                search_stack[-1].pruning(crash_index)#,next_action_index)
                self.backend.set_state(search_stack[-1].state_list,total_time = search_stack[-1].time)
                continue
            else:
                search_stack.append(SearchNode(new_state,new_time,next_action,next_action_index))
                if all_reach:
                    search_finish = True
        action_list = []
        action_index_list = []
        for node in search_stack[1:]:
            for _ in  range(multi_step):
                action_list.append(node.parent_action)
                action_index_list.append(node.parent_action_index)
        return action_list,action_index_list,search_step

    def get_trajectoy(self):
        trajectoy = []
        for agent_idx in range(self.agent_num):
            trajectoy_agent = []
            for idx in range(len(self.action_history)):
                if self.state_history[idx][agent_idx].movable :
                    done = not self.state_history[idx+1][agent_idx].movable
                    time = self.time_history[idx]
                    obs = self.obs_history[idx][agent_idx]
                    obs_next = self.obs_history[idx+1][agent_idx]
                    action = self.action_history[idx][agent_idx]
                    reward = self._calc_reward(self.state_history[idx+1][agent_idx],self.state_history[idx][agent_idx],self.time_history[idx+1]-self.time_history[idx])
                    trajectoy_agent.append({'obs':obs,'action':action,'reward': reward, 'obs_next':obs_next, 'done':done, 'time':time})
            trajectoy.append(trajectoy_agent)
        return trajectoy

    def get_result(self):
        vel_list = []
        result = {}
        crash_time = 0
        reach_time = 0
        total_reward = 0
        for list_idx in range(len(self.state_history)): 
            state_list = self.state_history[list_idx]
            for state_idx in range(len(state_list)):
                state = state_list[state_idx]
                if state.movable and (list_idx+1)<len(self.state_history):
                    total_reward += self._calc_reward(  self.state_history[list_idx+1][state_idx],
                                                        self.state_history[list_idx][state_idx],
                                                        self.time_history[list_idx+1]-self.time_history[list_idx])
                if state.movable:
                    vel_list.append(abs(state.vel_b))
                if list_idx>0:
                    crash_time += 1 if state.crash and not self.state_history[list_idx-1][state_idx].crash else 0
                    reach_time += 1 if state.reach and not self.state_history[list_idx-1][state_idx].reach else 0
        result['total_reward'] = total_reward
        result['crash_time'] = crash_time
        result['reach_time'] = reach_time
        result['mean_vel'] = sum(vel_list)/len(vel_list)
        result['total_time'] = self.time_history[-1]-self.time_history[0]
        return result

    def close(self):
        self.backend.close()