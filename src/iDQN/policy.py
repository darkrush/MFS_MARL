import torch
import math
import numpy as np
import copy
import ctypes
import random

ACTION_TABLE = np.array(((-1,-1),(-1,0),(-1,1),(0,0),(1,-1),(1,0),(1,1)))


def dis2conti(dis_action):
    action_list = []
    for idx in range(dis_action.shape[0]):
        conti_action = ACTION_TABLE[dis_action[idx],:]
        action_list.append(conti_action)
    return np.array(action_list)

# action of the agent
class Action(object):
    def __init__(self):
        self.ctrl_vel = 0 # ctrl_vel \belong [-1,1]
        self.ctrl_phi = 0 # ctrl_phi \belong [-1,1]
    
    def __str__(self):
        return 'ctrl_vel : '+str(self.ctrl_vel)+' ctrl_phi : '+str(self.ctrl_phi)
    def __repr__(self):
        return self.__str__()

def naive_inference(r,theta,dist=0.1,min_r=0.654):
    yt = math.sin(theta)*r
    xt = math.cos(theta)*r
    if abs(math.tan(theta)*r) < dist * 0.5:
        vel = np.sign(xt)
        phi = 0
    else:
        in_min_r = (xt**2+(abs(yt)-min_r)**2)< min_r**2
        vel = -1 if (bool(in_min_r) ^ bool(xt<0)) else 1
        phi = -1 if (bool(in_min_r) ^ bool(yt<0)) else 1
    return vel,phi

class NN_policy(object):
    def __init__(self,Qnetwork,epsilon,min_dis = 0):
        self.Qnetwork = copy.deepcopy(Qnetwork)   
        self.epsilon = epsilon 
        self.min_dis = min_dis
        self.cuda = next(self.Qnetwork.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs.pos for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs.laser_data for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            qvalue = self.Qnetwork(pos,laser_data).cpu().numpy()
            action_index = np.argmax(qvalue,1)
        if random.random() <self.epsilon:
            action_index = np.random.randint(ACTION_TABLE.shape[0],size=(action_index.shape[0]))

        action = dis2conti(action_index)
        action_list = []
        for idx in range(action.shape[0]):
            a = Action()
            xt = obs_list[idx].pos[0]
            yt = obs_list[idx].pos[1]
            if xt**2+yt**2 < self.min_dis**2:
                a.ctrl_vel,a.ctrl_phi = naive_inference(xt,yt)
            else:
                a.ctrl_vel = float(action[idx,0])
                a.ctrl_phi = float(action[idx,1])
            action_list.append(a)
        return action_list

class Mix_policy(object):
    def __init__(self,Qnetwork,epsilon,search_policy,replace_table,min_dis = 0):
        self.Qnetwork = copy.deepcopy(Qnetwork)
        self.epsilon = epsilon 
        self.min_dis = min_dis
        self.search_policy = search_policy
        self.replace_table = replace_table
        self.step = 0
        # Check Qnetwork is on cuda or not
        self.cuda = next(self.Qnetwork.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs.pos for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs.laser_data for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            qvalue = self.Qnetwork(pos,laser_data).cpu().numpy()
            action_index = np.argmax(qvalue,1)
        if random.random() <self.epsilon:
            action_index = np.random.randint(ACTION_TABLE.shape[0],size=(action_index.shape[0]))

        action = dis2conti(action_index)
        action_list = []
        for idx in range(action.shape[0]):
            a = Action()
            xt = obs_list[idx].pos[0]
            yt = obs_list[idx].pos[1]
            if xt**2+yt**2 < self.min_dis**2:
                a.ctrl_vel,a.ctrl_phi = naive_inference(xt,yt)
            else:
                a.ctrl_vel = float(action[idx,0])
                a.ctrl_phi = float(action[idx,1])
            action_list.append(a)
        if self.search_policy is not None:
            if self.step<len(self.search_policy):
                for agent_idx in range(len(action_list)):
                    if self.replace_table[self.step][agent_idx] >=0:
                        action_list[agent_idx].ctrl_vel = self.search_policy[self.step][agent_idx].ctrl_vel
                        action_list[agent_idx].ctrl_phi = self.search_policy[self.step][agent_idx].ctrl_phi
            self.step+=1
        return action_list     


class Agent_Mix_policy(object):
    def __init__(self,Qnetwork,epsilon,expert_Qnetwork,replace_list,min_dis = 0):
        self.Qnetwork = copy.deepcopy(Qnetwork)   
        self.epsilon = epsilon 
        self.min_dis = min_dis
        self.expert_Qnetwork = copy.deepcopy(expert_Qnetwork)
        self.replace_list = replace_list
        self.step = 0
        # Check Qnetwork is on cuda or not
        self.cuda = next(self.Qnetwork.parameters()).is_cuda
    def inference(self,obs_list,state_list):
        with torch.no_grad():
            pos = torch.Tensor(np.vstack([obs.pos for obs in obs_list]))
            laser_data = torch.Tensor(np.vstack([obs.laser_data for obs in obs_list]))
            if self.cuda:
                pos = pos.cuda()
                laser_data = laser_data.cuda()
            qvalue = self.Qnetwork(pos,laser_data).cpu().numpy()
            action_index = np.argmax(qvalue,1)
        if random.random() <self.epsilon:
            action_index = np.random.randint(ACTION_TABLE.shape[0],size=(action_index.shape[0]))

        action = dis2conti(action_index)
        action_list = []
        for idx in range(action.shape[0]):
            a = Action()
            xt = obs_list[idx].pos[0]
            yt = obs_list[idx].pos[1]
            if xt**2+yt**2 < self.min_dis**2:
                a.ctrl_vel,a.ctrl_phi = naive_inference(xt,yt)
            else:
                a.ctrl_vel = float(action[idx,0])
                a.ctrl_phi = float(action[idx,1])
            action_list.append(a)
        if self.replace_list is not None:
            with torch.no_grad():
                qvalue_replace = self.Qnetwork(pos,laser_data).cpu().numpy()
                action_replace = dis2conti(np.argmax(qvalue_replace,1))
            action_replace = np.clip(action_replace, -1., 1.)
            for agent_idx in self.replace_list:
                if state_list[agent_idx].crash or state_list[agent_idx].reach:
                    self.replace_list.remove(agent_idx)
            for agent_idx in self.replace_list:
                action_list[agent_idx].ctrl_vel = float(action_replace[agent_idx,0])
                action_list[agent_idx].ctrl_phi = float(action_replace[agent_idx,1])
        return action_list     




class naive_policy(object):
    def __init__(self,max_phi,l,dist):
        self.max_phi = max_phi
        self.l = l
        self.dist = dist
        self.min_r = self.l/np.tan(self.max_phi)
        self.right_o = np.array([self.min_r,0.0])
        self.left_o = np.array([-self.min_r,0.0])
    
    def inference(self,obs_list):
        obs_list = obs_list[0]
        #if isinstance(obs_list,torch.Tensor):
        #    obs_list = obs_list.tolist()
        action_list = []
        for obs in obs_list:
            #theta = obs[2]
            #xt = obs[3] - obs[0]
            #yt = obs[4] - obs[1]
            #xt,yt = (xt*np.cos(theta)+yt*np.sin(theta),yt*np.cos(theta)-xt*np.sin(theta))
            vel,phi = naive_inference(obs[0],obs[1])
            action_list.append([vel,phi])
        return action_list


class RVO_policy(object):
    def __init__(self):
        self.max_phi = 17.10/180*3.1415926
        self.l = 0.2
        self.dist = 0.1
        self.min_r = self.l/np.tan(self.max_phi)
        self.right_o = np.array([self.min_r,0.0])
        self.left_o = np.array([-self.min_r,0.0])
        self.function = ctypes.CDLL('./libRVO.so').AgentNewVel
    def inference(self,obs_list,state_list):
        MAX_SPEED = 0.18266
        agent_num = len(obs_list)
        x_position = []
        y_position = []
        velocity_x = []
        velocity_y = []
        prefer_vel_x = []
        prefer_vel_y = []
        #action_list = []

        for idx in range(agent_num):
            state = state_list[idx]
            
            xt = state.target_x-state.x
            yt = state.target_y-state.y

            t_theta = math.atan2(yt,xt)
            norm = (xt**2+yt**2)**0.5
            prefer_vel_x.append(xt/norm*MAX_SPEED)
            prefer_vel_y.append(yt/norm*MAX_SPEED)

            x_position.append(state.x)
            y_position.append(state.y)
            
            vx = math.cos(state.theta)*state.vel_b #if math.cos(t_theta-state.theta)>0 else -math.cos(state.theta)*MAX_SPEED 
            vy = math.sin(state.theta)*state.vel_b #if math.cos(t_theta-state.theta)>0 else -math.sin(state.theta)*MAX_SPEED 
            velocity_x.append(vx)
            velocity_y.append(vy)

            
            #xt = obs[3] - obs[0]
            #yt = obs[4] - obs[1]
            #tnorm = (xt**2+yt**2)**0.5
            #if tnorm<1:
            #    xt/=tnorm
            #    yt/=tnorm
            #prefer_vel_x.append(xt)
            #prefer_vel_y.append(yt)
        cpp_x_position = (ctypes.c_float * agent_num)(*x_position)
        cpp_y_position = (ctypes.c_float * agent_num)(*y_position)
        cpp_velocity_x = (ctypes.c_float * agent_num)(*velocity_x)
        cpp_velocity_y = (ctypes.c_float * agent_num)(*velocity_y)
        cpp_prefer_vel_x = (ctypes.c_float * agent_num)(*prefer_vel_x)
        cpp_prefer_vel_y = (ctypes.c_float * agent_num)(*prefer_vel_y)
        time_Horizon = ctypes.c_float(2)
        neighbor_Dist = ctypes.c_float(10.0)
        timestep = ctypes.c_float(1)
        maxspeed = [MAX_SPEED,]*agent_num
        radius = [0.24,]*agent_num
        cpp_radius = (ctypes.c_float * agent_num)(*radius)
        cpp_maxspeed = (ctypes.c_float * agent_num)(*maxspeed) 
        self.function( agent_num, cpp_x_position, cpp_y_position, cpp_velocity_x, cpp_velocity_y, cpp_prefer_vel_x, cpp_prefer_vel_y, cpp_radius, cpp_maxspeed, time_Horizon, neighbor_Dist, timestep)
        temp_velocity_x = np.array(cpp_velocity_x)
        temp_velocity_y = np.array(cpp_velocity_y)
        velocity_x = temp_velocity_x.tolist()
        velocity_y = temp_velocity_y.tolist()


        action_list = []
        for idx,state in enumerate(state_list):
            vel_norm = (velocity_x[idx]**2 + velocity_y[idx]**2)**0.5
            vel_theta = math.atan2(velocity_y[idx],velocity_x[idx])
            #print(velocity_x[idx],velocity_y[idx])
            vel = vel_norm/MAX_SPEED if math.cos(vel_theta-state.theta)>0 else -vel_norm/MAX_SPEED
            phi = 1 if (vel_theta-state.theta+math.pi)%(math.pi*2)-math.pi >0 else -1
            #vel_norm = (velocity_x[idx]**2 + velocity_y[idx]**2)**0.5
            #theta = state.theta
            #xt = velocity_x[idx]*1.5
            #yt = velocity_y[idx]*1.5
            #xt,yt = (xt*np.cos(theta)+yt*np.sin(theta),yt*np.cos(theta)-xt*np.sin(theta))
            #if abs(yt) < self.dist:
            #    vel = np.sign(xt)
            #    phi = 0
            #else:
            #    in_min_r = (xt**2+(abs(yt)-self.min_r)**2)< self.min_r**2
            #    vel = -1 if np.bitwise_xor(in_min_r,xt<0) else 1
            #    phi = -1 if np.bitwise_xor(in_min_r,yt<0) else 1
            xt = state.target_x-state.x
            yt = state.target_y-state.y
            a = Action()
            a.ctrl_vel = vel#*vel_norm
            a.ctrl_phi = phi
            if xt**2+yt**2 < 0.0**2:
                r = (xt**2+yt**2)**0.5
                theta = math.atan2(yt,xt)-state.theta
                a.ctrl_vel,a.ctrl_phi = naive_inference(r,theta)
            action_list.append(a)
        return action_list

class trace_policy(object):
    def __init__(self,trace,repeta):
        self.trace = trace
        self.step  = 0
        self.repeta = repeta
        self.repeta_state = 0
    def inference(self, obs,state):
        if self.step>=len(self.trace):
            action_list = self.trace[-1]
            for idx in range(len(action_list)):
                action_list[idx].ctrl_vel = 0
                action_list[idx].ctrl_phi = 0
        else:
            action_list = self.trace[self.step]
        self.repeta_state+=1
        if self.repeta_state== self.repeta:
            self.step += 1
            self.repeta_state = 0
        return action_list