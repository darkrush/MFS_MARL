from . import core
import math
import numpy as np


class Observation(object):
    def __init__(self):
        self.pos = [0,0,0,0,0] # x,y,theta,target_x,target_y
        self.laser_data = []   # float*n

    def __str__(self):
        return ' pos : '+str(self.pos)+' laser : '+str(self.laser_data)
    def __repr__(self):
        return self.__str__()

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    #r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return (r, g, b)


class MSE_backend(object):
    def __init__(self, scenario, step_t = 0.1, sim_t = 0.01, window_scale = 1.0):
        self.window_scale = window_scale
        self.agent_groups = scenario['agent_groups']
        self.use_gui = scenario['common']['use_gui']
        self.step_t = step_t
        self.sim_t = sim_t
        self.step_number = int(self.step_t/self.sim_t)
        if abs(step_t - self.step_number*sim_t)>1e-6:
            print('Warning : Step_t is not an integral multiple of Sim_t!')
        self.cam_range = 4
        self.viewer = None
        self.agent_number = 0
        for (_,agent_group) in self.agent_groups.items():
            for agent_prop in agent_group:
                self.agent_number += 1
        self.world = core.World(self.agent_groups,sim_t)
        self._reset_render()
    
    def _reset_render(self):
        self.agent_geom_list = None
    
    def render(self,time = '0', mode='human'):
        if self.viewer is None:
            from . import rendering 
            self.viewer = rendering.Viewer(800,800)
        self.agents = []
        self.color_list = []
        self.world.update_laser_state()
        for idx in range(self.agent_number):
            self.color_list.append(hsv2rgb(360.0/self.agent_number*idx,1.0,1.0))
            #self.agents.append(self.world.get_agent(idx))
        # create rendering geometry
        self.agent_geom_list = None
        if self.agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from . import rendering
            self.viewer.set_bounds(0-self.cam_range, 0+self.cam_range, 0-self.cam_range, 0+self.cam_range)
            self.agent_geom_list = []
            for idx,agent in enumerate(self.world.agents):
                enable =  agent.state.enable
                agent_geom = {}
                total_xform = rendering.Transform()
                agent_geom['total_xform'] = total_xform
                agent_geom['laser_line'] = []

                geom = rendering.make_circle(agent.R_reach)
                geom.set_color(*self.color_list[idx],alpha  = 1.0 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                agent_geom['target_circle']=(geom,xform)

                N = agent.N_laser
                for idx_laser in range(N):
                    theta_i = idx_laser*math.pi*2/N
                    #d = agent.R_laser
                    d = 1
                    end = (math.cos(theta_i)*d, math.sin(theta_i)*d)
                    geom = rendering.make_line((0, 0),end)
                    geom.set_color(0.0,1.0,0.0,alpha = 0.5 if enable else 0.0)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.add_attr(total_xform)
                    agent_geom['laser_line'].append((geom,xform))
                
                half_l = agent.L_car/2.0
                half_w = agent.W_car/2.0
                geom = rendering.make_polygon([[half_l,half_w],[-half_l,half_w],[-half_l,-half_w],[half_l,-half_w]])
                geom.set_color(*self.color_list[idx],alpha = 0.4 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['car']=(geom,xform)

                geom = rendering.make_line((0,0),(half_l,0))
                geom.set_color(1.0,0.0,0.0,alpha = 1 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['front_line']=(geom,xform)
                
                geom = rendering.make_line((0,0),(-half_l,0))
                geom.set_color(0.0,0.0,0.0,alpha = 1 if enable else 0.0)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['back_line']=(geom,xform)

                self.agent_geom_list.append(agent_geom)

            self.viewer.geoms = []
            for agent_geom in self.agent_geom_list:
                self.viewer.add_geom(agent_geom['target_circle'][0])
                for geom in agent_geom['laser_line']:
                    self.viewer.add_geom(geom[0])
                self.viewer.add_geom(agent_geom['car'][0])
                self.viewer.add_geom(agent_geom['front_line'][0])
                self.viewer.add_geom(agent_geom['back_line'][0])
        
        for agent,agent_geom in zip(self.world.agents,self.agent_geom_list):
            for idx,laser_line in enumerate(agent_geom['laser_line']):
                laser_line[1].set_scale(agent.laser_state[idx],agent.laser_state[idx]) 
            agent_geom['front_line'][1].set_rotation(agent.state.phi)
            agent_geom['target_circle'][1].set_translation(agent.state.target_x*self.window_scale,agent.state.target_y*self.window_scale)
            agent_geom['target_circle'][1].set_scale(self.window_scale,self.window_scale)
            agent_geom['total_xform'].set_scale(self.window_scale,self.window_scale)
            agent_geom['total_xform'].set_rotation(agent.state.theta)
            agent_geom['total_xform'].set_translation(agent.state.x*self.window_scale,agent.state.y*self.window_scale)
            
        return self.viewer.render(time,return_rgb_array = mode=='rgb_array')

    def step(self):
        render_frams = self.step_number if self.step_number <= 40 else int(self.step_number / 10)
        total_frams = self.step_number
        while total_frams>=0:
            self.world.step(render_frams if render_frams < total_frams else total_frams)
            if self.use_gui :
                self.render(time = '%.2f'%self.world.get_total_time())
            total_frams-=render_frams

    def get_state(self):
        return self.world.total_time,self.world.get_state()
    
    def set_state(self,state_list,enable_list = None,reset = False,total_time = None):
        if enable_list is None:
            enable_list = [True]* len(state_list)
        self.world.set_state(enable_list,state_list)
        if reset:
            self.world.set_total_time(0)
            self._reset_render()
        if total_time is not None:
            self.world.set_total_time(total_time)

    def get_obs(self):
        obs = []
        self.world.update_laser_state()
        origin_obs = self.world.get_obs()
        #for ob in origin_obs:
        #    print(ob)
        # x,y,theta,target_x,target_y
        for obs_idx in range(0,self.agent_number):
            obs_old = origin_obs[obs_idx]
            obs_new = Observation()
            theta = obs_old.pos[2]
            xt = obs_old.pos[3] - obs_old.pos[0]
            yt = obs_old.pos[4] - obs_old.pos[1]
            xt,yt = (xt*np.cos(theta)+yt*np.sin(theta),yt*np.cos(theta)-xt*np.sin(theta))
            #obs_new.pos = [xt,yt]
            #obs_new.pos = [obs_c.pos_x,obs_c.pos_y,obs_c.pos_theta,obs_c.pos_target_x,obs_c.pos_target_y]
            obs_new.pos = [(xt**2+yt**2)**0.5,math.atan2(yt,xt)]
            for i in range(len(obs_old.laser_data)):
                obs_new.laser_data.append(obs_old.laser_data[i])
            obs.append(obs_new)
        return obs

    def set_action(self,actions,enable_list= None):
        if enable_list is None:
            enable_list = [True]* len(actions)
        self.world.set_action(enable_list, actions)

    def close(self):
        pass