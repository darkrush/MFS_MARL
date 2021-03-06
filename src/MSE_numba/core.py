import numpy as np
import math
import numba

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

# state of agents (including internal/mental state)
class AgentState(object):
    def __init__(self):
        #center point position in x,y axis
        self.x = 0
        self.y = 0
        #linear velocity of back point
        self.vel_b = 0
        # direction of car axis
        self.theta = 0
        # Deflection angle of front wheel
        self.phi = 0
        # Movable
        self.enable = True
        self.movable = True
        self.crash = False
        self.reach = False
        # target x coordinate
        self.target_x   = 1
        # target y coordinate
        self.target_y   = 1

    def __str__(self):
        return ' pos : ['+str(self.x)+', '+str(self.y)+', '+str(self.theta)+']'
    
    def __repr__(self):
        return self.__str__()

# action of the agent
class Action(object):
    def __init__(self):
        self.ctrl_vel = 0 # ctrl_vel ∈ [-1,1]
        self.ctrl_phi = 0 # ctrl_phi ∈ [-1,1]

class Observation(object):
    def __init(self):
        self.pos = [0,0,0,0,0] # x,y,theta,target_x,target_y
        self.laser_data = []   # float*n
    
    def __str__(self):
        return ' pos : '+str(self.pos)+' laser : '+str(self.laser_data)
    
    def __repr__(self):
        return self.__str__()

# properties of agent entities
class Agent(object):
    def __init__(self,agent_prop = None):
        self.R_safe     = 0.2  # minimal distance not crash
        self.R_reach    = 0.1  # maximal distance for reach target
        self.L_car      = 0.3  # length of the car
        self.W_car      = 0.2  # width of the car
        self.L_axis     = 0.25 # distance between front and back wheel
        self.R_laser    = 4    # range of laser
        self.N_laser    = 360  # number of laser lines
        self.true_laser = 360
        self.K_vel      = 1    # coefficient of back whell velocity control
        self.K_phi      = 30   # coefficient of front wheel deflection control
        self.init_x     = -1   # init x coordinate
        self.init_y     = -1   # init y coordinate
        self.init_theta = 0    # init theta
        self.init_vel_b = 0    # init velocity of back point
        self.init_phi   = 0    # init front wheel deflection
        self.init_movable = True # init movable state
        self.init_target_x = 1 # init target x coordinate
        self.init_target_y = 1 # init target y coordinate

        if agent_prop is not None:
            for k,v in agent_prop.items():
                self.__dict__[k] = v
        self.N_laser = int(self.N_laser)

        if "true_laser" not in agent_prop.keys():
            self.true_laser = self.N_laser
        
        assert self.N_laser%self.true_laser==0
        self.state = AgentState()
        self.action = Action()
        self.color = [0,0,0]
        #temp laser state
        self.laser_state = np.array([self.R_laser]*self.N_laser)

        self.reset()

    def reset(self,state = None):
        self.total_time = 0
        if state is not None:
            self.state = state
        else:
            self.state.x = self.init_x
            self.state.y = self.init_y
            self.state.theta = self.init_theta
            self.state.vel_b = self.init_vel_b
            self.state.phi   = self.init_phi
            self.state.movable     = self.init_movable
            self.state.crash = False
            self.state.reach = False
        self.laser_state = np.array([self.R_laser]*self.N_laser)
        
    def check_AA_collisions(self,agent_b):
        min_dist = (self.R_safe + agent_b.R_safe)**2
        ab_dist = (self.state.x - agent_b.state.x)**2 + (self.state.y - agent_b.state.y)**2
        return ab_dist<=min_dist

    def check_reach(self):
        max_dist = self.R_reach**2
        at_dist = (self.state.x - self.state.target_x)**2 + (self.state.y - self.state.target_y)**2
        return at_dist<=max_dist

# multi-agent world
class World(object):
    def __init__(self,agent_groups,dt):
        
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        for (_,agent_group) in agent_groups.items():
            for agent_prop in agent_group:
                agent = Agent(agent_prop)
                self.agents.append(agent)
        for idx,agent in enumerate(self.agents):
            agent.color = hsv2rgb(360.0/len(self.agents)*idx,1.0,1.0)
        # simulation timestep
        self.dt = dt
        self.cam_range = 4
        self.viewer = None
        self.total_time = 0
        self.laser_dirty = True
        self._reset_render()
    def reset(self):
        self.total_time = 0
        for agent in self.agents:
            for k in agent.state.__dict__.keys():
                if k == 'reach' or k == 'crash':
                    continue
                agent.state.__dict__[k] = agent.__dict__['init_'+k]
            agent.state.crash = False
            agent.state.reach = False
        self.laser_dirty = True
        self._reset_render()
        return True

    def set_action(self,enable_list,actions):
        for enable,agent,action in zip(enable_list,self.agents,actions):
            if not enable: continue
            agent.action.ctrl_vel = action.ctrl_vel
            agent.action.ctrl_phi = action.ctrl_phi

    def get_state(self):
        return [agent.state for agent in self.agents]
    
    def set_state(self,enable_list,states):
        for enable,agent,state in zip(enable_list,self.agents,states):
            if not enable :continue
            for k in agent.state.__dict__.keys():
                agent.state.__dict__[k] = state.__dict__[k]
        self.laser_dirty = True

    def get_obs(self):
        obs_data = {'time':self.total_time,'obs_data':[]}
        self.update_laser_state()
        for idx_a in range(len(self.agents)):
            state = self.agents[idx_a].state
            pos = np.array([state.x, state.y, state.theta, state.target_x, state.target_y])
            laser_data = self.agents[idx_a].laser_state
            obs = Observation()
            obs.pos = pos
            obs.laser_data = laser_data
            obs_data['obs_data'].append(obs)
        return obs_data['obs_data']

    # update state of the world
    def step(self,step_num):
        for _ in range(step_num):
            self.apply_action()
            self.integrate_state()
            self.check_collisions()
            self.check_reach()
            self.total_time += self.dt
        self.laser_dirty = True
    
    # gather agent action forces
    def apply_action(self):
        # set applied forces
        for agent in self.agents:
            agent.state.vel_b = np.clip(agent.action.ctrl_vel, -1.0, 1.0)*agent.K_vel if agent.state.movable else 0
            agent.state.phi   = np.clip(agent.action.ctrl_phi, -1.0, 1.0)*agent.K_phi if agent.state.movable else 0

    def update_laser_state(self):
        if not self.laser_dirty:
            return
        for idx_a,agent_a in enumerate(self.agents):
            R = agent_a.R_laser
            N = agent_a.N_laser
            true_N = agent_a.true_laser
            agent_a.laser_state = np.array([R]*N)
            for idx_b,agent_b in enumerate(self.agents):
                if idx_a == idx_b:
                    continue
                l_laser = laser_agent_agent_wrap(R,N,true_N,agent_a,agent_b)
                agent_a.laser_state = np.min(np.vstack([agent_a.laser_state,l_laser]),axis = 0)
        self.laser_dirty = False


    # integrate physical state
    def integrate_state(self):
        for agent in self.agents:
            if not agent.state.movable: continue
            agent.state.x,agent.state.y,agent.state.theta = integrate_state_wrap(agent,self.dt)
        
    def check_collisions(self):
        for ia, agent_a in enumerate(self.agents):
            if agent_a.state.crash :
                continue
            for ib, agent_b in enumerate(self.agents):
                if ia==ib :
                    continue
                if agent_a.check_AA_collisions(agent_b) :
                    agent_a.state.crash = True
                    agent_a.state.movable = False
                    break
    
    def check_reach(self):
        for agent in self.agents:
            reach = agent.check_reach()
            if reach :
                agent.state.reach = True
                agent.state.movable = False
    
    def _reset_render(self):
        self.agent_geom_list = None
    
    def set_total_time(self, time):
        self.total_time = time

    def get_total_time(self):
        return self.total_time

def integrate_state_wrap(agent,dt):
    return integrate_state_njit(agent.state.phi,agent.state.vel_b,agent.state.theta,agent.L_axis,agent.state.x,agent.state.y,dt)

@numba.njit()
def integrate_state_njit(_phi,_vb,_theta,_L,_x,_y,dt):
    sth = np.sin(_theta)
    cth = np.cos(_theta)
    _xb = _x - cth*_L/2.0
    _yb = _y - sth*_L/2.0
    tphi = np.tan(_phi)
    _omega = _vb/_L*tphi
    _delta_theta = _omega * dt
    if abs(_phi)>0.00001:
        _rb = _L/tphi
        _delta_tao = _rb*(1-np.cos(_delta_theta))
        _delta_yeta = _rb*np.sin(_delta_theta)
    else:
        _delta_tao = _vb*dt*(_delta_theta/2.0)
        _delta_yeta = _vb*dt*(1-_delta_theta**2/6.0)
    _xb += _delta_yeta*cth - _delta_tao*sth
    _yb += _delta_yeta*sth + _delta_tao*cth
    _theta += _delta_theta
    _theta = (_theta/3.1415926)%2*3.1415926

    nx = _xb + np.cos(_theta)*_L/2.0
    ny = _yb + np.sin(_theta)*_L/2.0
    ntheta = _theta
    return nx,ny,ntheta

def laser_agent_agent_wrap(R_laser,N_laser,true_N,a,b):
    laser_result = laser_agent_agent_njit(R_laser,N_laser,true_N,a.state.x,a.state.y,a.state.theta,b.state.x,b.state.y,b.state.theta,b.L_car,b.W_car)
    return laser_result

@numba.njit()
def laser_agent_agent_njit(R_laser,N_laser,true_N,a_x,a_y,a_t,b_x,b_y,b_t,b_L,b_W):

    #l_laser = np.array([R_laser]*N_laser)
    true_l_laser = np.ones((true_N,),dtype=np.float32)*R_laser
    o_pos =  np.array([a_x,a_y])
    oi_pos = np.array([b_x,b_y])
    if np.linalg.norm(o_pos-oi_pos)>R_laser+(b_L**2 + b_W**2)**0.5 / 2.0:
        return np.ones((N_laser,),dtype=np.float32)*R_laser
    theta = a_t
    theta_b = b_t
    cthb= math.cos(theta_b)
    sthb= math.sin(theta_b)
    half_l_shift = np.array([cthb,sthb])*b_L/2.0
    half_w_shift = np.array([-sthb,cthb])*b_W/2.0
    car_points = []
    car_points.append(oi_pos+half_l_shift+half_w_shift-o_pos)
    car_points.append(oi_pos-half_l_shift+half_w_shift-o_pos)
    car_points.append(oi_pos-half_l_shift-half_w_shift-o_pos)
    car_points.append(oi_pos+half_l_shift-half_w_shift-o_pos)
    car_line = [[car_points[i],car_points[(i+1)%len(car_points)]] for i in range(len(car_points))]
    for start_point, end_point in  car_line:
        v_es = start_point-end_point
        tao_es = np.array((v_es[1],-v_es[0]))
        tao_es = tao_es/np.linalg.norm(tao_es)
        if abs(np.dot(start_point,tao_es))>R_laser:
            continue
        
        if start_point[0]*end_point[1]< start_point[1]*end_point[0] :
            start_point,end_point = end_point,start_point
        theta_start = np.arccos(start_point[0]/np.linalg.norm(start_point))
        if start_point[1]<0:
            theta_start = math.pi*2-theta_start
        theta_start-=theta
        theta_end = np.arccos(end_point[0]/np.linalg.norm(end_point))
        if end_point[1]<0:
            theta_end = math.pi*2-theta_end
        theta_end-=theta
        laser_idx_start = theta_start/(2*math.pi/true_N)
        laser_idx_end   =   theta_end/(2*math.pi/true_N)
        if laser_idx_start> laser_idx_end:
            laser_idx_end+=true_N
        if math.floor(laser_idx_end)-math.floor(laser_idx_start)==0:
            continue
        laser_idx_start = math.ceil(laser_idx_start)
        laser_idx_end = math.floor(laser_idx_end)
        for laser_idx in range(laser_idx_start,laser_idx_end+1):
            laser_idx%=true_N
            x1 = start_point[0]
            y1 = start_point[1]
            x2 = end_point[0]
            y2 = end_point[1]
            theta_i = theta+laser_idx*math.pi*2/true_N
            cthi = math.cos(theta_i)
            sthi = math.sin(theta_i)
            temp = (y1-y2)*cthi - (x1-x2)*sthi
            # temp equal zero when collinear
            if abs(temp) <= 1e-10:
                dist = R_laser 
            else:
                dist = (x2*y1-x1*y2)/(temp)
            if dist > 0 and dist < true_l_laser[laser_idx]:
                true_l_laser[laser_idx] = dist
    
    linear_sacle = N_laser//true_N
    if linear_sacle == 1:
        return true_l_laser
    else:
        l_laser = np.ones((N_laser,),dtype=np.float32)*R_laser
        for i in range(true_N):
            for j in range(linear_sacle):
                l_data = true_l_laser[i]
                r_data = true_l_laser[(i+1)%true_N]
                alpha = j/linear_sacle
                tmp = l_data*(1-alpha) + r_data*alpha
                l_laser[i*linear_sacle+j] = tmp
        return l_laser