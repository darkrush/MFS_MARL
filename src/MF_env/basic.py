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
        self.enable = True
        # Movable
        self.movable = True
        self.crash = False
        self.reach = False
        # target x coordinate
        self.target_x   = 1
        # target y coordinate
        self.target_y   = 1


# action of the agent
class Action(object):
    def __init__(self):
        self.ctrl_vel = 0 # ctrl_vel \belong [-1,1]
        self.ctrl_phi = 0 # ctrl_phi \belong [-1,1]
    
    def __str__(self):
        return 'ctrl_vel : '+str(self.ctrl_vel)+' ctrl_phi : '+str(self.ctrl_phi)
    def __repr__(self):
        return self.__str__()

class Observation(object):
    def __init__(self):
        self.pos = [0,0,0,0,0] # x,y,theta,target_x,target_y
        self.laser_data = []   # float*n

    def __str__(self):
        return ' pos : '+str(self.pos)+' laser : '+str(self.laser_data)
    def __repr__(self):
        return self.__str__()

# properties of agent entities
class AgentProp(object):
    def __init__(self,agent_prop = None):
        self.R_safe     = 0.2  # minimal distance not crash
        self.R_reach    = 0.1  # maximal distance for reach target
        self.L_car      = 0.3  # length of the car
        self.W_car      = 0.2  # width of the car
        self.L_axis     = 0.25 # distance between front and back wheel
        self.R_laser    = 4    # range of laser
        self.N_laser    = 360  # number of laser lines
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