from MSE import MSE_backend
from scenario.paser import  parse_senario 
from MF_env import basic
import time
scenario = parse_senario('./scenario/test_backend.yaml')

back_end = MSE_backend.MSE_backend(scenario, step_t = 0.1, sim_t = 0.01, window_scale = 1.0)
state_list = []
enable = True
for _,group in scenario['agent_groups'].items():
    for agent in group:
        state = basic.AgentState()
        state.x = agent['init_x']
        state.y = agent['init_y']
        state.theta = agent['init_theta']
        state.target_x = agent['init_target_x']
        state.target_y = agent['init_target_y']
        state.vel_b = agent['init_vel_b']
        state.phi = agent['init_phi']
        state.movable = agent['init_movable']
        state.enable = enable
        state.crash = False
        state.reach = False
        state_list.append(state)
#        enable = True
print(len(state_list))
back_end.set_state(state_list)
action = basic.Action()
action.ctrl_phi = 0.0
action.ctrl_vel = 0.9
back_end.set_action([action,]*4)
frame_num = 100

while True:
    start = time.time()
    back_end.set_state(state_list)
    back_end.set_action([action,]*4)
    for i in range(frame_num):   
        #print(back_end.get_state()[1][0].x)
        back_end.step()
        back_end.render()
        state = back_end.get_state()
        print(state)
    end = time.time()
