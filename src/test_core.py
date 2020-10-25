from MSE import core
from scenario.paser import  parse_senario
scenario = parse_senario('./scenario/test_core.yaml')
world = core.World(scenario['agent_groups'],0.1)
world.reset()
world._reset_render()
state = world.get_state()


world.set_total_time(10.0)
print(world.get_total_time())
for i in range(1000):
    world.step(step_num=1)
    action = core.Action()
    action.ctrl_vel = 0.5
    action.ctrl_phi = 0.22
    world.set_action([True] * 4,[action,action,action,action])
    obs = world.get_obs()
    #print(obs)
    #print('****************')
    #world.render()
