from src.run_ddpg import run_ddpg
import wandb

def parse_arg():
    import argparse
    parser = argparse.ArgumentParser(description='DDPG on pytorch')
    # Wandb args
    parser.add_argument('--name', default='test', type=str, help='name of this exp')
    parser.add_argument('--group', default='debug', type=str, help='group of this exp')
    parser.add_argument('--tags',nargs='*',  help='tags of this exp')
    parser.add_argument('--job-type', default='debug', type=str, help='jobv type of this exp')
    #Training args
    parser.add_argument('--nb-epoch', default=400, type=int, help='number of epochs')
    parser.add_argument('--nb-cycles-per-epoch', default=10, type=int, help='number of cycles per epoch')
    parser.add_argument('--nb-rollout-steps', default=100, type=int, help='number rollout steps')
    parser.add_argument('--nb-train-steps', default=20, type=int, help='number train steps')
    #parser.add_argument('--max-episode-length', default=1000, type=int, help='max steps in one episode')
    parser.add_argument('--nb-warmup-steps', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--train-mode', default=1, type=int, help='traing mode')
    parser.add_argument('--search-method', default = 0, type=int, help='1 if do policy search')
    parser.add_argument('--back-step', default = 7, type=int, help='back step for search policy')
    
    #Model args
    parser.add_argument('--hidden1', default=128, type=int, help='number of hidden1')
    parser.add_argument('--hidden2', default=128, type=int, help='number of hidden2')
    parser.add_argument('--not-LN', dest='layer_norm', action='store_false',help='model without LayerNorm')
    parser.set_defaults(layer_norm=True)
    
    #DDPG args
    parser.add_argument('--actor-lr', default=0.001, type=float, help='actor net learning rate')
    parser.add_argument('--critic-lr', default=0.01, type=float, help='critic net learning rate')
    parser.add_argument('--lr-decay', default=10.0, type=float, help='critic lr decay')
    parser.add_argument('--l2-critic', default=0.01, type=float, help='critic l2 regularization')
    parser.add_argument('--batch-size', default=256, type=int, help='minibatch size')
    parser.add_argument('--discount', default=0.99, type=float, help='reward discout')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--nocuda', dest='with_cuda', action='store_false',help='disable cuda')
    parser.set_defaults(with_cuda=True)
    parser.add_argument('--buffer-size', default=1e6, type=int, help='memory buffer size')
    #Exploration args
    parser.add_argument('--action-noise', dest='action_noise', action='store_true',help='enable action space noise')
    parser.set_defaults(action_noise=False)
    parser.add_argument('--parameter-noise', dest='parameter_noise', action='store_true',help='enable parameter space noise')
    parser.set_defaults(parameter_noise=False)
    parser.add_argument('--stddev', default=0.6, type=float, help='action noise stddev')
    parser.add_argument('--noise-decay', default=0, type=float, help='action noise decay')
    parser.add_argument('--SGLD-mode', default=0, type=int, help='SGLD mode, 0: no SGLD, 1: actor sgld only, 2: critic sgld only, 3: both actor & critic')
    parser.add_argument('--no-SGLD-noise', dest='SGLD_noise', action='store_false',help='disable SGLD noise')
    parser.set_defaults(SGLD_noise=True)
    parser.add_argument('--num-pseudo-batches', default=0, type=int, help='SGLD pseude batch number')
    parser.add_argument('--nb-rollout-update', default=50, type=int, help='number of SGLD rollout actor step')
    parser.add_argument('--temp', default=1, type=float, help='Temperature of SGLD')
    #Other args
    parser.add_argument('--rand-seed', default=314, type=int, help='random_seed')
    
    args = parser.parse_args()
    return args

args = parse_arg()

run = wandb.init(config=args, 
                project="multi-fidelity-sim",
                tags=args.tags,
                name=args.name,
                group=args.group,
                dir='./',
                job_type=args.job_type)

run.config.update(args)
run_ddpg(run.config,run)