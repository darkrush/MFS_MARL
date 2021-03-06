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
    parser.add_argument('--nb-epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--nb-cycles-per-epoch', default=10, type=int, help='number of cycles per epoch')
    parser.add_argument('--nb-rollout-steps', default=100, type=int, help='number rollout steps')
    parser.add_argument('--nb-train-steps', default=20, type=int, help='number train steps')
    parser.add_argument('--nb-warmup-steps', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--train-mode', default=1, type=int, help='traing mode')
    parser.add_argument('--decay-coef', default=0.2, type=float, help='decay coef')
    parser.add_argument('--search-method', default = 0, type=int, help='1 if do policy search')
    parser.add_argument('--back-step', default = 7, type=int, help='back step for search policy')
    parser.add_argument('--seed', default=0, type=int, help='random_seed')
    parser.add_argument('--expert-file', default='./expert.pkl', type=str, help='expert actor file dir')
    parser.add_argument('--cuda', default=1, type=int, help='cuda')
    parser.add_argument('--sync-step', default=1, type=int, help='1:Sync step for MARL,0:Async step for RL')

    
    #DDPG args
    parser.add_argument('--actor-lr', default=0.001, type=float, help='actor net learning rate')
    parser.add_argument('--critic-lr', default=0.01, type=float, help='critic net learning rate')
    parser.add_argument('--lr-decay', default=10.0, type=float, help='critic lr decay')
    parser.add_argument('--l2-critic', default=0.01, type=float, help='critic l2 regularization')
    parser.add_argument('--discount', default=0.99, type=float, help='reward discout')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--batch-size', default=256, type=int, help='minibatch size')
    parser.add_argument('--buffer-size', default=1e5, type=int, help='memory buffer size')

    #Model args
    parser.add_argument('--hidden1', default=128, type=int, help='number of hidden1')
    parser.add_argument('--hidden2', default=128, type=int, help='number of hidden2')
    parser.add_argument('--not-LN', dest='layer_norm', action='store_false',help='model without LayerNorm')
    parser.set_defaults(layer_norm=True)
    parser.add_argument('--nb-pos', default=2, type=int, help='number of pos')
    parser.add_argument('--nb-laser', default=128, type=int, help='number of laser')
    parser.add_argument('--nb-actions', default=2, type=int, help='number of actions')
    
    #Env args
    parser.add_argument('--train-env', default='./src/scenario/scenario_train_v1.yaml', type=str, help='train env file')
    parser.add_argument('--search-env', default='./src/scenario/scenario_search_v1.yaml', type=str, help='search env file')
    parser.add_argument('--eval-env', default='./src/scenario/scenario_train_v1.yaml', type=str, help='eval env file')
    parser.add_argument('--step-t', default=1.0, type=float, help='sim time for each step')
    parser.add_argument('--train-dt', default=0.001, type=float, help='sim delta time in train')
    parser.add_argument('--search-dt', default=1.0, type=float, help='sim delta time in search')
    parser.add_argument('--eval-dt', default=0.001, type=float, help='sim delta time in eval')

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