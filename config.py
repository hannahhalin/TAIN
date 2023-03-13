import argparse
import json
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--dataset_name', type=str, default='vimeo90k',
                      choices=['vimeo90k', 'ucf', 'snufilm', 'middlebury'],
                      help='Name of the dataset [vimeo90k, ucf, snufilm, middlebury]')
data_arg.add_argument('--data_root', type=str, default='path/to/vimeo90k')
                      
# model
model_arg = add_argument_group('Model')
model_arg.add_argument('--depth', type=int, default=3, help='# of pooling')
model_arg.add_argument('--n_resblocks', type=int, default=12)
data_arg.add_argument('--n_resgroups', type=int, default=5,
                      help='Number of resgroups.')

# training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='test',
                       choices=['train', 'test'])
learn_arg.add_argument('--loss', type=str, default='1*L1')
learn_arg.add_argument('--lr', type=float, default=1e-4)
learn_arg.add_argument('--beta1', type=float, default=0.9)
learn_arg.add_argument('--beta2', type=float, default=0.99)
learn_arg.add_argument('--batch_size', type=int, default=16)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--test_mode', type=str, default='hard',
                      choices=['easy', 'medium', 'hard', 'extreme'],
                       help='Test mode to evaluate on SNU-FILM dataset')
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=200)
learn_arg.add_argument('--resume', action='store_true', default=False)
learn_arg.add_argument('--resume_exp', type=str, default='TAIN')

# misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--exp_name', type=str, default='exp')
misc_arg.add_argument('--log_iter', type=int, default=1000)
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--num_workers', type=int, default=2)


def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))

    # save all arguments in a json file
    if not os.path.exists('checkpoint/' + args.exp_name):
        os.makedirs('checkpoint/' + args.exp_name)

        args_file = os.path.join('checkpoint/', args.exp_name, 'args.json')
        
        # save the passed arguments
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2,
                      sort_keys=True)

    return args, unparsed
