import argparse
parser = argparse.ArgumentParser(description='Train prototypical networks')

# data args
parser.add_argument('--data.train', type=str, default='cu_birds', metavar='trainset', nargs='+', help="data set name (default: imagenet)")
parser.add_argument('--data.val', type=str, default='cu_birds', metavar='valset', nargs='+',
                    help="data set name (default: imagenet)")
parser.add_argument('--data.test', type=str, default='cu_birds', metavar='testset', nargs='+',
                    help="data set name (default: imagenet)")
parser.add_argument('--data.num_workers', type=int, default=32, metavar='NEPOCHS',
                    help='number of workers that pre-process images in parallel')

# model args
default_model_name = 'noname'
parser.add_argument('--model.name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.many_names', type=str, default='', metavar='many_names', nargs='+',
                    help="when testing on several networks")
parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
parser.add_argument('--model.classifier', type=str, default='cosine', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")
parser.add_argument('--model.dropout', type=float, default=0, help="Adding dropout inside a basic block of widenet")

# train args
parser.add_argument('--train.determ', type=int, default=1, help="Set a random seed in the beginning of the training (default: True)")
parser.add_argument('--train.batch_size', type=int, default=16, metavar='BS',
                    help='number of images in a batch')
parser.add_argument('--train.max_iter', type=int, default=500000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.weight_decay', type=float, default=7e-4, metavar='WD',
                    help="weight decay coef")
parser.add_argument('--train.optimizer', type=str, default='momentum', metavar='OPTIM',
                    help='optimization method (default: momentum)')

parser.add_argument('--train.learning_rate', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.lr_policy', type=str, default='cosine', metavar='LR_policy',
                    help='learning rate decay policy')
parser.add_argument('--train.lr_decay_step_gamma', type=int, default=1e-1, metavar='DECAY_GAMMA',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.lr_decay_step_freq', type=int, default=10000, metavar='DECAY_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_final_lr', type=float, default=8e-5, metavar='FINAL_LR',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.exp_decay_start_iter', type=int, default=30000, metavar='START_ITER',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.cosine_anneal_freq', type=int, default=4000, metavar='ANNEAL_FREQ',
                    help='the value to divide learning rate by when decayin lr')
parser.add_argument('--train.nesterov_momentum', action='store_true', help="If to augment query images in order to avearge the embeddings")

parser.add_argument('--train.eval_freq', type=int, default=5000, metavar='CKPT',
                    help='Saving model params after each CKPT epochs (default: 1000)')
parser.add_argument('--train.eval_size', type=int, default=300, metavar='CKPT',
                    help='Saving model params after each CKPT epochs (default: 1000)')
parser.add_argument('--train.resume', type=int, default=1, help="Resume training starting from the last checkpoint (default: True)")
parser.add_argument('--train.image_summary', type=float, default=0, help="If wants to drop tensorboard images")


parser.add_argument('--dump.name', type=str, default='noname', metavar='CKPT',
                    help='Name for dataset dump')
parser.add_argument('--dump.mode', type=str, default='noname', metavar='CKPT',
                    help='Name for dataset dump')
parser.add_argument('--dump.size', type=int, default=1000, metavar='CKPT',
                    help='how many tasks to dump')


# test args
parser.add_argument('--test.set', type=str, default='test', choices=['train', 'test', 'val', 'test_new'], metavar='NCOPY', help='The number of test episodes sampled')
parser.add_argument('--test.size', type=int, default=600, metavar='NCOPY',
                    help='The number of test episodes sampled')
parser.add_argument('--test.augment_query', action='store_true', help="If to augment query images in order to avearge the embeddings")
parser.add_argument('--test.augment_support', action='store_true', help="If to augment support images in order to avearge the embeddings")
parser.add_argument('--test.distance', type=str, choices=['cos', 'l2'], default='cos', help="If to augment support images in order to avearge the embeddings")
parser.add_argument('--test.no_log', action='store_true', help="not storing the results")

# log args
args = vars(parser.parse_args())
