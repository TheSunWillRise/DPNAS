import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from graphviz import Digraph

from nas_framework import DPNetwork
from nas_architect import Architect
from genotypes import PRIV_PRIMITIVES
import random
from scheduler import CosineWithRestarts


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--batch_size_dp', type=int, default=300, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.02, help='init learning rate')

parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
parser.add_argument('--report_freq_w', type=int, default=20, help='report frequency')
parser.add_argument('--report_freq_theta', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=2, help='test (go over all validset) frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='number of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.6, help='portion of data for model weights training')

parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch masters')
parser.add_argument('--arch_weight_decay', type=float, default=0, help='weight decay for arch encoding')
parser.add_argument('--gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--inner_steps', type=int, default=3, help='number of inner updates')
parser.add_argument('--inner_lr', type=float, default=0.001, help='learning rate for inner updates')
parser.add_argument('--valid_inner_steps', type=int, default=3, help='number of inner updates for validation')
parser.add_argument('--n_archs', type=int, default=10, help='number of candidate archs')
parser.add_argument('--prefix', type=str, default='.', help='parent save path: /opt/ml/disk/ for seven')
# lstm
parser.add_argument('--controller_type', type=str, default='ENAS', help='SAMPLE | LSTM | ENAS')
parser.add_argument('--controller_hid', type=int, default=100, help='hidden num of controller')
parser.add_argument('--controller_temperature', type=float, default=None, help='temperature for lstm')
parser.add_argument('--controller_tanh_constant', type=float, default=None, help='tanh constant for lstm')
parser.add_argument('--entropy_coeff', nargs='+', type=float, default=[0.005, 0.005], help='coefficient for entropy: [normal, reduce]')
parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of layers in lstm')
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5, help='coefficient for entropy')
# controller warmup
parser.add_argument('--controller_start_training', type=int, default=20, help='When training of controller starts')
# scheduler restart
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--T_mul', type=float, default=2.0, help='multiplier for cycle')
parser.add_argument('--T0', type=int, default=10, help='The maximum number of epochs within the first cycle')
parser.add_argument('--store', type=int, default=0, help='Whether to store the model')
parser.add_argument('--benchmark_path', type=str, default=None, help='Path to restore the benchmark model')
parser.add_argument('--restore_path', type=str, default=None, help='Path to restore the model')

parser.add_argument('--no_dp', action='store_true', default=False)
parser.add_argument('--archmaster', type=str, default='dp')
parser.add_argument('--multi_forward', action='store_true', default=False)
parser.add_argument('--dp_clip', type=float, default=1)
parser.add_argument('--dp_sigma', type=float, default=0.8)



args = parser.parse_args()
if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)

args.private = not args.no_dp
if "SEVEN_JOB_ID" in os.environ:
    args.save = '{}-DP-search-{}'.format(os.environ['SEVEN_JOB_ID'], time.strftime("%Y%m%d-%H%M%S"))
else:
    args.save = 'NAS-DP-search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = os.path.join(args.prefix, args.save)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()
logger.info('Enable DP: '+str(args.private))

CIFAR_CLASSES = 10


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info('GPU device = %d' % args.gpu)
    else:
        logging.info('no GPU available, use CPU!!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)
    logging.info("================= OPS =================" )
    logger.info(PRIV_PRIMITIVES)
    logging.info("=======================================" )

    if (args.private):
        criterion = nn.CrossEntropyLoss(reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')

    criterion = criterion.to(device)
    model = DPNetwork(args.init_channels, CIFAR_CLASSES, args.layers,
                    criterion, device, controller_type=args.controller_type,
                    controller_hid=args.controller_hid,
                    controller_temperature=args.controller_temperature,
                    controller_tanh_constant=args.controller_tanh_constant,
                    controller_op_tanh_reduce=args.controller_op_tanh_reduce,
                    entropy_coeff=args.entropy_coeff, args=args)

    optimizer = torch.optim.SGD(
        model.model_parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)

    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)

    split = int(np.floor(args.train_portion * num_train))
    # valid_meta_begin = int(np.floor(args.train_portion[0] * num_train))
    # valid_arch_begin = int(np.floor(args.train_portion[1] * num_train))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_dp if args.private else args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=4
    )


    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True, num_workers=4
    )
    if args.scheduler == "naive_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
    elif args.scheduler == 'cosine_restart':
        scheduler = CosineWithRestarts(
            optimizer, t_0=args.T0, eta_min=args.learning_rate_min, last_epoch=-1, factor=args.T_mul
        )
    else:
        assert False, "unsupported schudeler type: %s" % args.scheduler

    benchmark_model = None

    if args.restore_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.prefix, args.benchmark_path)))

    model.to(device)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model.optimizer = optimizer
    model.arch_normal_master.force_uniform = True
    try:
        model.arch_reduce_master.force_uniform = True
    except:
        pass

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        # # test lr
        # for param_group in meta_optimizer.param_groups:
        #     logging.info('###epoch %d LR %f %f', epoch, param_group['lr'], lr)
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj, train_normal_ent, train_reduce_ent = update_w(
            train_queue, model, device)

        logging.info('Updating W: train_acc %f train_normal ent %f train_reduce_ent %f', train_acc, train_normal_ent, train_reduce_ent)
        if epoch >= args.controller_start_training:
            model.arch_normal_master.force_uniform = False
            try:
                model.arch_reduce_master.force_uniform = False
            except:
                pass
            reward, n_ent, r_ent = update_theta(valid_queue, architect, device, epoch)
            logging.info('Updating Theta: Average Reward %f Normal ent %f Reduce ent %f', reward, n_ent, r_ent)

        if epoch % args.test_freq == 0:
            model.test(test_queue, args.n_archs, logger, args.save, "%d" % epoch, benchmark_model)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

    # TODO: do final test
    model.test(test_queue, args.n_archs, logger, args.save, "Final", benchmark_model)

    # save model
    if args.store == 1:
        # torch.save(model, os.path.join(args.save, 'models.pt'))
        utils.save(model, os.path.join(args.save, 'models.pt'))


def update_w(train_queue, model, device):
    # update of meta-weights is defined here.
    objs = utils.AvgrageMeter()
    normal_ent = utils.AvgrageMeter()
    reduce_ent = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()     # just set train flag, for bn
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)

        logits, loss, _, _, n_ent, _, _, r_ent = model.step(input, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        normal_ent.update(n_ent.item(), 1)
        reduce_ent.update(r_ent.item(), 1)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq_w == 0:
            logging.info('Updating W: Step=%03d Loss=%e Top1=%f Top5=%f Noraml_ENT=%f, Reduce_ENT=%f',
                         step, objs.avg, top1.avg, top5.avg, normal_ent.avg, reduce_ent.avg)
    return top1.avg, objs.avg, normal_ent.avg, reduce_ent.avg

def update_theta(valid_queue, architect, device, epoch=0):
    objs = utils.AvgrageMeter()
    normal_ent = utils.AvgrageMeter()
    reduce_ent = utils.AvgrageMeter()
    for step, (input, target) in enumerate(valid_queue):
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)
        reward, n_ent, r_ent = architect.step(input, target, epoch)
        objs.update(reward.item(), n)
        normal_ent.update(n_ent.item(), 1)
        reduce_ent.update(r_ent.item(), 1)
        if step % args.report_freq_theta == 0:
            logging.info('Updating Theta Step=%03d Reward=%e Noraml_ENT=%f, Reduce_ENT=%f', step, objs.avg,
                         normal_ent.avg, reduce_ent.avg)

    return objs.avg, normal_ent.avg, reduce_ent.avg


if __name__ == '__main__':
    main()






