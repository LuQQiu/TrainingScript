import argparse
import os
import logging
import time

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from torch.utils.data import DataLoader

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from common.dataset import ImageList

def parse():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet DALI Data Loading')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path to dataset')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default: 1)')
    parser.add_argument('--process', default=4, type=int, metavar='N',
                        help='number of data loading processes (default: 4) in each node')
    parser.add_argument('--thread', default=4, type=int, metavar='N',
                        help='number of data loading threads (default: 4) in each process')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size(default: 256)')
    parser.add_argument('--mock-time', default=0, type=int, metavar='N',
                        help='mock training time in milliseconds')
    args = parser.parse_args()
    return args

def processReadFunc(train_dir, batch_size, num_workers, mock_time, num_shards, shard_id):
    full_file_name = '/header.txt'
    subset_file_name = '/headerPartial.txt'
    writeShardToFile(full_file_name, subset_file_name, num_shards, shard_id);
    train_set = ImageList(subset_file_name, train_dir)
    train_data = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=False, num_workers=num_workers, drop_last=True)
    costs = []
    batch_index = 0
    e_st = time.time()
    g_time = time.time()
    for batch_index, (batch_imgs, batch_labels) in enumerate(train_data):
        cost = time.time() - e_st
        if mock_time != 0:
            time.sleep(mock_time * 0.001)
        costs.append(cost)
        print('[%s] pid: %s, batch %s, cost %.4f, cur sum %s' % (datetime.datetime.now(), os.getpid(), batch_index, cost, len(batch_imgs)))
        e_st = time.time()
    total_time = time.time() - g_time
    qps = batch_index * args.batch_size / total_time
    print("[%s] pid: %s, cost %.4f, qps %.4f" % (datetime.datetime.now(), os.getpid(), total_cost, qps))
    return total_time, qps

def writeShardToFile(full_file_name, subset_file_name, num_shards, shard_id):
    selected_file = open(subset_file_name, "w")
    with open(full_file_name, "r") as full_file:
        for i, line in enumerate(full_file):
            if i % num_shards == shard_id:
                selected_file.write(line)
    selected_file.close()

def main():
    global args
    args = parse()

    if not len(args.data):
        raise Exception("error: No data set provided")

    # set cluster logging
    logger = logging.getLogger('alluxio_dali_data_loading')
    logger.setLevel(logging.INFO)

    master_addr = "N/A"
    master_port = "N/A"
    if 'MASTER_ADDR' in os.environ:
        master_addr = os.environ['MASTER_ADDR']
    if 'MASTER_PORT' in os.environ:
        master_port = os.environ['MASTER_PORT']

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    # arena doesn't support the pytorch multi-processing
    # rank is the node rank instead of the process rank
    rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])

    args.world_size = 1
    if args.distributed:
        logger.info('Launching distributed test with gloo backend')
        # distributed information will be passed in through environment variable WORLD_SIZE and RANK
        torch.distributed.init_process_group(backend='gloo',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        logger.info('Launched distributed test in {} nodes'.format(args.world_size))

    # mock the actual machine learning process
    # each process in each node read a portion of the whole dataset
    num_shards = args.world_size * args.process
    shard_id = range(rank * args.process, (rank + 1) * args.process)

    train_dir = args.data[0]

    logger.info('Launching training script: train_dir[{}], world_size[{}], master_addr[{}], master_port[{}], '
                'rank[{}], processes[{}], threads per process[{}], batch_size[{}], mock_time[{}], num_shards[{}], current_shard_id[{}]'
                .format(train_dir, args.world_size, master_addr, master_port,
                        rank, args.process, args.threads, args.batch_size, args.mock_time, num_shards, shard_id))

    pool = Pool(processes=args.process)
    process_read_func = partial(processReadFunc, train_dir, args.batch_size, args.thread, args.mock_time, args.num_shards)

    for epoch in range(0, args.epochs):
        results = pool.map(process_read_func, shard_id)
        total_qps = 0.0
        for result in results:
            logger.info("Epoch{} process read time {} qps {}".format(epoch, result[0], result[1]))
            total_qps += result[1]
        # TODO(lu) add a socket to receive the img/sec from all nodes in the cluster
        logger.info("Epoch {} training end: total qps {}".format(epoch, total_qps))
        # clear buffer cache requires special docker privileges
        # as a workaround, we clear the buffer cache manually
        # TODO(lu) enable setting docker privileges in arena to support clear buffer cache in script
        logger.info("Starts sleeping to give time for clearing system buffer cache manually")
        time.sleep(300)


if __name__ == '__main__':
    main()
