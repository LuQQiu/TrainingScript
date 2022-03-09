import argparse
import logging
import multiprocessing
import os
import time

from datetime import datetime
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
                        help='path to dataset, and name of the file containing all file names of the dataset')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default: 1)')
    parser.add_argument('--process', default=4, type=int, metavar='N',
                        help='number of processes (default: 4) to do data loading and mock training in each node')
    parser.add_argument('--subprocess', default=2, type=int, metavar='N',
                        help='number of data loading subprocesses (default: 2) in each mock training process')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size(default: 128)')
    parser.add_argument('--mock-time', default=0, type=int, metavar='N',
                        help='mock training time in milliseconds (default: 0)')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    args = parser.parse_args()
    return args


def process_read(train_dir, file_name_list, batch_size, num_workers, mock_time, print_freq, num_shards, shard_id, message_queue, res_queue):
    full_file_name = "/root/code/TrainingScript/" + file_name_list
    pid = os.getpid()
    subset_file_name = '/root/code/TrainingScript/headerPartial{}.txt'.format(pid)
    select_files_to_read(full_file_name, subset_file_name, num_shards, shard_id)

    train_set = ImageList(subset_file_name, train_dir)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    batch_index = 0
    e_st = time.time()
    g_time = time.time()
    for batch_index, (batch_imgs, batch_labels) in enumerate(train_data):
        cost = time.time() - e_st
        if mock_time != 0:
            time.sleep(mock_time * 0.001)
        if batch_index % print_freq == 0:
            message_queue.put('pid: {}, batch {}, cost {:3f}, cur sum {:3f}'.format(
                os.getpid(), batch_index, cost, len(batch_imgs)))
        e_st = time.time()
    total_train_time = time.time() - g_time
    batch_train_time = total_train_time / batch_index
    qps = batch_index * args.batch_size / total_train_time
    message_queue.put("pid: {}, total cost {:3f}, batch cost {:3f}, qps {:3f},".format(pid, total_train_time, batch_train_time, qps))
    res_queue.put([batch_train_time, qps])


def select_files_to_read(full_file_name, subset_file_name, num_shards, shard_id):
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

    train_dir = args.data[0]
    file_name_list = args.data[1]

    print('Launching training script: train_dir[{}], world_size[{}], master_addr[{}], master_port[{}], '
                'rank[{}], processes[{}], subprocesses per mock training process[{}], batch_size[{}], mock_time[{}], num_shards[{}]'
                .format(train_dir, args.world_size, master_addr, master_port,
                        rank, args.process, args.subprocess, args.batch_size, args.mock_time, num_shards))
    jobs = [None] * args.process
    res_queue = multiprocessing.Queue()
    message_queue = multiprocessing.Queue()
    for epoch in range(0, args.epochs):
        for i in range(0, args.process):
            p = multiprocessing.Process(target=process_read, args=(
                train_dir, file_name_list, int(args.batch_size), args.subprocess, args.mock_time, args.print_freq, num_shards, rank * args.process + i, message_queue, res_queue))
            jobs[i] = p
            p.start()

        while True:
            if message_queue.empty():
                time.sleep(30)
            else:
                print(message_queue.get())
            if message_queue.empty():
                break

        for proc in jobs:
            proc.join()

        total_qps = 0.0
        total_time = 0.0
        while not res_queue.empty():
            res = res_queue.get()
            res_time = res[0]
            res_qps = res[1]
            print("Epoch{} process average batch read time {} qps {}".format(epoch, res_time, res_qps))
            total_qps += res_qps
            total_time += res_time
        # TODO(lu) add a socket to receive the img/sec from all nodes in the cluster
        average_batch_train_time = total_time / args.process
        print("Epoch {} training end: average per training process batch train time {}, per node qps {}".format(epoch, average_batch_train_time, total_qps))
        # clear buffer cache requires special docker privileges
        # as a workaround, we clear the buffer cache manually
        # TODO(lu) enable setting docker privileges in arena to support clear buffer cache in script
        print("Starts sleeping to give time for clearing system buffer cache manually")
        time.sleep(300)


if __name__ == '__main__':
    main()
