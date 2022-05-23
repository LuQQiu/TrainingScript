#!/usr/bin/python

# Require python > 3.0

import argparse
import csv
import datetime
import os
import torch
import time

from prometheus_client import multiprocess, CollectorRegistry, Summary
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)
LATENCY = Summary("read_latency", "Read request latency")


def get_file_name_list(csv_file, max_len):
    with open(csv_file) as cf:
        reader = csv.reader(cf)
        urls = []
        count = 0
        for row in reader:
            urls.append(row[0].strip())
            count += 1
            if count >= max_len:
                break
    print("Prepared dataset with {} files".format(len(urls)))
    return urls


class LocalDataset(Dataset):
    def __init__(self, size, filelist, prefix, H=224, W=224):
        self.file_name_list = get_file_name_list(filelist, size)
        self.size = len(self.file_name_list)
        self.prefix = prefix
        if self.size == 0:
            raise (RuntimeError("Found 0 files in the given file name list"))
        self.H = H
        self.W = W

    def __getitem__(self, index):
        file_name = self.file_name_list[index]
        file_path = os.path.join(self.prefix, file_name)
        v = 0
        chunk_size = 8192
        first_read = True
        try:
            start = time.time()
            with open(file_path, "rb") as f:
                while True:
                    if first_read:
                        data = f.read(12)
                        if not data:
                            break
                        if len(data) > data.count(b"\x00"):
                            raise (RuntimeError("File {} contains {} all zero bytes".format(file_name, len(data))))
                    else:
                        data = f.read(chunk_size)
                    first_read = False
                    if not data:
                        break
                    v += len(data)
            LATENCY.observe(time.time() - start)
        except Exception as e:
            print(e)
            pass
        return v

    def __len__(self):
        return self.size


def start_load(args):
    torch.distributed.init_process_group(backend='gloo', init_method='env://', timeout=datetime.timedelta(seconds=300))
    dataset = LocalDataset(size=args.number_of_files, filelist=args.file_name_list, prefix=args.path_prefix)
    sampler = DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=True)

    for epoch in range(0, args.epochs):
        count = 0
        st = time.time()
        global LATENCY
        LATENCY = Summary("read_latency_{}".format(epoch), "Read request latency for epoch {}".format(epoch))
        for _ in data_loader:
            if count % 100 == 0 and args.local_rank == 0:
                print("epoch {}: processing {} in".format(epoch, count))
            count += 1
        total_ts = time.time() - st
        print("epoch {}: time cost for {} batches: {}".format(epoch, count, total_ts))
        if count > 0:
            print("epoch {}: overall Avg.TrainingLatency {:.2f} ms, per node throughput {:.2f} files/second"
                  .format(epoch, total_ts * 1000.0 / count / args.batch_size,
                          count * args.batch_size / total_ts))
            for metric in registry.collect():
                name = metric.name
                num = 0
                total = 0
                for item in metric.samples:
                    if item.name == name + "_sum":
                        total = item.value
                    elif item.name == name + "_count":
                        num = item.value
                print("metric: {}: {:.2f} ms".format(name, total * 1000.0/num))


def main():
    parser = argparse.ArgumentParser(description='Alluxio POSIX API benchmark test')
    parser.add_argument('-e', '--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default: 1)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size(default: 128)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-i', '--file_name_list', default='data.csv', type=str,
                        help='path to input csv file with file names to load')
    parser.add_argument('-n', '--number_of_files', default=5000000, type=int, metavar='N',
                        help='number of files to be processed')
    parser.add_argument('-p', '--path_prefix', type=str, metavar='N',
                        help='path prefix of the list files')
    parser.add_argument('-r', '--local_rank', type=int, metavar='N',
                        help='local rank')
    args = parser.parse_args()
    print(args)
    start_load(args)


if __name__ == '__main__':
    main()
