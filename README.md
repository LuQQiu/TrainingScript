# TrainingScript

Dataset 1: 1.3 million * 100KB = 130GB small files
alluxio root: overmind-imagenet
mounted to local path: /alluxio/alluxio-mountpoint/alluxio-fuse/
```
arena --loglevel info submit pytorch --name=test-job --gpus=0 --workers=2 --cpu 6 --memory 24G --selector alluxio-type=client \
--image=nvcr.io/nvidia/pytorch:21.05-py3 --data-dir=/alluxio/ --sync-mode=git --sync-source=https://github.com/LuQQiu/TrainingScript.git \
"python /root/code/TrainingScript/main.py --epochs 1 --process 2 --subprocess 2 --batch-size 128 --mock-time 0 --print-freq 10 /alluxio/alluxio-mountpoint/alluxio-fuse/dali/train header-1-3m-100kb-130gb.txt"
```

Dataset 2: 1k * 130MB = 130GB medium files
alluxio root: overmind-imagenet
mounted to local path:  /alluxio/alluxio-mountpoint/alluxio-fuse/
```
/alluxio/alluxio-mountpoint/alluxio-fuse/full header-1k-130mb-130gb.txt
```
