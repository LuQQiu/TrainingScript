# TrainingScript

Dataset 1: 1.3 million * 100KB = 130GB small files
alluxio root: overmind-imagenet
mounted to local path: /alluxio/alluxio-mountpoint/alluxio-fuse/
```
arena --loglevel info submit pytorch --name=test-job --gpus=0 --workers=4 --cpu 22 --memory 32G --selector alluxio-type=client \
--image=nvcr.io/nvidia/pytorch:21.05-py3 --data-dir=/alluxio/ --sync-mode=git --sync-source=https://github.com/LuQQiu/TrainingScript.git \
"python /root/code/TrainingScript/main.py --epochs 1 --process 8 --subprocess 4 --batch-size 128 --mock-time 0 --print-freq 10 /alluxio/alluxio-mountpoint/alluxio-fuse/dali/train header-1-3m-100kb-130gb.txt"
```

Dataset 2: 1k * 130MB = 130GB medium files
alluxio root: overmind-imagenet
mounted to local path:  /alluxio/alluxio-mountpoint/alluxio-fuse/
```
/alluxio/alluxio-mountpoint/alluxio-fuse/full header-1k-130mb-130gb.txt
```

Dataset 3: 10k * 130MB = 1.3TB medium files
alluxio root: overmind-ml
mounted to local path:  /alluxio/alluxio-mountpoint/alluxio-fuse/
```
arena --loglevel info submit pytorch --name=test-job --gpus=0 --workers=8 --cpu 22 --memory 32G --selector alluxio-type=client \
--image=nvcr.io/nvidia/pytorch:21.05-py3 --data-dir=/alluxio/ --sync-mode=git --sync-source=https://github.com/LuQQiu/TrainingScript.git \
"python /root/code/TrainingScript/main.py --epochs 1 --process 5 --subprocess 8 --batch-size 8 --mock-time 0 --print-freq 10 /alluxio/alluxio-mountpoint/alluxio-fuse/10k130mb header-10k-130mb-1-3TB.txt"
```
```
/alluxio/alluxio-mountpoint/alluxio-fuse/10k130mb header-10k-130mb-1-3TB.txt
```

Dataset: 4: 
```
arena --loglevel info submit pytorch --name=test-job --gpus=0 --workers=4 --cpu 22 --memory 32G --selector alluxio-type=client \
--image=nvcr.io/nvidia/pytorch:21.05-py3 --data-dir=/alluxio/ --sync-mode=git --sync-source=https://github.com/LuQQiu/TrainingScript.git \
"python /root/code/TrainingScript/main.py --epochs 1 --process 8 --subprocess 4 --batch-size 128 --mock-time 0 --print-freq 10 /alluxio/alluxio-mountpoint/alluxio-fuse/10m100kb header-1-3m-100kb-130gb.txt"
```
