# TrainingScript

```
arena --loglevel info submit pytorch --name=test-job --gpus=0 --workers=2 --cpu 6 --memory 24G --selector alluxio-type=client --image=nvcr.io/nvidia/pytorch:21.05-py3 --data-dir=/alluxio/ --sync-mode=git --sync-source=https://github.com/LuQQiu/TrainingScript.git "python /root/code/TrainingScript/main.py --epochs 1 --process 2 --subprocess 2 --batch-size 128 --mock-time 500 --print-freq 10 /alluxio/alluxio-mountpoint/alluxio-fuse/full"
```