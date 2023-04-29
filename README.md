# Neural Stochastic Differential Games for Time-series Analysis


## Training and Evaluation
### CIFAR10
 - DenseNet40
```
python main.py -gpu_num 0 -dataset "cifar10" -network 0 
```
 - WideResNet32
```
python main.py -gpu_num 0 -dataset "cifar10" -network 1
```
 - ResNet110
```
python main.py -gpu_num 0 -dataset "cifar10" -network 2
```
 - ResNet110 SD
```
python main.py -gpu_num 0 -dataset "cifar10" -network 3 
```

## CIFAR100
 - DenseNet40
```
python main.py -gpu_num 0 -dataset "cifar100" -network 0 -lamda 0.01
```
 - WideResNet32
```
python main.py -gpu_num 0 -dataset "cifar100" -network 1
```
 - ResNet110
```
python main.py -gpu_num 0 -dataset "cifar100" -network 2
```
 - ResNet110 SD
```
python main.py -gpu_num 0 -dataset "cifar100" -network 3 -lamda 0.01
```
### ImageNet
 - DenseNet162
```
python main.py -gpu_num 0 -dataset "imagenet" -network 0 -initial 3
```
 - ResNet152
```
python main.py -gpu_num 0 -dataset "imagenet" -network 1 -initial 3
```
