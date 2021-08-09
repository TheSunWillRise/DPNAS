
### MNIST
```
CUDA_VISIBLE_DEVICES=0 python mian.py --dataset=mnist --batch_size=2048 --lr=2 --noise_multiplier=2.15 --epochs=40
```

### FashionMNIST
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=fmnist --batch_size=2048 --lr=2 --noise_multiplier=2.15 --epochs=40
```

### CIFAR10
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=cifar10 --batch_size=2048 --lr=2 --noise_multiplier=2.0664 --epochs=30
```

