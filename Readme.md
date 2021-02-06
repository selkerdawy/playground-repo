- To infer image:
```
python main.py -i grumpy.jpg -s 1
```

- To train on CIFAR10:
```
python main.py -s 1 --epochs 200
```

- To evaluate Imagenet dataset:
```
python main.py --evaluate -s 1 --data <path to imagenet>
```

- To train on Imagnet:
```
python main.py -s 1 --data <path to imagenet>
```