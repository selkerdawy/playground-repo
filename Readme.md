
# Without Converting Model
- To infer image:
```
python main.py -i grumpy.jpg
```

- To train on CIFAR10:
```
python main.py --epochs 200
```

- To evaluate Imagenet dataset:
```
python main.py --evaluate --data-dir <path to imagenet>
```

- To train on Imagnet:
```
python main.py --data-dir <path to imagenet>
```

# Converting Model
- To convert convolution to APoT 5-bit quantized convolution:
```
python main.py -i grumpy.jpg --apot '{"bit": 5}'
```

- To convert convolution and linear layers to HAQ 4-bit quantization:
```
python main.py -i grumpy.jpg --haq '{"w_bit": 5, "a_bit": 5}'
```

- To increase stride of convolution and upsample
```
python main.py -i grumpy.jpg --convup '{"scale": 2, "mode": "bilinear"}'
```

- To perform Tucker decomposition
```
python main.py --data-dir ~/pytorch_datasets/imagenet --tucker-decompose '{"ranks":[20,20]}' --task imagenet --pretrained True --arch resnet18 --layer-start 1
```

- To perform depthwise decomposition
```
python main.py --data-dir ~/pytorch_datasets/imagenet --depthwise-decompose '{"threshold":0.3}' --task imagenet --pretrained True --arch resnet18 --layer-start 1
```

- To downsize every other epoch
```
python main.py --data-dir ~/pytorch_datasets/ --scale-input '{"scale_factor":0.25, "recompute_scale_factor":true}'  --task cifar10 --pretrained False --arch resnet20 --conversion-epoch-start 0 --conversion-epoch-end 200 --conversion-epoch-step 2
```