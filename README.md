# Getting Started
- Install some libraries:
```
### clone repo ###
cd TorchPlayground
pip install - requirements.txt
```

# Command Template
```
python main.py --task <imagenet | cifar10 | mnist> --arch <model>
            [--train | --infer <sample path> | --evaluate]
            --<conversion> '{<conversion parameters>}'
                [--layer-start <num>] [--layer-end <num>]
                [--conversion-epoch-start <num>] [--conversion-epoch-end <num>] [--conversion-epoch-step <num>]
            [--epochs <num>] [--batch-size <num>] [--momentum <num>] [--optimizer <opt>] [--pretrained <true | false>]
            [--lr <num>] [--lr-schedule <scheduler>] [--lr-step-size <num>] [--lr-milestones <nums>]
            [--cpu | --gpu <gpu-id>]
```

There are more options that can be listed by running `python main.py --help`

# Without Converting Model
- To infer image:
```
python main.py --task cifar10 -i grumpy.jpg
```

- To train on CIFAR10:
```
python main.py --task cifar10 --epochs 200
```

- To evaluate Imagenet dataset:
```
python main.py --task imagenet --evaluate --data-dir <path to imagenet>
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

- To quantize convolution and linear layers using DeepShift:
```
python main.py --deepshift '{"shift_type": "PS"}'
```

- To increase stride of convolution and upsample
```
python main.py -i grumpy.jpg --convup '{"scale": 2, "mode": "bilinear"}'
```

- To perform Tucker decomposition
```
python main.py --data-dir ~/datasets/imagenet --tucker-decompose '{"ranks":[20,20]}' --task imagenet --pretrained True --arch resnet18 --layer-start 1
```

- To perform depthwise decomposition
```
python main.py --data-dir ~/datasets/imagenet --depthwise-decompose '{"threshold":0.3}' --task imagenet --pretrained True --arch resnet18 --layer-start 1
```

- To downsize every other epoch
```
python main.py --data-dir ~/datasets/ --scale-input '{"scale_factor":0.25, "recompute_scale_factor":true}'  --task cifar10 --pretrained False --arch resnet20 --conversion-epoch-start 0 --conversion-epoch-end 200 --conversion-epoch-step 2
```
