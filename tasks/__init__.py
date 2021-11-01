from . import mnist
from . import cifar10
from . import imagenet
from . import imdb

# todo: modify this. Maybe return list of models for each task
model_names = list(set(imagenet.model_names) | set(cifar10.model_names) | set(mnist.model_names))