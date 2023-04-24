import tensorflow as tf
from networks import Networks

model = Networks.tiny_darknet()

model.summary()