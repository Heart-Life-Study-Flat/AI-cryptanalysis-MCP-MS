import simeck48_diff as si

import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
import copy
import random

block_size = 48
key_size = si.WORD_SIZE()

def test_bits_sensitivity(n=10**7, nr=7, net_path='./', diff=(0x0, 0x200), folder='./key_bit_sensitivity_res/', s_groups=4, id=[0]):
    acc = np.zeros(1+2)
    X, Y = si.make_train_data(n=n, nr=nr, diff=diff, s_groups=4)
    net = load_model(net_path)
    list = net.evaluate(X, Y, batch_size=10000, verbose=0)
    loss = list[0]
    acc[2] = list[1]
    print('The initial acc is ', acc[2])
    X, Y = si.make_train_data_key(n=n, nr=nr, diff=diff, s_groups=4, id=id)
    list1 = net.evaluate(X, Y, batch_size=10000, verbose=0)

    acc[0] = list1[1]
    print('cur bit positions are ', id)
    print('Half key: the decrease of the acc is ', acc[2] - acc[0])



net_path = './freshly_trained_nets/11-round-ND/best11depth5(0x0,1).h5'
folder = './key_bit_sensitivity_res/11-round-ND/0x0-0x1/'

id = []

for i in range(key_size):
    id.append(i)
    test_bits_sensitivity(n=10**6, nr=11, net_path=net_path, diff=(0x0, 0x10), folder=folder, s_groups=4, id=id)

id = []
for i in range(key_size-1, -1, -1):
    id.insert(0, i)
    test_bits_sensitivity(n=10**6, nr=11, net_path=net_path, diff=(0x0, 0x10), folder=folder, s_groups=4, id=id)