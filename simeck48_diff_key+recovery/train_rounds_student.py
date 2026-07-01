import train_nets_student as tn
from time import time
import sys
import os
import argparse

if __name__=='__main__':

    
    parser = argparse.ArgumentParser()
    
    
    # input file location
    parser.add_argument("--GPU_device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=5000)    #与每一层训练的次数有关，bs越大，处理的次数越少，总数据量10**7*2除以bs就是每一层的处理数
    parser.add_argument("--s_groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)    #这个可以加吧？改成10？  可以加，这个可以增加训练的层数以提高准确率.
    parser.add_argument("--num_rounds", type=int, default=14)  #这个是轮数吗？
    parser.add_argument("--depth_value", type=int, default=5)
    parser.add_argument("--file_name", type=str, default="default")
    parser.add_argument("--lr_choice", type=str, default="gohr")
    parser.add_argument("--d0", type=int, default=512)
    parser.add_argument("--d1", type=int, default=48)#改成96？   32是64   所以d1和d2到底是什么？它的大小影响的是什么？
    # parser.add_argument("--d1", type=int, default=96)
    parser.add_argument("--d2", type=int, default=48)
    #parser.add_argument("--d2", type=int, default=96)
    parser.add_argument("--ks_value_1", type=int, default=1)
    parser.add_argument("--ks_value_2", type=int, default=3)
    parser.add_argument("--ks_value_3", type=int, default=3)
    parser.add_argument("--dropout_value", type=float, default=0.8)
    parser.add_argument("--machine_name", type=str, default="vgc")
    parser.add_argument("--num_filters_1", type=int, default=24) #改成48？  32分组的是32, 滤波器
    #parser.add_argument("--num_filters_1", type=int, default=48)
    parser.add_argument("--num_filters_2", type=int, default=24)
    #parser.add_argument("--num_filters_2", type=int, default=48)
    parser.add_argument("--num_filters_3", type=int, default=24)
    #parser.add_argument("--num_filters_3", type=int, default=48)
    parser.add_argument("--reg_param", type=float, default=10**-4)
    parser.add_argument("--high_lr", type=float, default=0.002)
    parser.add_argument("--low_lr", type=float, default=0.0001)
    parser.add_argument("--lr_epoch", type=int, default=10)
    parser.add_argument("--se_ratio", type=int, default=16)
    args = parser.parse_args()
    
    
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_device
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()  
    config.gpu_options.allow_growth = True  
    session = tf.compat.v1.Session(config=config)
    filename = args.file_name
    
    start = time()
    tn.train_distinguisher(num_epochs=args.epochs, num_rounds=args.num_rounds, depth=args.depth_value, filename=args.file_name, s_groups=args.s_groups, bs=args.batch_size, lr_choice=args.lr_choice, d0=args.d0, d1=args.d1, d2=args.d2, ks_value_1=args.ks_value_1, ks_value_2=args.ks_value_2,
                           ks_value_3=args.ks_value_3, machine_name=args.machine_name, dropout_value=args.dropout_value, num_filters_1=args.num_filters_1, num_filters_2=args.num_filters_2, num_filters_3=args.num_filters_3, reg_param=args.reg_param, high_lr=args.high_lr, low_lr=args.low_lr,
                           lr_epoch=args.lr_epoch, se_ratio=args.se_ratio)
    end = time()
    print("time is: "+str(end-start)+" s")
