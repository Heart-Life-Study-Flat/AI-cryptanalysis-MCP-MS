import numpy as np
from os import urandom

#SIMON96/96
def WORD_SIZE():
    return(48)

MASK_VAL = 2 ** WORD_SIZE() - 1

const = [0xfffd, 0xfffd, 0xfffd, 0xfffd,
         0xfffd, 0xfffc, 0xfffd, 0xfffc,
         0xfffc, 0xfffc, 0xfffd, 0xfffc,
         0xfffc, 0xfffd, 0xfffc, 0xfffd,
         0xfffc, 0xfffd, 0xfffd, 0xfffc,
         0xfffc, 0xfffc, 0xfffc, 0xfffd,
         0xfffd, 0xfffd, 0xfffc, 0xfffc]


def rol(x, k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))) 


def ror(x, k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)) 


def enc_one_round_simon(p, k):
    c0, c1 = p[0], p[1]
    # print("c0 shape",c0)
    ls_1_x = ((c0 >> (WORD_SIZE() - 1)) + (c0 << 1)) & MASK_VAL
    ls_8_x = ((c0 >> (WORD_SIZE() - 8)) + (c0 << 8)) & MASK_VAL
    ls_2_x = ((c0 >> (WORD_SIZE() - 2)) + (c0 << 2)) & MASK_VAL
    # XOR Chain
    xor_1 = (ls_1_x & ls_8_x) ^ c1
    xor_2 = xor_1 ^ ls_2_x
    # print("xor_2 = ",xor_2)
    new_c0 = k ^ xor_2
    return (new_c0, c0)

def dec_one_round_simon(c,k):        #一轮解密
    c0, c1 = c[0], c[1]
    ls_1_c1 = ((c1 >> (WORD_SIZE() - 1)) + (c1 << 1)) & MASK_VAL
    ls_8_c1 = ((c1 >> (WORD_SIZE() - 8)) + (c1 << 8)) & MASK_VAL
    ls_2_c1 = ((c1 >> (WORD_SIZE() - 2)) + (c1 << 2)) & MASK_VAL
    # Inverse XOR Chain
    xor_1 = k ^ c0
    xor_2 = xor_1 ^ ls_2_c1
    new_c0 = (ls_1_c1 & ls_8_c1) ^ xor_2
    return (c1, new_c0)


#SIMON96/96
def expand_key_simon(k, nr, version=96):
    # k = k & ((2 ** 64) - 1)
    #zseq = 0b01100111000011010100100010111110110011100001101010010001011111   #fixed sequence  z0
    fs = 0b11001101101001111110001000010100011001001011000000111011110101     #fixed sequence  z2
    ks = []
    #k = np.transpose(k)
    for i in range(len(k)):
        ks.append([])
    # k_init = [[((k >> (WORD_SIZE() * (3 - i))) & MASK_VAL) for i in range(2)]]
    # print("k_init = ", k_init)
    for i in range(len(k)):
        ks[i] = k[i] & 0x0000ffffffffffff
    rc = MASK_VAL ^ 3
    for x in range(2, nr):
        rs_3 = ((ks[0] << (WORD_SIZE() - 3)) + (ks[0] >> 3)) & MASK_VAL
        rs_1 = ((rs_3 << (WORD_SIZE() - 1)) + (rs_3 >> 1)) & MASK_VAL
        #rs_3 = ror(k_reg[0], 3)
        #rs_1 = ror(rs_3, 1)
        c_z = ((fs >> (x % 62)) & 1) ^ rc
        new_k = c_z ^ rs_1 ^ rs_3 ^ ks[1]
        ks.insert(0, new_k)
    ks = list(reversed(ks))   #rk0,rk1,rk2,rk3,...

    return ks


def encrypt_simon(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x, y = enc_one_round_simon((x, y), k)
    return(x, y)


def convert_to_binary(arr, s_groups=1):
  X = np.zeros((8 * WORD_SIZE() * s_groups, len(arr[0])), dtype=np.uint8)
  for i in range(8 * WORD_SIZE() * s_groups):
    index = i // WORD_SIZE() 
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose() 
  return(X)


def make_train_data(n, nr, diff=(0x0, 0x800000000000), s_groups=1):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.random.randint(0, 2 ** 48, size=(2, n), dtype=np.uint64)  #(2, n)由密钥长度决定
    num_rand_samples = np.sum(Y == 0)
    ks = expand_key_simon(keys, nr)
    X_result = []
    
    
    for i in range(s_groups):
        plain0l = np.random.randint(0, 2**48, size=n, dtype=np.uint64)
        plain0r = np.random.randint(0, 2**48, size=n, dtype=np.uint64)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        plain1l[Y == 0] = np.random.randint(0, 2**48, size=num_rand_samples, dtype=np.uint64)
        plain1r[Y == 0] = np.random.randint(0, 2**48, size=num_rand_samples, dtype=np.uint64)
        ctdata0l, ctdata0r = encrypt_simon((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt_simon((plain1l, plain1r), ks)
        
        delta_ctdata0l = ctdata0l ^ ctdata1l
        delta_ctdata0r = ctdata0r ^ ctdata1r

        delta_ctdata0 = ctdata0l ^ ctdata0r
        delta_ctdata1 = ctdata1l ^ ctdata1r

        delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r
        secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l
        
        
 
        secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l
 
        delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
        
        
        thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r, 8) & rol(secondLast_ctdata0r, 1) ^ rol(secondLast_ctdata0r, 2)
        thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r, 8) & rol(secondLast_ctdata1r, 1) ^ rol(secondLast_ctdata1r, 2)
        
        
        delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r


        X_result.append(delta_ctdata0l)
        X_result.append(delta_ctdata0r)

        X_result.append(ctdata0l)
        X_result.append(ctdata0r)
        
        X_result.append(ctdata1l)
        X_result.append(ctdata1r)

        #X_result.append(secondLast_ctdata0r)
        #X_result.append(secondLast_ctdata1r)
        
        X_result.append(delta_secondLast_ctdata0r)
        X_result.append(delta_thirdLast_ctdata0r)
    
    X= convert_to_binary(X_result, s_groups=s_groups)
    #X = np.tile(X,s_groups)
    return (X, Y)

def make_target_key_bit_diffusion_data(ks, id=0):
    ks = ks ^ 2 ** id
    return ks

def make_train_data_key(n, nr, diff=(0x0, 0x800000000000), s_groups=1,id=0):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.random.randint(0, 2 ** 48, size=(2, n), dtype=np.uint64)  # (2, n)由密钥长度决定
    num_rand_samples = np.sum(Y == 0)
    ks = expand_key_simon(keys, nr+1)
    X_result = []

    for i in range(s_groups):
        plain0l = np.random.randint(0, 2 ** 48, size=n, dtype=np.uint64)
        plain0r = np.random.randint(0, 2 ** 48, size=n, dtype=np.uint64)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        plain1l[Y == 0] = np.random.randint(0, 2 ** 48, size=num_rand_samples, dtype=np.uint64)
        plain1r[Y == 0] = np.random.randint(0, 2 ** 48, size=num_rand_samples, dtype=np.uint64)
        ctdata0l0, ctdata0r0 = encrypt_simon((plain0l, plain0r), ks)
        ctdata1l0, ctdata1r0 = encrypt_simon((plain1l, plain1r), ks)

        ks[nr] = make_target_key_bit_diffusion_data(ks[nr], id)

        ctdata0l, ctdata0r = dec_one_round_simon((ctdata0l0, ctdata0r0), ks[nr])  # 解密1轮
        ctdata1l, ctdata1r = dec_one_round_simon((ctdata1l0, ctdata1r0), ks[nr])  # 解密1轮

        delta_ctdata0l = ctdata0l ^ ctdata1l
        delta_ctdata0r = ctdata0r ^ ctdata1r

        delta_ctdata0 = ctdata0l ^ ctdata0r
        delta_ctdata1 = ctdata1l ^ ctdata1r

        delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r
        secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l

        secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l

        delta_secondLast_ctdata0r = secondLast_ctdata0r ^ secondLast_ctdata1r

        thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r, 8) & rol(secondLast_ctdata0r, 1) ^ rol(
            secondLast_ctdata0r, 2)
        thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r, 8) & rol(secondLast_ctdata1r, 1) ^ rol(
            secondLast_ctdata1r, 2)

        delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r

        X_result.append(delta_ctdata0l)
        X_result.append(delta_ctdata0r)

        X_result.append(ctdata0l)
        X_result.append(ctdata0r)

        X_result.append(ctdata1l)
        X_result.append(ctdata1r)

        # X_result.append(secondLast_ctdata0r)
        # X_result.append(secondLast_ctdata1r)

        X_result.append(delta_secondLast_ctdata0r)
        X_result.append(delta_thirdLast_ctdata0r)

    X = convert_to_binary(X_result, s_groups=s_groups)
    # X = np.tile(X,s_groups)
    return (X, Y)



