import numpy as np
import simeck48_diff as si
from os import urandom
from keras.models import load_model
import time
import weakref
import gc

WORD_SIZE = si.WORD_SIZE()
MASK_VAL = 2**WORD_SIZE - 1



def extract_sensitive_bits(raw_x, bits=None):
    id0 = [WORD_SIZE - 1 - v for v in bits]
    id1 = [v + WORD_SIZE * i for i in range(32) for v in id0]
    new_x = raw_x[:, id1]
    return new_x


def make_plaintext_structure(diff, neutral_bits):
    p0l = np.random.randint(0, 2**24, size=1, dtype=np.uint32)
    p0r = np.random.randint(0, 2**24, size=1, dtype=np.uint32)
    for i in neutral_bits:
        if isinstance(i, int):
            i = [i]
        d0 = 0
        d1 = 0
        for j in i:
            d = 1 << j
            d0 |= d >> WORD_SIZE
            d1 |= d & MASK_VAL
        p0l = np.concatenate([p0l, p0l ^ d0])
        p0r = np.concatenate([p0r, p0r ^ d1])
    p1l = p0l ^ diff[0]
    p1r = p0r ^ diff[1]
    return p0l, p0r, p1l, p1r


def collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks):
    p0l, p0r = si.dec_one_round_simeck((p0l, p0r), 0)
    p1l, p1r = si.dec_one_round_simeck((p1l, p1r), 0)
    c0l, c0r = si.encrypt_simeck((p0l, p0r), ks)
    c1l, c1r = si.encrypt_simeck((p1l, p1r), ks)

    return c0l, c0r, c1l, c1r


def attack_with_one_nd(cts, kg_bit_num, kg_offset, sur_kg_low, net, bits, c):
    sur_kg = []
    sur_kg_socres = []
    kg_batch = 2 ** kg_bit_num
    c0l, c0r, c1l, c1r = cts[0], cts[1], cts[2], cts[3]
    n = len(c0l)
    c0l, c0r, c1l, c1r = np.tile(c0l, kg_batch), np.tile(c0r, kg_batch), np.tile(c1l, kg_batch), np.tile(c1r, kg_batch)
    kg_high = np.arange(kg_batch, dtype=np.uint32) << kg_offset
    if sur_kg_low is not None:
        for kg_low in sur_kg_low:
            kg = kg_high | kg_low
            key_guess = np.repeat(kg, n)

            ctdata0l, ctdata0r = si.dec_one_round_simeck((c0l, c0r), key_guess)
            ctdata1l, ctdata1r = si.dec_one_round_simeck((c1l, c1r), key_guess)

            delta_ctdata0l = ctdata0l ^ ctdata1l
            delta_ctdata0r = ctdata0r ^ ctdata1r

            delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r

            secondLast_ctdata0r = si.rol(ctdata0r, 5) & si.rol(ctdata0r, 0) ^ si.rol(ctdata0r, 1) ^ ctdata0l
            secondLast_ctdata1r = si.rol(ctdata1r, 5) & si.rol(ctdata1r, 0) ^ si.rol(ctdata1r, 1) ^ ctdata1l

            delta_secondLast_ctdata0r = secondLast_ctdata0r ^ secondLast_ctdata1r

            X_result = []
            for i in range(4):
                X_result.append(delta_ctdata0l)
                X_result.append(delta_ctdata0r)

                X_result.append(ctdata0l)
                X_result.append(ctdata0r)

                X_result.append(ctdata1l)
                X_result.append(ctdata1r)

                X_result.append(delta_ctdata0rr)
                X_result.append(delta_secondLast_ctdata0r)
            X = si.convert_to_binary(X_result, s_groups=4)

            x = extract_sensitive_bits(X, bits)
            z = net.predict(x, batch_size=10000)
            z = np.log2(z / (1 - z))
            z = np.reshape(z, (kg_batch, n))
            mask = z > 1
            z = np.where(mask, z, 0)
            s = np.sum(z, axis=1)
            for i in range(kg_batch):
                if s[i] > c:
                    sur_kg.append(kg[i])
                    sur_kg_socres.append(s[i])
            X_result.clear()
            weakref.ref(X)
            weakref.ref(x)
            del X_result, X, x
    else:
        kg = kg_high
        key_guess = np.repeat(kg, n)

        ctdata0l, ctdata0r = si.dec_one_round_simeck((c0l, c0r), key_guess)
        ctdata1l, ctdata1r = si.dec_one_round_simeck((c1l, c1r), key_guess)

        delta_ctdata0l = ctdata0l ^ ctdata1l
        delta_ctdata0r = ctdata0r ^ ctdata1r

        delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r

        secondLast_ctdata0r = si.rol(ctdata0r, 5) & si.rol(ctdata0r, 0) ^ si.rol(ctdata0r, 1) ^ ctdata0l
        secondLast_ctdata1r = si.rol(ctdata1r, 5) & si.rol(ctdata1r, 0) ^ si.rol(ctdata1r, 1) ^ ctdata1l

        delta_secondLast_ctdata0r = secondLast_ctdata0r ^ secondLast_ctdata1r

        X_result = []
        for i in range(4):
            X_result.append(delta_ctdata0l)
            X_result.append(delta_ctdata0r)

            X_result.append(ctdata0l)
            X_result.append(ctdata0r)

            X_result.append(ctdata1l)
            X_result.append(ctdata1r)

            X_result.append(delta_ctdata0rr)
            X_result.append(delta_secondLast_ctdata0r)
        X = si.convert_to_binary(X_result, s_groups=4)

        x = extract_sensitive_bits(X, bits)
        z = net.predict(x, batch_size=5000)
        z = np.log2(z / (1 - z))
        z = np.reshape(z, (kg_batch, n))
        mask = z > 1
        z = np.where(mask, z, 0)
        s = np.sum(z, axis=1)
        for i in range(kg_batch):
            if s[i] > c:
                sur_kg.append(kg[i])
                sur_kg_socres.append(s[i])
        X_result.clear()
        weakref.ref(X)
        weakref.ref(x)
        del X_result, X, x

    return sur_kg, sur_kg_socres

def output_sk_kg_diff(sk, sur_kg, kg_scores):
    for i in range(len(sur_kg)):
        print('difference between surviving kg and sk is {}, rank score is {}'.format(hex(sk ^ np.uint32(sur_kg[i])), kg_scores[i]))


def select_top_k_candidates(sur_kg, kg_scores, k=3):
    num = len(sur_kg)
    tp = kg_scores.copy()
    tp.sort(reverse=True)

    if num > k:
        base = tp[k]
    else:
        return sur_kg, kg_scores
    filtered_subkey = []
    filtered_score = []
    for i in range(num):
        if kg_scores[i] > base:
            filtered_subkey.append(sur_kg[i])
            filtered_score.append(kg_scores[i])
    return filtered_subkey, filtered_score


def attack_with_dual_NDs(t, nr, diffs, NBs, nds, bits, c, k):
    nets = []
    for nd in nds:
        nets.append(load_model(nd))
    acc = 0
    time_consumption = np.zeros(t)
    data_consumption = np.zeros(t, dtype=np.uint32)
    for i in range(t):
        print('attack index: {}'.format(i))
        data_num = 0
        start = time.time()

        key = np.random.randint(0, 2**24, size=(4, 1), dtype=np.uint32)
        ks = si.expand_key_simeck(key, nr)
        tk = ks[-1][0]

        num = 0
        while True:
            if num >= 2**4:
                num = -1
                break

            p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[0], NBs[0])
            c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
            sur_kg_1, kg_scores_1 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 0, None, nets[0], bits[0], c[0])

            num += 1
            data_num += 1
            if len(sur_kg_1) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                del c0l, c0r, c1l, c1r
                gc.collect()
                continue
            else:
                print(' ')
                print('Stage 1: ', len(sur_kg_1), ' subkeys survive')
                weakref.ref(c0l)
                weakref.ref(c0r)
                weakref.ref(c1l)
                weakref.ref(c1r)
                del c0l, c0r, c1l, c1r
                gc.collect()
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        kg_1, kg_scores_1 = select_top_k_candidates(sur_kg_1, kg_scores_1, k[0])

        num = 0
        while True:
            if num >= 2**4:
                num = -1
                break
            p0l, p0r, p1l, p1r = make_plaintext_structure(diffs[1], NBs[1])
            c0l, c0r, c1l, c1r = collect_ciphertext_structure(p0l, p0r, p1l, p1r, ks)
            sur_kg_2, kg_scores_2 = attack_with_one_nd([c0l, c0r, c1l, c1r], 12, 12, kg_1, nets[1], bits[1], c[1])
            data_num += 1
            num += 1
            if len(sur_kg_2) == 0:
                print('\r {} plaintext structures generated'.format(num), end='')
                del c0l, c0r, c1l, c1r
                gc.collect()
                continue
            else:
                print(' ')
                print('Stage 2: ', len(sur_kg_2), ' subkeys survive')
                weakref.ref(c0l)
                weakref.ref(c0r)
                weakref.ref(c1l)
                weakref.ref(c1r)
                del c0l, c0r, c1l, c1r
                gc.collect()
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        sur_kg, kg_scores = select_top_k_candidates(sur_kg_2, kg_scores_2, k[1])
        end = time.time()
        output_sk_kg_diff(tk, sur_kg, kg_scores)

        print('{} plaintext structures are generated.'.format(data_num))
        print('the time consumption is ', end - start)
        print('')
        time_consumption[i] = end - start
        data_consumption[i] = data_num

    print('average time consumption is', np.mean(time_consumption))
    print('average structure consumption is', np.mean(data_consumption))



if __name__ == '__main__':
    start0 = time.time()

    nd1 = './freshly_trained_student_nets/13-round-ND/best13depth5(0x0,1000)_student.h5'
    nd2 = './freshly_trained_student_nets/13-round-ND/best13depth5(0x0,1)_student.h5'

    selected_bits_1 = [23, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    selected_bits_2 = [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]

    diff_1 = (0x1000, 0x22000)
    diff_2 = (0x1, 0x22)

    NB_1 = [10 - i for i in range(10)]
    NB_2 = [20 - i for i in range(10)]

    attack_with_dual_NDs(t=100, nr=16, diffs=(diff_1, diff_2), NBs=(NB_1, NB_2), nds=(nd1, nd2),
                         bits=(selected_bits_1, selected_bits_2), c=(512, 512), k=(3, 3))

    end0 = time.time()
    print('the all time consumption is ', end0 - start0)
    print('the one attack of time consumption is ', (end0 - start0)/100)