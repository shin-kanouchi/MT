#/usr/bin/env python
#-*-coding:utf-8-*-

import argparse
import collections
import time
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import pickle
from gensim.models import word2vec

parser = argparse.ArgumentParser()
parser.add_argument('--train', '-t', default='/work/kanouchi/git/shin/MT/exp_KFTT/kyoto-dev.1000best.ja-en', type=str, help='input train file')
parser.add_argument('--source', '-s', default='/work/kanouchi/git/shin/MT/exp_KFTT/test/kyoto-dev.ja', type=str, help='input source file')
parser.add_argument('--output', '-o', default='../result/kyoto-dev.1best.ja-en', type=str, help='output data')
#parser.add_argument('--unit',  '-u', default=50, type=int, help='number of units per layer')
parser.add_argument('--msd',   '-m', default='msrl', type=str, help='imput rl or msd or msrl')
parser.add_argument('--null',  '-n', default='null', type=str, help='use null or not')
parser.add_argument('--w2v',   '-c', default='w2v', type=str, help='use (w2v) or (No)')
parser.add_argument('--f_name', '-f', default='wordph50k-jaen-null-NOpre-cw2v-u25-w1-d0-NOTkana-AdaDelta-msrl', type=str, help='output e.g. model.XXXX.1')
parser.add_argument('--epoch_num', '-e', default='1', type=str, help='word dict id')
parser.add_argument('--ph_table', '-p', default='/work/kanouchi/git/shin/MT/exp_KFTT/model/ja-en/model/phrase-table', type=str, help='')
parser.add_argument('--f2e', '-d', default='/work/kanouchi/git/shin/MT/exp_KFTT/model/ja-en/model/lex.f2e', type=str, help='f2e lex')
parser.add_argument('--e2f', '-g', default='/work/kanouchi/git/shin/MT/exp_KFTT/model/ja-en/model/lex.e2f', type=str, help='e2f lex')
parser.add_argument('--use_phrase', '-u', default='Yes', type=str, help='use phrase or no')

args = parser.parse_args()

SHIN = '..'
#SHIN = '/work/kanouchi/ALAGIN/shin'
#n_units   = args.unit   # number of units per layer
file_name = args.f_name
msd_label = args.msd
use_null  = args.null
w2v_No    = args.w2v
#en_dict_file = '%s/word_dict/word_dict_en.dump.%s' % (SHIN, file_name)
#ja_dict_file = '%s/word_dict/word_dict_ja.dump.%s' % (SHIN, file_name)
model_file   = '%s/model/model.%s' % (SHIN, file_name)
#F_lang, E_lang = 'en', 'ja'
F_lang, E_lang = 'ja', 'en'
w2v_f        = '%s/w2v_model/matrix25.w3.%s.model.python3' % (SHIN, F_lang)
w2v_e        = '%s/w2v_model/matrix25.w3.%s.model.python3' % (SHIN, E_lang)


if msd_label == 'msrl':
    m_label, s_label, r_label, l_label = 0, 1, 2, 3
elif msd_label == 'msd':
    m_label, s_label, r_label, l_label = 0, 1, 2, 2
else:
    m_label, s_label, r_label, l_label = 0, 1, 0, 1
n_label = l_label + 1

print("load model...")
model = pickle.load(open('/work/kanouchi/git/shin/MT/model/model.%s.%s' % (args.f_name, args.epoch_num), 'rb'))
print("load en_dic")
with open('../word_dict/word_dict_en.dump.' + args.f_name, 'rb') as f_en:
    vocab_en = pickle.load(f_en)
print("load ja_dic")
with open('../word_dict/word_dict_ja.dump.' + args.f_name, 'rb') as f_ja:
    vocab_ja = pickle.load(f_ja)

f_sents = []
for line in open(args.source):
    f_sents.append(line[:-1])

dic_ph = {}
dic_f2e = {}
dic_e2f = {}

print("load p(lex)")
for line in open(args.f2e):
    items = line.strip().split(' ')
    e = items[0]
    f = items[1]
    score = items[2]
    dic_f2e['%s ||| %s' % (f, e)] = float(score)

for line in open(args.e2f):
    items = line.strip().split(' ')
    f = items[0]
    e = items[1]
    score = items[2]
    dic_e2f['%s ||| %s' % (f, e)] = float(score)

print("load phraase-table")
for line in open(args.ph_table):
    items = line.strip().split(' ||| ')
    f_phrase = items[0]
    e_phrase = items[1]
    ph_score = list(map(float, items[2].split(' ')))
    align = items[3].split(' ')
    scores = items[4].split(' ')
    #if scores[2] != '2': continue
    f_list = f_phrase.split(' ')
    e_list = e_phrase.split(' ')
    e2f_list, f2e_list = [], []
    for f_e in align:
        f, e = f_e.split('-')
        #if '%s|||%s' % (en_list[int(en)], ja_list[int(ja)]) in dic:
        e2f_list.append(dic_e2f['%s ||| %s' % (f_list[int(f)], e_list[int(e)])])
        f2e_list.append(dic_f2e['%s ||| %s' % (f_list[int(f)], e_list[int(e)])])
    dic_ph['%s ||| %s' % (f_phrase, e_phrase)] = (align, ph_score, f2e_list, e2f_list)


def where_align(f_list, num_f, num_e, len_e, lex_f2e, lex_e2f):
    if num_e == 0:
        f_list[num_f][1] = 1.0
    if num_e == len_e - 1:
        f_list[num_f][3] = 1.0
    if num_e < len_e - 1 and num_e > 0:
        f_list[num_f][2] = 1.0
    f_list[num_f][0] = .0
    if float(lex_f2e) > f_list[num_f][4]:
        f_list[num_f][4] = float(lex_f2e)
    if float(lex_e2f) > f_list[num_f][5]:
        f_list[num_f][5] = float(lex_e2f)
    return f_list


def use_align(f_ph, e_ph, aligns, f2e, e2f):
    len_f, len_e = len(f_ph), len(e_ph)
    f_aligns, e_aligns = [], []
    f_list = [[1.0, .0, .0, .0, .0, .0] for col1 in range(len_f)]
    e_list = [[1.0, .0, .0, .0, .0, .0] for col1 in range(len_e)]
    for align, lex_f2e, lex_e2f in zip(aligns, f2e, e2f):
        num_f_str, num_e_str = align.split('-')
        num_f, num_e = int(num_f_str), int(num_e_str)
        f_list = where_align(f_list, num_f, num_e, len_e, lex_f2e, lex_e2f)
        e_list = where_align(e_list, num_e, num_f, len_f, lex_e2f, lex_f2e)
    return f_list, e_list


def embed_word(word, embed, align, f_e, tra):
    if word >= 0:
        w = np.array([word], np.int32)
        x = chainer.Variable(w, volatile=not tra)
        p = embed(x)
    else:
        if f_e == 'e' and word in w2v_matrix_e_inv:  # vocab_ja_inv[word].decode("utf-8") in w2v_matrix_e.vocab:  # in w2v_matrix_e.vocab.keys():
            w = w2v_matrix_e[[w2v_matrix_e_inv[word]]]
            p = chainer.Variable(w, volatile=not tra)
        elif f_e == 'f' and word in w2v_matrix_f_inv:  # vocab_en_inv[word] in w2v_matrix_f.vocab:
            w = w2v_matrix_f[[w2v_matrix_f_inv[word]]]
            p = chainer.Variable(w, volatile=not tra)
        else:
            w = np.array([0], np.int32)
            x = chainer.Variable(w, volatile=not tra)
            p = embed(x)

    if use_null == 'null':
        add_emb = np.array([align], np.float32)
        z = chainer.Variable(add_emb, volatile=not tra)
        return F.concat((p, z))
    else:
        return p


def cul_RAE(phrase, embed, w1, w2, align_list, f_e, tra=False):
    ph_key = '%s|||%s' % ('|'.join(map(str, phrase)), f_e)
    if ph_key in ph_dict:
        p_left = ph_dict[ph_key]
    else:
        i = 0
        p_left = embed_word(phrase[0], embed, align_list[i], f_e, tra)
        for w in phrase[1:]:
            w_key = '%s||%s' % (str(w), f_e)
            i += 1
            if w_key in ph_dict:
                p_left = ph_dict[w_key]
            else:
                p_right = embed_word(w, embed, align_list[i], f_e, tra)
                c1_c2 = F.concat((p_left, p_right))
                p_left = F.tanh(w1(c1_c2))
                ph_dict[w_key] = p_left
        ph_dict[ph_key] = p_left
    return p_left


def return_MSD(r0, l0, r1, l1):
    if r0 + 1 == l1:
        return m_label  # "mono"
    elif r1 + 1 == l0:
        return s_label  # "swap"
    elif l1 - r0 > 0:
        return r_label  # "dright"
    elif l0 - r1 > 0:
        return l_label  # "dleft"
    else:
        return 4


def train(tree, tra=False):
    f0, e0, f1, e1, ans, _, f0_null, e0_null, f1_null, e1_null, ph0_score, ph1_score = tree
    
    if '%s||%s||%s||%s||%d' % ('|'.join(map(str, f0)), '|'.join(map(str, f1)),'|'.join(map(str, e0)),'|'.join(map(str, e1)), ans) in all_dic:
        return_ans = all_dic['%s||%s||%s||%s||%d' % ('|'.join(map(str, f0)), '|'.join(map(str, f1)),'|'.join(map(str, e0)),'|'.join(map(str, e1)), ans)]
    else:
        p_f0 = cul_RAE(f0, model.embed_f, model.w1_f, model.w2_f, f0_null, 'en', tra)
        p_f1 = cul_RAE(f1, model.embed_f, model.w1_f, model.w2_f, f1_null, 'en', tra)
        p_e0 = cul_RAE(e0, model.embed_e, model.w1_e, model.w2_e, e0_null, 'ja', tra)
        p_e1 = cul_RAE(e1, model.embed_e, model.w1_e, model.w2_e, e1_null, 'ja', tra)
        if args.use_phrase == 'Yes':
            ph0_np = np.array([ph0_score], np.float32)
            ph0 = chainer.Variable(ph0_np, volatile=not tra)
            ph1_np = np.array([ph1_score], np.float32)
            ph1 = chainer.Variable(ph1_np, volatile=not tra)

            f0e0 = F.concat((p_f0, p_e0, ph0))
            f1e1 = F.concat((p_f1, p_e1, ph1))
            p_f0e0 = F.tanh(model.trainable.ww0(f0e0))
            p_f1e1 = F.tanh(model.trainable.ww1(f1e1))

            f01_e01 = F.concat((p_f0e0, p_f1e1))
        else:
            f01_e01 = F.concat((p_f0, p_e0, p_f1, p_e1))
        y = F.softmax(F.tanh(model.trainable.w0(f01_e01)))  # y = F.tanh(model.trainable.w0(f01_e01))
        return_ans = np.log(cuda.to_cpu(y.data)[0][ans])
        all_dic['%s||%s||%s||%s||%d' % ('|'.join(map(str, f0)), '|'.join(map(str, f1)),'|'.join(map(str, e0)),'|'.join(map(str, e1)), ans)] = return_ans
    return return_ans


def convert_tree(vocab, phrase, w2v_c, f_e):
    phrase_list = []
    for word in phrase:
        if word in vocab:
            pass
        elif word.isdigit() and len(word) > 3:
            vocab[word] = 1
        elif word.isdigit():
            vocab[word] = 2
        elif word not in vocab and w2v_No == 'w2v':
            vocab[word] = w2v_c
            w2v_c -= 1
        elif word not in vocab and w2v_No == 'NOTw2v':
            vocab[word] = 0
        phrase_list.append(vocab[word])
    return phrase_list, w2v_c


def read_1000best(sent_e, sent_f):
    ph_f = sent_f.split(' ')
    e1 = []
    trees = []
    o_sent = []
    items = sent_e.split(' ')
    i = 0
    w2v_c = -1
    for item in items:
        if item.startswith('|') and item.endswith('|'):
            l1_str, r1_str = item[1: -1].split('-')
            l1, r1 = int(l1_str), int(r1_str)
            f1 = ph_f[l1: r1 + 1]
            ph_f1, w2v_c = convert_tree(vocab_en, f1, w2v_c,'en')
            ph_e1, w2v_c = convert_tree(vocab_ja, e1, w2v_c,'ja')
            o_sent += e1
            if i == 0:
                i = 1
                ph_e0, ph_f0, l0, r0, e0, f0 = ph_e1, ph_f1, l1, r1, e1, f1
                e1 = []
                continue
            #print ph_e1, ph_f1, l1, r1, e1, f1
            if '%s ||| %s' % (' '.join(f0), ' '.join(e0)) not in dic_ph:
                dic_ph['%s ||| %s' % (' '.join(f0), ' '.join(e0))] = (['0-0'], [1.0,1.0,1.0,1.0], [1], [1])
            if '%s ||| %s' % (' '.join(f1), ' '.join(e1)) not in dic_ph:
                dic_ph['%s ||| %s' % (' '.join(f1), ' '.join(e1))] = (['0-0'], [1.0,1.0,1.0,1.0], [1], [1])
            align_0, ph0_score, f2e0, e2f0 = dic_ph['%s ||| %s' % (' '.join(f0), ' '.join(e0))]
            align_1, ph1_score, f2e1, e2f1 = dic_ph['%s ||| %s' % (' '.join(f1), ' '.join(e1))]
            #print align_0, ph0_score, f2e0, e2f0
            #ph0_score, ph1_score = map(float, ph0_score_str), map(float, ph1_score_str)
            f0_null, e0_null = use_align(ph_f0, ph_e0, align_0, f2e0, e2f0)
            f1_null, e1_null = use_align(ph_f1, ph_e1, align_1, f2e1, e2f1)
            MSD = return_MSD(r0, l0, r1, l1)
            trees.append([ph_f0, ph_e0, ph_f1, ph_e1, MSD, '0, 0, 0', f0_null, e0_null, f1_null, e1_null, ph0_score, ph1_score])
            ph_e0, ph_f0, l0, r0, e0, f0 = ph_e1, ph_f1, l1, r1, e1, f1
            e1 = []
        else:
            e1.append(item)
    return trees, o_sent


if w2v_No == 'w2v':
    print("load w2v f")
    w2v_matrix_f = word2vec.Word2Vec.load(w2v_f)
    print("load w2v e")
    w2v_matrix_e = word2vec.Word2Vec.load(w2v_e)
    w2v_matrix_f_inv = {v: k for k, v in w2v_matrix_f.vocab.items()}
    w2v_matrix_e_inv = {v: k for k, v in w2v_matrix_e.vocab.items()}

count = 0
start_at = time.time()
cur_at = start_at
f = open(args.output, "w")
ph_dict = {}
all_dic = {}
for line in open(args.train):
    if line == "":
        continue
    msd_score = [0 for raw in range(n_label)]
    num, sent, score, total = line.strip().split(' ||| ')
    scores = score.split(' ')
    trees, new_sent = read_1000best(sent, f_sents[int(num)])
    for tree in trees:
        msd_score[tree[4]] += train(tree)
        #print tree, train(tree)
    #print 'LexReo0=%.4f LexReo1=%.4f LexReo2=%.4f LexReo3=%.4f %s%s %s%s %s%s %s%s TM0=%s TM1=%s TM2=%s TM3=%s' % (msd_score[0], msd_score[1], msd_score[2], msd_score[3], scores[7], scores[8], scores[9], scores[10], scores[11], scores[12], scores[13], scores[14], scores[16], scores[17], scores[18], scores[19])
    if n_label == 2:
        reo_score = 'LexReo0=%.4f LexReo1=%.4f' % (msd_score[0], msd_score[1])
    elif n_label == 3:
        reo_score = 'LexReo0=%.4f LexReo1=%.4f LexReo2=%.4f' % (msd_score[0], msd_score[1], msd_score[2])
    else:
        reo_score = 'LexReo0=%.4f LexReo1=%.4f LexReo2=%.4f LexReo3=%.4f' % (msd_score[0], msd_score[1], msd_score[2], msd_score[3])
    new_score = '%s %s%s %s%s %s%s %s%s TM0=%s TM1=%s TM2=%s TM3=%s' % (reo_score, scores[9], scores[10], scores[11], scores[12], scores[13], scores[14], scores[15], scores[16], scores[18], scores[19], scores[20], scores[21])
    f.write('%s ||| %s ||| %s ||| %s\n' % (num, ' '.join(new_sent), total, new_score))
    count += 1
    if count % 10000 == 0:
        now = time.time()
        print(num)
        print('{:.2f} sec'.format(now - cur_at))
