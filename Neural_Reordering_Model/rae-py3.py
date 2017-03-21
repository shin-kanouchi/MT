#/usr/bin/env python
#-*-coding:utf-8-*-

import argparse
import collections
import time
import numpy as np
import chainer
#from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import pickle
from datetime import datetime
from gensim.models import word2vec
import random

now_date = datetime.now().strftime('%m%d%H%M')
parser = argparse.ArgumentParser()
parser.add_argument('--train', '-r', default='phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.head5000k.50k', type=str, help='train file')
parser.add_argument('--test',  '-t', default='phrase_mslr.over2.pre2.align.ph-lex-score.rmD10.mid500k.10k', type=str, help='test file')
parser.add_argument('--opt',   '-o', default='SGD', type=str, help='input SGD, AdaGrad, AdaDelta or Adam')
parser.add_argument('--pt',    '-l', default=0.02, type=float, help='input optimized first score')
parser.add_argument('--unit',  '-u', default=25, type=int, help='number of units per layer')
parser.add_argument('--batch', '-b', default=25, type=int, help='minibatch size')
parser.add_argument('--epoch', '-e', default=15, type=int, help='number of epochs')
parser.add_argument('--eval',  '-v', default=1, type=int, help='number of epochs per evaluation')
parser.add_argument('--save',  '-s', default=1, type=int, help='number of epochs per save model')
parser.add_argument('--drop',  '-d', default=0, type=float, help='rate of drop out')
parser.add_argument('--alpha', '-a', default=0.12, type=float, help='HyperP between rec and reo')
parser.add_argument('--unk_w', '-w', default=1, type=float, help='UNK word limit')
parser.add_argument('--msd',   '-m', default='msrl', type=str, help='imput rl or msd or msrd')
parser.add_argument('--pre2',  '-p', default=False, type=str, help='use pre2(True) or not(False)')
parser.add_argument('--phrase', '-x', default='phrase', type=str, help='use phrase(True) or not(False)')
parser.add_argument('--null',  '-n', default='null', type=str, help='use null or not')
parser.add_argument('--w2v',   '-c', default='w2v', type=str, help='use (w2v) or (No)')
parser.add_argument('--kana',  '-k', default='NOTkana', type=str, help='marge (kana) or (not)')
parser.add_argument('--f_name', '-f', default='now_date', type=str, help='output e.g. model.XXXX.1')
parser.add_argument('--log_path', '-g', default='.', type=str)
parser.add_argument('--jaen', '-j', default='ja-en', type=str)
args = parser.parse_args()

SHIN = '..'
LOG = args.log_path
UNK_NUM   = args.unk_w  # under n is unk word
n_epoch   = args.epoch  # number of epochs
n_units   = args.unit   # number of units per layer
batchsize = args.batch  # minibatch size
Alpha     = args.alpha  # a parameter between RAE error and REO error
drop_out  = args.drop   # rate of drop out
per_eval  = args.eval   # number of epochs per evaluation
per_save  = args.save   # number of epochs per save
file_name = args.f_name
opt_score = args.pt
opt_name  = args.opt
msd_label = args.msd
use_pre2  = args.pre2
use_null  = args.null
w2v_No    = args.w2v
use_kana  = args.kana
en_dict_file = '%s/word_dict/word_dict_en.dump.%s' % (SHIN, file_name)
ja_dict_file = '%s/word_dict/word_dict_ja.dump.%s' % (SHIN, file_name)
model_file   = '%s/model/model.%s' % (SHIN, file_name)
log_file     = '%s/log/%s/out.log.%s' % (SHIN, LOG, file_name)
if args.jaen == 'ja-en': F_lang, E_lang = 'ja', 'en'
else: F_lang, E_lang = 'en', 'ja'
w2v_f        = '%s/w2v_model/matrix25.w3.%s.model.python3' % (SHIN, F_lang)
w2v_e        = '%s/w2v_model/matrix25.w3.%s.model.python3' % (SHIN, E_lang)
unk_count = 0

if msd_label == 'msrl':
    m_label, s_label, r_label, l_label = 0, 1, 2, 3
    n_label = 4
elif msd_label == 'msd':
    m_label, s_label, r_label, l_label = 0, 1, 2, 2
    n_label = 3
else:
    m_label, s_label, r_label, l_label = 0, 1, 0, 1
    n_label = 2


def convert_tree(vocab, phrase, w_count, tra):
    """
    単語を数字で置き換える
    未知(頻度N回以下)の単語が入力された時0を返す
    ３桁以上の数字（年号）が来た時は例外で１を返す
    （未知語をさらに人出で細かく分けることで，改善の余地あり）
    出力はそれぞれ固有の単語を表す数字のlist(句)を返す
    """
    phrase_list = []
    for word in phrase:
        if word in vocab:
            pass
        elif word.isdigit() and len(word) > 3:
            vocab[word] = 1
        elif word.isdigit():
            vocab[word] = 2
        elif w_count[word] == 1 and random.randint(1, 50) % 29 == 0 and tra == 'Yes':
            vocab[word] = 0
        elif tra == 'Yes':
            vocab[word] = vocab['len_vocab']
            vocab['len_vocab'] += 1
        elif tra != 'No' and word in tra.vocab:
            vocab[word] = tra[[word]]
        else:
            vocab[word] = 0
        phrase_list.append(vocab[word])
    return phrase_list


def word_count(w_count, phrase):
    for word in phrase:
        w_count[word] += 1


def counts(path, f_count, e_count):
    for line in open(path):
        if line.strip() == "": continue
        item = line.strip().split(' ||| ')
        f0, e0, f1, e1 = item[1], item[2], item[3], item[4]
        word_count(f_count, f0.split())
        word_count(f_count, f1.split())
        word_count(e_count, e0.split())
        word_count(e_count, e1.split())


def where_align(f_list, num_f, num_e, len_e, lex_f2e, lex_e2f):
    if num_e == 0:
        f_list[num_f][1] = 1.0
    if num_e == len_e -1:
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
    len_f, len_e  = len(f_ph), len(e_ph)
    f_aligns, e_aligns = [], []
    f_list = [[1.0,.0,.0,.0,.0,.0] for col1 in range(len_f)]
    e_list = [[1.0,.0,.0,.0,.0,.0] for col1 in range(len_e)]
    for align, lex_f2e, lex_e2f in zip(aligns, f2e, e2f):
        num_f_str, num_e_str = align.split('-')
        num_f, num_e = int(num_f_str), int(num_e_str)
        f_list = where_align(f_list, num_f, num_e, len_e, lex_f2e, lex_e2f)
        e_list = where_align(e_list, num_e, num_f, len_f, lex_e2f, lex_f2e)
    return f_list, e_list


def read_corpus(path, vocab_f, vocab_e, f_count, e_count, tra):
    trees = []
    for line in open(path):
        if line.strip() == "": continue
        item = line.strip().split(' ||| ')
        # 60155 ||| pattern ||| 模様 ||| came ||| に ちな ||| dleft11 ||| 0.353659 0.060976 0.012195 0.573171 ||| 0-0 ||| 0.168103 0.222222 0.110482 0.111406 ||| f2e 0.1114058 ||| e2f 0.2222222 ||| 0-1 ||| 0.0152284 0.0333333 0.00102494 0.000102824 ||| f2e 0.0015400 ||| e2f 0.0333333
        #sent_num = item[0]
        f0_list, e0_list = item[1].split(), item[2].split()
        f1_list, e1_list = item[3].split(), item[4].split()
        MSD, preMSD = item[5][:-2], list(map(float, item[6].split()))
        align_0, align_1 = item[7].split(), item[11].split()
        f2e0, e2f0 = item[9].split()[1:], item[10].split()[1:]
        f2e1, e2f1 = item[13].split()[1:], item[14].split()[1:]
        ph0_score = list(map(float, item[8].split()))
        ph1_score = list(map(float, item[12].split()))
        
        if MSD == 'mono': ans = m_label
        elif MSD == 'swap': ans = s_label
        elif MSD == 'dright': ans = r_label
        else: ans = l_label

        f0_null, e0_null = use_align(f0_list, e0_list, align_0, f2e0, e2f0)
        f1_null, e1_null = use_align(f1_list, e1_list, align_1, f2e1, e2f1)

        f0l = convert_tree(vocab_f, f0_list, f_count, tra[0])
        f1l = convert_tree(vocab_f, f1_list, f_count, tra[0])
        e0l = convert_tree(vocab_e, e0_list, e_count, tra[-1])
        e1l = convert_tree(vocab_e, e1_list, e_count, tra[-1])
        trees.append([f0l, e0l, f1l, e1l, ans, preMSD, f0_null, e0_null, f1_null, e1_null, ph0_score, ph1_score])
    return trees


def embed_word(word, embed, align, tra):
    if type(word) == int:
        w = np.array([word], np.int32)
        x = chainer.Variable(w, volatile=not tra)
        p = embed(x)
    else: p = chainer.Variable(word, volatile=not tra)

    if use_null == 'null':
        add_emb = np.array([align], np.float32)
        z = chainer.Variable(add_emb, volatile=not tra)
        return F.concat((p, z))
    else:
        return p


def cul_RAE(phrase, embed, w1, w2, align_list, tra):
    """
    ワードをembeddingしてからRAEの畳み込みを行う
    フレーズが１単語からなる場合はembeddingのみ行う
    lossは２乗誤差関数を利用
    lossとフレーズベクトルを返す
    """
    loss = 0
    i = 0
    len_phrase = len(phrase) - 1
    p_left = embed_word(phrase[0], embed, align_list[i], tra)
    if len(phrase) == 1: 
        len_phrase = 1
    for w in phrase[1:]:
        i += 1
        p_right = embed_word(w, embed, align_list[i], tra)
        c1_c2   = F.concat((p_left, p_right))
        p_left  = F.tanh(w1(c1_c2))
        cc1_cc2 = F.tanh(w2(p_left))
        loss   += F.mean_squared_error(cc1_cc2, c1_c2)/2
    if drop_out != 0:
        p_left = F.dropout(p_left, ratio=drop_out, train=tra)
    return loss/len_phrase, p_left


def train(tree, evaluate, tra):
    """
    Neural Reordering Modelを使って４つのフレーズを畳み込みn_unitにする
    ソフトマックスクロスエントロピーを使い正解のMSD（n次元でどこかひとつだけ１で残りは０）との誤差を計算して返す
    """
    f0, e0, f1, e1, ans, preMSD, f0_null, e0_null, f1_null, e1_null, ph0_score, ph1_score = tree
    loss_f0, p_f0 = cul_RAE(f0, model.embed_f, model.w1_f, model.w2_f, f0_null, tra)
    loss_f1, p_f1 = cul_RAE(f1, model.embed_f, model.w1_f, model.w2_f, f1_null, tra)
    loss_e0, p_e0 = cul_RAE(e0, model.embed_e, model.w1_e, model.w2_e, e0_null, tra)
    loss_e1, p_e1 = cul_RAE(e1, model.embed_e, model.w1_e, model.w2_e, e1_null, tra)
    #if use_pre2 == 'pre':
    #    preMSD_np = np.array([preMSD], np.float32)
    #    preMSD_emb = chainer.Variable(preMSD_np, volatile=not tra)
    #    f01_e01 = F.concat((preMSD_emb, p_f0, p_e0, p_f1, p_e1))
    if args.phrase == 'phrase':
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
    y = F.tanh(model.trainable.w0(f01_e01))

    label = np.array([ans], np.int32)
    t = chainer.Variable(label, volatile=not tra)
    rec_loss = (loss_f0 + loss_e0 + loss_f1 + loss_e1)/4
    reo_loss = F.softmax_cross_entropy(y, t)
    predict = y.data.argmax(1)[0]
    if predict == ans:
        evaluate['correct_node'] += 1
        evaluate['correct_'+str(ans)] += 1
    evaluate['total'] += 1
    evaluate['total_'+str(ans)] += 1
    evaluate['pred_'+str(predict)] += 1
    return rec_loss, reo_loss


def cul_acc(r):
    r_list = []
    acc = 100.0 * r['correct_node'] / r['total']
    acc_m = 100.0 * r['correct_0'] / r['total_0']
    acc_s = 100.0 * r['correct_1'] / r['total_1']
    r_list.append('  Accuracy : {0:.2f}  %%  ({1:,d}/{2:,d})'.format(acc, r['correct_node'], r['total']))
    r_list.append(' m accuracy: {0:.2f}  %%  ({1:,d}/{2:,d})'.format(acc_m, r['correct_0'], r['total_0']))
    r_list.append(' s accuracy: {0:.2f}  %%  ({1:,d}/{2:,d})'.format(acc_s, r['correct_1'], r['total_1']))

    if n_label == 2:
        r_list.append('pred (m{0:,d} : s{1:,d})\n\n'.format(r['pred_0'], r['pred_1']))
    elif n_label == 3:
        acc_r = 100.0 * r['correct_2'] / r['total_2']
        r_list.append(' d accuracy: {0:.2f}  %%  ({1:,d}/{2:,d})'.format(acc_r, r['correct_2'], r['total_2']))
        r_list.append('pred (m{0:,d} : s{1:,d} : dr{2:,d})\n\n'.format(r['pred_0'], r['pred_1'], r['pred_2']))
    else:
        acc_r = 100.0 * r['correct_2'] / r['total_2']
        acc_l = 100.0 * r['correct_3'] / r['total_3']
        r_list.append('dr accuracy: {0:.2f}  %%  ({1:,d}/{2:,d})'.format(acc_r, r['correct_2'], r['total_2']))
        r_list.append('dl accuracy: {0:.2f}  %%  ({1:,d}/{2:,d})'.format(acc_l, r['correct_3'], r['total_3']))
        r_list.append('pred (m{0:,d} : s{1:,d} : dr{2:,d} : dl{3:,d})\n\n'.format(r['pred_0'], r['pred_1'], r['pred_2'], r['pred_3']))
    return r_list


def evaluate(test_trees):
    total_loss, total_rec_loss, total_reo_loss = 0, 0, 0
    result = collections.defaultdict(lambda: 0)
    result_list = []
    for tree in test_trees:
        rec_loss, reo_loss = train(tree, result, tra=False)
        loss = Alpha * rec_loss + (1 - Alpha) * reo_loss
        total_loss += float(loss.data)
        if type(rec_loss) != float:
            total_rec_loss += float(rec_loss.data)
        total_reo_loss += float(reo_loss.data)
    result_list.append('all loss: {:.4f}'.format(total_loss/len_test))
    result_list.append('rec loss: {:.4f}\nreo loss: {:.4f}'.format(
        total_rec_loss/len_test, total_reo_loss/len_test))
    result_list += cul_acc(result)
    return result_list


def init_model():
    #Make models
    if use_pre2 == 'pre': pre_unit = 4
    else: pre_unit = 0
    if use_null == 'null': null_unit = 6
    else: null_unit = 0
    if args.phrase == 'phrase':
        phrase_unit = 4
        model = chainer.FunctionSet(
            trainable=chainer.FunctionSet(
                w0=F.Linear(n_units * 2 + null_unit * 2, n_label),
                ww0=F.Linear(n_units * 2 + pre_unit + null_unit * 2 + phrase_unit, n_units + null_unit),
                ww1=F.Linear(n_units * 2 + pre_unit + null_unit * 2 + phrase_unit, n_units + null_unit),
            ),
            w1_f=F.Linear(n_units * 2+ null_unit * 2, n_units + null_unit), #source input
            w2_f=F.Linear(n_units + null_unit, n_units * 2 + null_unit * 2), #source output
            w1_e=F.Linear(n_units * 2 + null_unit * 2, n_units + null_unit), #target input
            w2_e=F.Linear(n_units + null_unit, n_units * 2 + null_unit * 2), #target output
            embed_f=F.EmbedID(vocab_f['len_vocab'], n_units), #source word embedding
            embed_e=F.EmbedID(vocab_e['len_vocab'], n_units), #target word embedding
        )
    else:
        model = chainer.FunctionSet(
            trainable=chainer.FunctionSet(
                w0=F.Linear(n_units * 4 + null_unit * 4, n_label),
            ),
            w1_f=F.Linear(n_units * 2+ null_unit * 2, n_units + null_unit), #source input
            w2_f=F.Linear(n_units + null_unit, n_units * 2 + null_unit * 2), #source output
            w1_e=F.Linear(n_units * 2 + null_unit * 2, n_units + null_unit), #target input
            w2_e=F.Linear(n_units + null_unit, n_units * 2 + null_unit * 2), #target output
            embed_f=F.EmbedID(vocab_f['len_vocab'], n_units), #source word embedding
            embed_e=F.EmbedID(vocab_e['len_vocab'], n_units), #target word embedding 
        )
    if opt_name == 'SGD':
        optimizer = optimizers.SGD(lr=0.02)  # (lr=opt_score)  # lr=0.01
    elif opt_name == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=0.001)  # (lr=opt_score)  # lr=0.001
    elif opt_name == 'AdaDelta':
        optimizer = optimizers.AdaDelta(rho=0.9)  # (rho=opt_score)  # rho=0.9
    elif opt_name == 'Adam':
        optimizer = optimizers.Adam(alpha=0.0001)  # (alpha=opt_score)  # alpha=0.0001
    optimizer.setup(model)  # .collect_parameters()
    return model, optimizer


def make_test_data():
    if w2v_No == 'w2v':
        w2v_matrix_f = word2vec.Word2Vec.load(w2v_f)
        w2v_matrix_e = word2vec.Word2Vec.load(w2v_e)
        for key_f, _ in vocab_f.items():
            if type(vocab_f[key_f]) == int and key_f in w2v_matrix_f.vocab:
                model.embed_f.W.data[vocab_f[key_f]] = w2v_matrix_f[key_f]
        print("load f word2vec")
        for key_e, _ in vocab_e.items():
            if type(vocab_e[key_e]) == int and key_e in w2v_matrix_e.vocab:
                model.embed_e.W.data[vocab_e[key_e]] = w2v_matrix_e[key_e]
        print("load e word2vec")
        test_trees  = read_corpus(args.test, vocab_f, vocab_e, f_count, e_count, (w2v_matrix_f, w2v_matrix_e))
        print("load test corpus")
        return test_trees
    else:
        return read_corpus(args.test, vocab_f, vocab_e, f_count, e_count, ['No'])

#Read corpus
vocab_f = {'len_vocab': 3}
vocab_e = {'len_vocab': 3}
f_count = collections.defaultdict(lambda: 0)
e_count = collections.defaultdict(lambda: 0)
counts(args.train, f_count, e_count)
train_trees = read_corpus(args.train, vocab_f, vocab_e, f_count, e_count, ['Yes'])
len_corpus  = len(train_trees)
with open(en_dict_file, 'wb') as f_en:
    pickle.dump(vocab_f, f_en)
with open(ja_dict_file, 'wb') as f_ja:
    pickle.dump(vocab_e, f_ja)

#Make models
model, optimizer = init_model()

#Read test data
test_trees = make_test_data()
len_test = len(test_trees)
#print(len_test)

#Writing Setting
f = open(log_file, "w")
f.write('date=%s\nfile_name=%s\nepoch=%d\nunit=%d\n' % (now_date, file_name, n_epoch, n_units))
f.write('batchsize=%d\nlabel=%d\ndrop_out=%f\nAlpha=%f\n' % (batchsize, n_label, drop_out, Alpha))
f.write('en_vocab=%d\nja_vocab=%d\nLen of train=%d\n' % (vocab_f['len_vocab'], vocab_e['len_vocab'], len_corpus))
f.write('%s\n%s\nw2v=%s\n%s\n' % (use_null, use_pre2, w2v_No, use_kana))
f.write('optimizer=%s\n' % (opt_name))
f.close()

start_at = time.time()
accum_loss, count = 0, 0
#pre_total_rec_loss = 1000
for epoch in range(n_epoch):
    result = collections.defaultdict(lambda: 0)
    total_loss, total_rec_loss, total_reo_loss = 0, 0, 0
    cur_at = time.time()
    np.random.shuffle(train_trees)
    for tree in train_trees:
        rec_loss, reo_loss = train(tree, result, tra=True)
        loss = Alpha * rec_loss + (1 - Alpha) * reo_loss
        accum_loss += loss
        if type(rec_loss) != float:
            total_rec_loss += float(rec_loss.data)
        total_reo_loss += float(reo_loss.data)
        count += 1

        if count >= batchsize:
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.weight_decay(0.0001)
            optimizer.update()
            total_loss += float(accum_loss.data)
            accum_loss, count = 0, 0

    #if pre_total_rec_loss - float(total_rec_loss.data) <= 0:
    #    optimizer.setup(model.trainable)
    #pre_total_rec_loss = float(total_rec_loss.data)
    now = time.time()
    throuput = float(len(train_trees)) / (now - cur_at)
    f = open(log_file, "a")
    f.write('------------- Epoch: {0:d} -------------\n'.format(epoch))
    f.write('{:.2f} iters/sec, {:.2f} sec\n'.format(throuput, now - cur_at))
    f.write('Train data evaluation:\n')
    f.write('all loss: {:.4f}\nrec loss: {:.4f}\nreo loss: {:.4f}\n'.format(
      total_loss/len_corpus, total_rec_loss/len_corpus, total_reo_loss/len_corpus))
    result_list = cul_acc(result)
    f.write('\n'.join(result_list))
    if (epoch + 1) % per_eval == 0:
        r_list = evaluate(test_trees)
        f.write( 'Test data evaluation:\n%s'  % ('\n'.join(r_list)))
    f.close()
    if (epoch + 1) % per_save == 0:
        pickle.dump(model, open('%s.%d' % (model_file, epoch+1), 'wb'), -1)
