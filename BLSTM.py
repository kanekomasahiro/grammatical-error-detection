from EWE import SSWE_model
import gensim
import sys
from functions import fill_batch, make_dict, take_len
from gensim.models import word2vec
import numpy as np
import collections
import chainer.links as L
from chainer import functions as F
from chainer import optimizers as O
from chainer import Chain, Variable , cuda, serializers
import pickle
import generators as gens
import random
import time

train_txt = "train.txt"
test_txt = "dev.txt"
vocab_dict = "model/BLSTMVocab.pkl"
load_model = "model/BLSTM.model0"
ana = "model/ana.txt"
state_model = "model/BLSTM.sta"

embed_type = False
if embed_type:
    state_embed = "SSWEmodel/sswe.sta"
    embedVocab = 'SSWEmodel/ssweVoc.pkl'
    embedModel = 'SSWEmodel/sswe.model14'
    embed_vocab_dict = 'SSWEmodel/ssweVocabDict.pkl'


vocab_size = take_len(train_txt)
batch_size = 64
embed_size = 300
output_size = 2
hidden_size = 200
extra_hidden_size = 50
epoch = 2
gpu = -1


xp = cuda.cupy if gpu >= 0 else np
if gpu >= 0:
    cuda.check_cuda_available()

random.seed(0)
xp.random.seed(0)

def precision_recall_f(pres, tags, cons):
    c_p = 0
    correct_p = 0
    c_r = 0
    correct_r = 0
    tags = Variable(xp.array(tags, dtype=xp.int32).T)
    for num, a in enumerate(zip(pres, tags)):
            pre_l = [int(xp.argmax(a[0].data[k])) for k in range(len(a[0].data)) if cons[num][k] == True]
            tag_l = [int(a[1].data[n]) for n in range(len(a[1].data)) if cons[num][n] == True]
            for a, b in zip(tag_l, pre_l):
                if a == 1:
                    c_r += 1
                    if b == a:
                        correct_r += 1
                if b == 1:
                    c_p += 1
                    if b == a:
                        correct_p += 1
    return c_p, correct_p, c_r, correct_r

def evaluate(model, word2id):
    c_p = 0
    correct_p = 0
    c_r = 0
    correct_r = 0
    m = model.copy()
    m.volatile = True
    gen1 = gens.word_list(test_txt)
    gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
    batchs = [b for b in gen2]
    for batch in batchs:
        tag0 = batch[:]
        tags = [a[:-1]  for a in tag0]
        batch = [b[1:] for b in batch]
        batch = fill_batch([b[-1].split() for b in batch])
        tags = fill_batch(tags, token=-1)
        pres, cons = forward(batch, tags, m, word2id, mode = False)
        a, b, c, d =  precision_recall_f(pres, tags, cons)
        c_p += a
        correct_p += b
        c_r += c
        correct_r += d
    try:
        precision = correct_p/c_p
        recall = correct_r/c_r
        f_measure = (1 + 0.5**2)*precision*recall/(0.5**2*precision + recall)
    except:
        precision = 'nothing'
        recall = 'nothing'
        f_measure = 'nothing'
    print('Precision:\t{}'.format(precision))
    print('Recall:\t{}'.format(recall))
    print('F-value\t{}'.format(f_measure))

class BLSTMw2v(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super().__init__(
            x2e = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            e2h_for = L.LSTM(embed_size, hidden_size),
            e2h_back = L.LSTM(embed_size, hidden_size),
            h2s = L.Linear(hidden_size*2, extra_hidden_size),
            s2o = L.Linear(extra_hidden_size, output_size)
        )

    def __call__(self, x):
        self._reset_state()
        e_states = []
        h_back_states = []
        h_states = []
        o_states = []

        e_states = [F.relu(self.x2e(w)) for w in x]
        for e in e_states[::-1]:
            h_back = self.e2h_back(e)
            h_back_states.insert(0, h_back)
        for e, h_back in zip(e_states, h_back_states):
            h_for = self.e2h_for(e)
            h_states.append(F.concat((h_for, h_back)))
        o_states = [self.s2o(F.relu(self.h2s(h))) for h in h_states]
        return o_states

    def _reset_state(self):
        self.zerograds()
        self.e2h_for.reset_state()
        self.e2h_back.reset_state()

    def initialize_embed(self, word2vec_model, word_list, word2id, id2word):
        if embed_type:
            for i in range(len(word2vec_model.x2e.W.data)):
                self.x2e.W.data[i] = word2vec_model.x2e.W.data[i]
        else:
            for i in range(len(word_list)):
                word = word_list[i]
                if word in word2vec_model:
                    self.x2e.W.data[i+2] = word2vec_model[word]

def forward(batchs, tags, model, word2id, mode):

    if mode:
        accum_loss = Variable(xp.zeros((),dtype=xp.float32))
        x = Variable(xp.array([[word2id[word] if word in word2id else word2id['<unk>'] for word in sen] for sen in batchs], dtype=xp.int32).T)
        pres = model(x)
        tags = Variable(xp.array(tags, dtype=xp.int32).T)
        for tag, pre in zip(tags, pres):
            accum_loss += F.softmax_cross_entropy(pre, tag)
        sortmax_pres = [F.softmax(pre) for pre in pres]
        condition = xp.array(x.data!=-1)
        return accum_loss, sortmax_pres, condition

    else:
        x = Variable(xp.array([[word2id[word] if word in word2id else word2id["<unk>"] for word in sen] for sen in batchs], dtype=xp.int32).T)
        pres = model(x)
        sortmax_pres = [F.softmax(pre) for pre in pres]
        condition = xp.array(x.data!=-1)
        return sortmax_pres, condition


def train():
    id2word = {}
    word2id = {}
    word_freq = collections.defaultdict(lambda: 0)
    id2word[0] = "<unk>"
    word2id["<unk>"] = 0
    id2word[1] = "<s>"
    word2id["<s>"] = 1
    id2word[-1] = "</s>"
    word2id["</s>"] = -1
    if embed_type:
        word2id = pickle.load(open(embed_vocab_dict, 'rb'))
        id2word = {}
        for key, value in word2id.items():
            id2word[value] = key
        word_list = None
        sta = pickle.load(open(state_embed, "rb"))
        word2vec_model = SSWE_model(sta["vocab_size"], sta["embed_size"], sta["hidden_size"])
    else:
        word2id, id2word, word_list, word_freq = make_dict(train_txt, word2id, id2word, word_freq)
        word2vec_model = gensim.models.Word2Vec.load_word2vec_format('embedding.txt')
    model = BLSTMw2v(vocab_size, embed_size, hidden_size, output_size)
    model.initialize_embed(word2vec_model, word_list, word2id, id2word)
    if gpu >= 0:
        cuda.get_device(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    opt = O.Adam(alpha=0.001)
    opt.setup(model)

    for i in range(epoch):
        print("epoch{}".format(i+1))
        start = time.time()
        total_loss = 0
        gen1 = gens.word_list(train_txt)
        gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
        batchs = [b for b in gen2]
        bl = list(range(len(batchs)))
        random.shuffle(bl)
        for n, j in enumerate(bl):
            tag0 = batchs[j][:]
            tags = [[int(c) for c in a[:-1]] for a in tag0]
            batch = fill_batch([b[-1].split() for b in batchs[j]])
            tags = fill_batch(tags, token = 0)
            accum_loss, pres, cons = forward(batch, tags, model, word2id, mode = True)
            accum_loss.backward()
            opt.update()
            total_loss += accum_loss.data
        print("total_loss {}".format(total_loss))
        serializers.save_npz("{}{}".format(load_model, i), model)
        evaluate(model, word2id)
        ff = open('embeding.txt','w')
        ff.write('{} {}\n'.format(len(model.x2e.W.data)-3, embed_size))
        for num in range(2, len(model.x2e.W.data)-1):
            ff.write('{} {}\n'.format(id2word[num], ' '.join([str(model.x2e.W.data[num][numnum]) for numnum in range(embed_size)])))
        print("time: {}".format(time.time() - start))
    pickle.dump(dict(word2id), open(vocab_dict, 'wb'))
    serializers.save_npz(load_model, model)
    state_d = {}
    state_d["vocab_size"] = vocab_size
    state_d["hidden_size"] = hidden_size
    state_d["embed_size"] = embed_size
    pickle.dump(state_d, open(state_model, "wb"))

def test():
    sta = pickle.load(open(state_model, "rb"))
    word2id = pickle.load(open(vocab_dict, 'rb'))
    c_p = 0
    correct_p = 0
    c_r = 0
    correct_r = 0
    predicts = []
    total_predicts = []
    total_tags = []
    total_batchs = []
    model = BLSTMw2v(sta["vocab_size"], sta["embed_size"], sta["hidden_size"], output_size)
    serializers.load_npz(load_model, model)
    if gpu >= 0:
        model.to_gpu()
    for batchs in gens.batch(gens.word_list(test_txt), batch_size):
        tag0 = batchs[:]
        tags = [a[:-1]  for a in tag0]
        batchs = [b[1:] for b in batchs]
        total_batchs.append(batchs)
        batchs = fill_batch([b[-1].split() for b in batchs])
        tags = fill_batch(tags, token=-1)
        pres, cons = forward(batchs, tags, model, word2id, mode = False)
        for num, a in enumerate(zip(pres, tags)):
            pre_l = [int(xp.argmax(a[0].data[k])) for k in range(len(a[0].data)) if cons[num][k] == True]
        a, b, c, d =  precision_recall_f(pres, tags, cons)
        c_p += a
        correct_p += b
        c_r += c
        correct_r += d
    precision = correct_p/c_p
    recall = correct_r/c_r
    f_measure = 2*precision*recall/(precision + recall)
    print('Precision:\t{}'.format(precision))
    print('Recall:\t{}'.format(recall))
    print('F-value\t{}'.format(f_measure))


def main():
    if len(sys.argv) != 2:
        print('No arguments!!')
        exit()
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        print('Miss type!!')

if __name__ == '__main__':
    main()
