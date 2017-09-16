import gensim
import generators as gens
from functions import make_dict, fill_batch, take_len, make_spell_miss_list
import numpy as np
import collections
from chainer import *
from chainer import functions as F
from chainer import optimizers as O
from chainer import links as L
import pickle
import random
import math
import time

train_txt = "train.txt"
ssweV = 'EWEmodel/eweVoc.pkl'
ssweM = 'EWEmodel/ewe.model'
state_model = "EWEmodel/ewe.sta"
vocab_dict = 'EWEmodel/eweVocabDict.pkl'

vocab_size = take_len(train_txt)
embed_size = 300
output_size = 2
hidden_size = 200
epoch = 13
window_size = 3
batch_size = 1
gpu = -1
r = 0.01
noise_size = 10



if gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu >= 0 else np

random.seed(0)
xp.random.seed(0)

id2word = {}
word2id = {}
word_freq = collections.defaultdict(lambda: len(word2id))
word2id["<unk>"] = 0 
word2id["</s>"] = 1
word2id["EOS"] = -1
id2word[0] = '<unk>'
id2word[1] = '</s>'
id2word[-1] = 'EOS'

uni_dict = pickle.load(open('source_target.pkl','rb'))
word2id, id2word, word_list, word_freq = make_dict(train_txt, word2id, id2word, word_freq)

pickle.dump(word2id, open(vocab_dict, 'wb'))

def norm(data):
    return xp.sqrt(xp.sum(data**2))

def cos_sim(v1, v2):
    return xp.dot(v1, v2) / (norm(v1) * norm(v2))


def evaluate(model):
    cos_sum = 0
    m = model.copy()
    m.volatile = True
    s_l = make_spell_miss_list()
    count = 0
    for word in s_l:
        if word[0] in word2id and word[1] in word2id:
            count += 1
            cos_sum += cos_sim(m.x2e.W.data[word2id[word[0]]], m.x2e.W.data[word2id[word[1]]])

def convert_word(sentence):
    return [word2id[word] for word in sentence]

def make_sequences(sentences, tag):
    csequences = xp.array([sentences[i:i+window_size] for i in range(len(sentences)-window_size+1)], dtype=xp.int32)
    nls = list(csequences[:,1:2])
    noise = []
    for nl in nls:
        if int(*nl) in id2word and id2word[int(*nl)] in uni_dict:
            d = id2word[int(*nl)]
            ns = uni_dict[d]
            noise.append([[word2id[n[0]]] if n[0] in word2id else [word2id['<unk>']] for i,n in enumerate(sorted(ns.items(), key=lambda x:x[1], reverse=True)) if i < noise_size])
            for no in noise:
                while len(no) < noise_size:
                    no.append([random.randint(2, vocab_size)])
        else:
            noise.append(xp.random.randint(2,vocab_size, size=(noise_size, 1)))
    noise = xp.array(noise, dtype=xp.int32)
    sequences = xp.array([sentences[i:i+window_size] for i in range(len(sentences)-window_size+1)], dtype=xp.int32)
    zeros = xp.zeros((len(csequences),noise_size, window_size), dtype=xp.int32)
    csequences = xp.reshape(csequences, (len(csequences),1,window_size))
    csequences = csequences + zeros
    noise = xp.random.randint(vocab_size, size=(len(csequences),noise_size,1))
    csequences[:,:,1:2] = noise
    tags = xp.array([tag[i:i+window_size] for i in range(len(tag)-window_size+1)], dtype=xp.int32)
    tags = xp.sum(tags, axis=1)
    tags = xp.where(tags>0, xp.ones(len(tags), dtype=xp.int32), xp.zeros(len(tags), dtype=xp.int32))
    return sequences, csequences, tags

class SSWE_model(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, train=True):
        super(SSWE_model, self).__init__(
            x2e = L.EmbedID(vocab_size, embed_size),
            e2h = L.Linear(embed_size*window_size, hidden_size),
            h2h = L.Linear(hidden_size, hidden_size),
            h2o = L.Linear(hidden_size, 1),
            )

    def __call__(self, sequence):
        self._reset_state()
        e = F.tanh(self.x2e(sequence))
        h = F.tanh(self.e2h(e))
        o = self.h2o(h)
        return o

    def _reset_state(self):
        self.zerograds()

    def initialize_embed(self, word2vec_model, word_list, word2id):
        for i in range(len(word_list)):
            word = word_list[i]
            if word in word2vec_model:
                self.x2e.W.data[i+2] = word2vec_model[word]

def forward(x, tags, model):
    accum_loss = Variable(xp.zeros((), dtype=xp.float32))
    x, c, t = make_sequences(x, tags)
    x = Variable(x)
    leng = len(c)
    c = Variable(xp.reshape(c, (leng*noise_size, window_size)))
    t = Variable(t)
    o = model(x)
    co = model(c)
    co = F.reshape(co, (leng,noise_size,1))
    o = F.broadcast_to( F.reshape(o, (leng,1,1)), (leng, noise_size, 1))
    loss = F.sum(F.clipped_relu(1-co+o, 1e999))
    return loss

def train(model, opt):
    for i in range(epoch):
        print("epoch {}".format(i))
        start = time.time()
        total_loss = 0
        gen1 = gens.word_list(train_txt)
        gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
        batchs = [ b for b in gen2]
        bl = list(range(len(batchs)))
        random.shuffle(bl)
        for n, j in enumerate(bl):
            if window_size > len(batchs[j][0])-4:
                continue
            tag0 = batchs[j][:]
            tags = [[int(c) for c in a[:-1]] for a in tag0]
            batch = fill_batch([b[-1].split() for b in batchs[j]])
            batch = [convert_word(b)  for b in batch]
            tags = xp.array(tags, dtype=xp.int32)
            accum_loss = forward(*batch, *tags, model)
            accum_loss.backward()
            opt.update()
            total_loss += accum_loss.data
        print(total_loss)
        evaluate(model)
        serializers.save_npz("{}{}".format(ssweM, i), model)
        print("time: {}".format(time.time() - start))

def main():
    model = SSWE_model(vocab_size, embed_size, hidden_size)
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    model.initialize_embed(word2vec_model, word_list, word2id)
    if gpu >= 0:
        cuda.get_device(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    model.x2e.W.data[0] = xp.zeros(embed_size, dtype=xp.float32)
    opt = O.Adam(alpha = 0.001)
    opt.setup(model)
    train(model, opt)
    with open(ssweV, 'wb') as f:
        pickle.dump(dict(word2id), f)
    state_d = {}
    state_d["vocab_size"] = vocab_size
    state_d["hidden_size"] = hidden_size
    state_d["embed_size"] = embed_size
    pickle.dump(state_d, open(state_model, "wb"))

if __name__ == '__main__':
    main()
