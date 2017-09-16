from linereader import copen
import random

random.seed(0)

def batch(generator, batch_size):
    batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

def sorted_parallel(generator1, pooling, order=1):
    gen1 = batch(generator1, pooling)
    for batch1 in gen1:
        for x in sorted(batch1, key=lambda x: len(x)):
            yield x

def word_list(filename):
    f = copen(filename)
    lines = f.count("\n")
    bl = list(range(lines))
    random.shuffle(bl)
    for i in bl:
        l = f.getline(i)
        yield l.split('\t')

