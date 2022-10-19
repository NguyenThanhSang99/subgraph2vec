import os
import random
import gensim
import json
from gensim.models.Annaword2vec import *
#from gensim.models.word2vec import *
from pprint import pprint


class sg2vec_sentences (object):
    def __init__(self, dirname, num_files, extn):
        self.dirname = dirname
        self.fnames = []
        if num_files:
            for fname in os.listdir(self.dirname)[:num_files]:
                if fname.endswith(extn):
                    self.fnames.append(os.path.join(self.dirname, fname))
        else:
            for fname in os.listdir(self.dirname):
                if fname.endswith(extn):
                    self.fnames.append(os.path.join(self.dirname, fname))

    def __iter__(self):
        for fname in self.fnames:
            for line in open(fname):
                yield line.split()


def shuffle(sents):
    random.shuffle(sents)
    return sents


def check_gensim_version():
    v = gensim.__version__
    if v != '0.12.4':
        exit(-1)


def main(debug=False, save_subgraph_vec_map=True):
    wlk_target_contexts_dir = "data/proteins"
    # WL kernel sentences of the format: <target subgraph> <context subgraphs 1> <context subgraphs 2> ...
    extn = 'WL1'
    opfname_prefix = "embedding"

    embedding_dim = 8
    iters = 20
    n_cpus = 4

    max_num_files = 20

    check_gensim_version()

    sentences = sg2vec_sentences(wlk_target_contexts_dir, max_num_files, extn)

    neg = 20  # num of negative samples for skipgram model -TUNE ACCORDING TO YOUR EXPERIMENTAL SETTING
    sg = 1

    model = Word2Vec(sentences,
                     min_count=1,
                     size=embedding_dim,
                     sg=sg,  # make sure this is ALWAYS 1, else cbow model will be used instead of skip gram
                     negative=neg,
                     workers=n_cpus,
                     iter=iters)

    if debug:
        vocab = model.vocab.keys()
        for w in vocab[:10]:
            pprint(model.most_similar(w))

    op_fname = opfname_prefix + '_' + '_'.join(['numfiles',
                                                str(max_num_files),
                                                'embeddingdim',
                                                str(embedding_dim),
                                                'numnegsamples',
                                                str(neg),
                                                'numiteration',
                                                str(iters)])
    model.save(os.path.join('../models', op_fname))

    if save_subgraph_vec_map:
        subgraph_vec_map = {
            subgraph: model[subgraph].tolist() for subgraph in model.vocab}
        op_fname = op_fname + '_map.json'
        with open(op_fname, 'w') as fh:
            json.dump(obj=subgraph_vec_map, fp=fh, indent=4)


if __name__ == '__main__':
    main(debug=False)
