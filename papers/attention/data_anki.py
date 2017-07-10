# uses data from http://www.manythings.org/anki/
from __future__ import print_function
import os
from os import path
from os.path import join
import subprocess
import zipfile
import platform
import numpy as np


anki_dir = join('data', 'anki')


class Data(object):
    def __init__(self, pair_name='fra-eng'  , seed=123, train_test_split=0.8):
        self.pair_name = pair_name
        self.seed = seed
        self.train_test_split = train_test_split
        self.maybe_fetch_data(pair_name)
        self.split_train_test()

    @classmethod
    def wget(cls, url, out_file):
        if platform.uname()[0] == 'Darwin':
            print(subprocess.check_output(['curl', url, '-o', out_file]))
        else:
            print(subprocess.check_output(['wget', url, '-O', out_file]))

    def maybe_fetch_data(self, pair_name='fra-eng'):
        if not path.isdir(anki_dir):
            os.makedirs(anki_dir)
        if not path.isfile(join(anki_dir, '%s.zip' % pair_name)):
            self.wget('http://www.manythings.org/anki/%s.zip' % pair_name,
                join(anki_dir, '%s.zip' % pair_name))
        if not path.isfile(join(anki_dir, '%s.txt' % pair_name)):
            f_zip = zipfile.ZipFile(join(anki_dir, '%s.zip' % pair_name), 'r')
            member_name = None
            for member in f_zip.infolist():
                if not member.filename.startswith('_'):
                    member_name = member.filename
                    break
            with f_zip.open(member_name, 'r') as f_txt:
                contents = f_txt.read()
            f_zip.close()
            with open(join(anki_dir, '%s.txt' % pair_name), 'w') as f:
                if not isinstance(contents, str):
                    contents = contents.decode('utf-8')
                f.write(contents)
        assert path.isfile(join(anki_dir, '%s.txt' % pair_name))
        with open(join(anki_dir, '%s.txt' % pair_name)) as f:
            contents = f.read()
            assert len(contents) > 200
            # print(contents[:200])

    def split_train_test(self):
        all_examples = []
        with open(join(anki_dir, '%s.txt' % self.pair_name)) as f:
            for line in f:
                line = line.strip()
                if line == '':
                    contintue
                split_line = line.split('\t')
                all_examples.append({'first': split_line[0], 'second': split_line[1]})

        self.N_total = len(all_examples)
        self.N_train = int(self.N_total * self.train_test_split)
        self.N_test = self.N_total - self.N_train
        r = np.random.RandomState(self.seed)
        self.shuffled_indices = r.choice(self.N_total, self.N_total, replace=False)
        self.train_indices = self.shuffled_indices[:self.N_train]
        self.test_indices = self.shuffled_indices[self.N_train:]
        self.train = [all_examples[i] for i in self.train_indices]
        self.test = [all_examples[i] for i in self.test_indices]

    def get_training(self, pair_name='fra-eng', N=0):
        # maybe_fetch_data(pair_name=pair_name)
        if N == 0:
            return self.train
        else:
            return self.train[:N]

    def get_test(self, pair_name='fra-eng', N=0):
        # maybe_fetch_data(pair_name=pair_name)
        if N == 0:
            return self.test
        else:
            return self.test[:N]
