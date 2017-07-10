import numpy as np


source_words = 'this is a test of some foo bar paris london whatever near far'.split(' ')
num_available_words = len(source_words)


def get_examples(seed=123, N=100):
    r = np.random.RandomState(123)
    examples = []
    for n in range(N):
        num_words = r.choice(range(3,8))
        chosen_word_idxes = r.choice(num_available_words, num_words, replace=True)
        chosen_words = []
        for i in chosen_word_idxes:
            chosen_words.append(source_words[i])
        num_starred = r.choice(range(1, num_words + 1))
        starred = set(r.choice(num_words, num_starred, replace=False))
        for i in starred:
            chosen_words[i] = '*' + chosen_words[i]
        sentence = ' '.join(chosen_words)
        target_words = []
        for i in starred:
            target_words.append(chosen_words[i][1:])
        target_sentence = ' '.join(target_words)
        examples.append({'input': sentence, 'target': target_sentence})
    return examples

def get_training(N=100):
    return get_examples(seed=123, N=N)


def get_test(N=100):
    return get_examples(seed=124, N=N)
