import re
import numpy as np
from math import log
from collections import defaultdict
from itertools import permutations
import string


class languagemodel:

    def __init__(self, training_corpus_dir, out_file_path):
        self.training_corpus_dir = training_corpus_dir
        self.out_file_path = out_file_path
        self.Probkn = defaultdict(float)  # the probability of Kneser-Ney method
        self.Probadd = defaultdict(float)  # the probability of addk method which is used to compare the performance
        #  of Knerse-Ney method

        self.cw0w2 = defaultdict(float)  # count of w0w1w2
        self.cw1w2 = defaultdict(float)  # count of w1w2
        self.cw0w1 = defaultdict(float)  # count of w0w1
        self.cw1 = defaultdict(float)  # count of w1
        # self.cw2 = defaultdict(float)  # count of w2

        self.content = []
        self.STARTSYMBOL = '##'
        self.STOPSYMBOL = '#'
        self.d = 0.75
        self.alpha = 1
        self.vocab_size = 30

    # generate all possible brigram and trigram combinations
    def keys_gen(self):
        tri_keys = list(map(lambda x: ''.join(x), permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 3)))
        tri_keys1 = []
        tri_keys2 = []
        tri_keys3 = []
        for ch in (string.ascii_lowercase + '0' + ' ' + '#' + '.'):
            tri_keys1 += list(
                map(lambda x: ch + x[0] + ch, permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 1)))
        for ch in (string.ascii_lowercase + '0' + ' ' + '#' + '.'):
            tri_keys2 += list(
                map(lambda x: ch + ch + x[0], permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 1)))
        for ch in (string.ascii_lowercase + '0' + ' ' + '#' + '.'):
            tri_keys3 += list(
                map(lambda x: x[0] + ch + ch, permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 1)))

        tri_keys = tri_keys + tri_keys1 + tri_keys2 + tri_keys3
        values_tri = [0 for i in range(len(tri_keys))]
        self.cw0w2 = dict(zip(tri_keys, values_tri))

        bi_keys = list(
            map(lambda x: ''.join(x), permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 2)))
        bi_keys1 = []
        for ch in (string.ascii_lowercase + '0' + '#' + '.' + ' '):
            bi_keys1.append(ch + ch)

        bi_keys = bi_keys + bi_keys1
        bi_values = [0 for i in range(len(bi_keys))]

        self.cw0w1 = dict(zip(bi_keys, bi_values))
        self.cw1w2 = dict(zip(bi_keys, bi_values))

        uni_keys = list(i for i in (string.ascii_lowercase + '0' + ' ' + '.' + '#'))
        uni_values = [0 for i in range(len(uni_keys))]
        self.cw1 = dict(zip(uni_keys, uni_values))
        # self.cw2 = dict(zip(uni_keys, uni_values))

    # preprocess input line,
    def preprocess_line(self, line):
        # remove all characters which are not English alphabet, space, digits, ‘.', '?', '!’
        line = re.sub('[^A-Za-z0-9 .?!]', '', line)
        # transform all digits to '0'
        line = re.sub('[0-9]', '0', line)
        # transform all '?'or'!' to '.'
        line = re.sub('[?!]', '.', line)
        # transform all double space'  ' to space
        line = re.sub('  ', ' ', line)
        # transform all space dot' .' to '.'
        line = re.sub(' \.', '.', line)
        # lowercase all remaining characters
        line = line.lower()
        # lowercase all remaining characters
        line = self.STARTSYMBOL + line + self.STOPSYMBOL
        return line

    # count the number of occurrences of each trigram, bigram or unigram
    def get_Ngram(self, line):
        for i in range(len(line) - 2):
            self.cw0w2[line[i:i + 3]] += 1
            self.cw0w1[line[i:i + 2]] += 1
            self.cw1w2[line[i + 1:i + 3]] += 1
            self.cw1[line[i + 1]] += 1
            # self.cw2[line[i+2]] += 1


    # get the probability based on Kneser-Ney smoothing
    def get_Prob_KN(self):

        # compute the value of λ(w0w1)
        lmbdw0w1 = defaultdict(float)
        for w0w1 in list(self.cw0w1.keys()):
            w2 = []
            for ch in string.ascii_lowercase + '0' + ' ' + '.' + '#':
                if self.cw0w2[w0w1 + ch] > 0:
                    w2.append(ch)
            if self.cw0w1[w0w1] == 0:
                lmbdw0w1[w0w1] = 0
                continue
            lmbdw0w1[w0w1] = self.d / self.cw0w1[w0w1] * len(w2)

        # compute P(w2|w1) for bigram by Knerse-Ney method
        pknw1w2 = defaultdict(float)
        for w1w2 in list(self.cw1w2.keys()):
            if self.cw1[w1w2[0]] == 0:
                pknw1w2[w1w2] = 0 # + lmbdw1[w1w2[0]] * self.cw2[w1w2[1]] / sumcw2
                continue
            pknw1w2[w1w2] = self.cw1w2[w1w2] / self.cw1[w1w2[0]] # + lmbdw1[w1w2[0]] * self.cw2[w1w2[1]] / sumcw2

        # compute the P(w2|w0w1) for trigram by Knerse-Ney method
        pknw0w2 = defaultdict(float)
        for w0w2 in list(self.cw0w2.keys()):
            if self.cw0w1[w0w2[0:2]] == 0:
                pknw0w2[w0w2] = 0 + lmbdw0w1[w0w2[0:2]] * pknw1w2[w0w2[1:3]]
                continue
            pknw0w2[w0w2] = max(0.0, self.cw0w2[w0w2] - self.d) / self.cw0w1[w0w2[0:2]] + lmbdw0w1[w0w2[0:2]] * pknw1w2[
                w0w2[1:3]]

        self.Probkn = sorted(pknw0w2.items(), key=lambda x: x[0], reverse=0)
        print(self.Probkn)

        return self.Probkn

    # transform dictionary to a readable text
    def dict_to_text(self):
        for i in range(len(self.Probkn)):
            self.content.append(self.Probkn[i][0] + '\t' + str('{:.3e}'.format(self.Probkn[i][1])) + '\n')
        return self.content

    # save the language model
    def save_to_file(self):
        file = open(self.out_file_path, 'w')
        for i in range(len(self.content)):
            file.write(self.content[i])
        file.close()

    # randomly generate sequence from specific language model
    def generate_from_lm(self, language_model, length):
        head = ['#', '#']  # let start two characters be ##
        ss = ''
        lmdict = {}
        # read language model file
        with open(language_model) as f:
            for line in f:
                item1 = line.strip('\n').split('\t')
                lmdict[item1[0]] = float(item1[1])
        for i in range(length - 2):
            keys = []
            values = []
            stop = 0
            # get specific bigram keys and values in language model
            for key in lmdict.keys():
                if head[0] == key[0] and head[1] == key[1]:
                    keys.append(key)
                    values.append(lmdict[key])
                    stop = 1
                    continue
                if stop == 1:
                    break

            outcomes = np.array(keys)
            probs = np.array(values)
            # make an array with the cumulative sum of probabilities at each index
            bins = np.cumsum(probs)
            # if there is no element in bins are all elements equal 0, just go to a new paragraph
            if bins.size == 0 or np.sum(bins) == 0:
                head[0] = '#'
                head[1] = '#'
                ss += '\n'
                continue
            np.random.seed(1545 + i)
            # create a random float number from 0-1, then use digitize to get index
            # the digitize function tells us which bin the random number fall into.
            # use the index to find the corresponding trigram, and the last character is what we want to generate
            generate = outcomes[np.digitize(np.random.rand(), bins)]
            # if generated character is #, it means should start a new paragraph
            if generate[2] == '#':
                head[0] = '#'
                head[1] = '#'
                ss += '\n'
            else:
                ss += generate[2]
                head[0] = head[1]
                head[1] = generate[2]
        return ss

    # compute test files' perplexity based on specific language model
    def compute_perplexity(self, testfile, language_model):
        prob = 0.0
        lmdict = {}
        count = 0

        # read in language model as dictionary
        with open(language_model) as f:
            for line in f:
                item1 = line.strip('\n').split('\t')
                lmdict[item1[0]] = float(item1[1])

        with open(testfile) as f:
            data = f.readlines()

        for line in data:
            line = self.preprocess_line(line)
            ss = '###'  # give the initial ## to calculate the first character in line
            for i in range(2, len(line)):  # start from the first non # character
                count += 1
                ss = ss[1:] + line[i]
                # if the trigram not in language model or probability is equal to 0,
                # just let the probability equals 1/30
                if ss not in lmdict or lmdict[ss] == 0:
                    prob = prob + log(1.0 / 30, 2)
                else:
                    prob = prob + log(lmdict[ss], 2)
        return 2 ** (-1 / count * prob)


if __name__ == "__main__":

    Inputfile = "training.en"
    Outputfile = "language_model.en"

    # initial class
    lm = languagemodel(Inputfile, Outputfile)


    # generate all possible brigram and trigram combinations
    lm.keys_gen()


    # count the number of occurrences of each trigram, bigram or unigram
    with open(Inputfile) as f:
        for line in f:
            line = lm.preprocess_line(line)
            lm.get_Ngram(line)

    # generate model based on Kneser-Ney smoothing
    lm.get_Prob_KN()

    # save the model generated
    lm.dict_to_text()
    lm.save_to_file()

    print("Perplexity of our English language model:", lm.compute_perplexity("test", Outputfile))
    print("Perplexity of our English language new model:", lm.compute_perplexity("newtest.txt", Outputfile))
    print("Perplexity of our English language gen model:", lm.compute_perplexity("gentest", Outputfile))
    # print("Perplexity of model-br.en:", lm.compute_perplexity("test", "model-br.en"))
    # print("The sequence generated by our English language model:")
    # print(lm.generate_from_lm(Outputfile, 300) + "\n")
    # print("The sequence generated by model-br.en:")
    # print(lm.generate_from_lm("model-br.en", 300))
