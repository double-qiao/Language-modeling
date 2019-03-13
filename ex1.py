import re
import sys
from random import random
from math import log
from collections import defaultdict
from itertools import permutations
import string



class languagemodel:

    def __init__(self,training_corpus_dir,out_file_path,alpha):
      self.training_corpus_dir = training_corpus_dir
      self.out_file_path = out_file_path
      self.Pro_counts = defaultdict(float)
      self.Pro_counts_new = defaultdict(float)
      self.tri_counts = defaultdict(int)
      self.bi_counts = defaultdict(int)
      self.trigram = []
      self.content = []
      self.STARTSYMBOL = '##'
      self.STOPSYMBOL = '#'
      self.vocab_size = 30
      self.alpha = alpha

#generate all possible brigram and trigram combinations
    def keys_gen(self):

        tri_keys = list(map(lambda x: ''.join(x), permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 3)))
        tri_keys1 = []
        tri_keys2 = []
        tri_keys3 = []
        for ch in (string.ascii_lowercase + '0' + ' ' + '#'+'.'):
            tri_keys1 += list(
                map(lambda x: ch + x[0] + ch, permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 1)))
        for ch in (string.ascii_lowercase + '0' + '#' + '.'+'  '):
            tri_keys2 += list(
                map(lambda x: ch + ch + x[0], permutations(string.ascii_lowercase + '0' + ' ' + '.'+'#', 1)))
        for ch in (string.ascii_lowercase + '0'+'.'+' '):
            tri_keys3 += list(
                map(lambda x: x[0] + ch + ch, permutations(string.ascii_lowercase + '0' + ' ' + '.' + '#', 1)))

        # print(len(tri_keys1))
        # print(len(tri_keys2))

        tri_keys = tri_keys + tri_keys1 + tri_keys2 + tri_keys3
        # print(len(tri_keys))

        values_tri = [0 for i in range(len(tri_keys))]
        self.tri_counts = dict(zip(tri_keys, values_tri))

        # print(len(tri_counts))

        bi_keys = list(
            map(lambda x:''.join(x), permutations(string.ascii_lowercase+'0'+' '+'.'+'#', 2)))
        bi_keys1 = []
        for ch in (string.ascii_lowercase + '0' + '#'+'.'+' '):
            bi_keys1.append(ch+ch)

        bi_keys = bi_keys+bi_keys1
        bi_values = [0 for i in range(len(bi_keys))]

        self.bi_counts = dict(zip(bi_keys, bi_values))

        return self.tri_counts, self.bi_counts

    def preprocess_line(self, line):

      line = re.sub('[^A-Za-z0-9 .]', "", line)
      line = re.sub('[0-9]','0',line)
      line = line.lower()
      line = self.STARTSYMBOL+line+self.STOPSYMBOL

      return line

    def get_Ngram(self, line):
        for i in range(len(line) - (2)):
            self.tri_counts[line[i:i+3]] += 1
            self.bi_counts[line[i:i+2]] +=1
        #
        # for i in range(len(line) - (1)):
        #     self.bi_counts[line[i:i+2]] += 1
        # print(tri_counts)
        return self.tri_counts, self.bi_counts

    def get_Prob(self):

      self.Pro_counts = self.tri_counts
      self.trigram = self.tri_counts.keys()

      for i in range(len(self.trigram)):
         list = []
         for ch in self.trigram[i]:
            list.append(ch)
         # print(list)
         bi_str = list[0]+list[1]
         # print(bri_str)
         # print(trigram[i])
         # print(tri_counts[trigram[i]])
         # print(bri_counts[bri_str])
         # if self.bi_counts[bi_str] ==0:
         #     self.Pro_counts[self.trigram[i]] = 0
         # else:
         self.Pro_counts[self.trigram[i]] = float(self.tri_counts[self.trigram[i]] + self.alpha) / (self.bi_counts[bi_str] + self.alpha * self.vocab_size)



      self.Pro_counts_new = sorted(self.Pro_counts.items(), key=lambda x: x[0], reverse=0)

      # sum=0.0
      # for values in self.Pro_counts.values():
      #     sum += values
      print(self.Pro_counts)
         # print(Pro_counts[trigram[i]])
      # print(self.Pro_counts_new)
      return self.Pro_counts_new

    def dict_to_text(self):
      for i in range(len(self.Pro_counts_new)):
         self.content.append(self.Pro_counts_new[i][0] +'\t'+str('{:.3e}'.format(self.Pro_counts_new[i][1]))+'\n')
      return self.content

    def save_to_file(self):
      file = open(self.out_file_path, 'w')
      for i in range(len(self.content)):
         file.write(self.content[i])
      file.close()




if __name__ == "__main__":

   Inputfile = "training.de"
   Outputfile = "language_model.de"


   lm = languagemodel(Inputfile, Outputfile,alpha=0.1)
   tri_counts, bi_counts = lm.keys_gen()
   # file = open(lm.out_file_path, 'w')

   with open(Inputfile) as f:
       for line in f:
           line = lm.preprocess_line(line)
           # file.write(line)
           lm.get_Ngram(line)

   lm.get_Prob()
   lm.dict_to_text()
   lm.save_to_file()
   

# line = 'SiWei twdhnefjw.c%$ h12. .359ABead   \mjoijo 9 '
# lines = preprocess_line(line)
# print (lines)
