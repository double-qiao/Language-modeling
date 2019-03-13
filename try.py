import re
import sys
from random import random
from math import log
from collections import defaultdict
#
# line = 'abcde fg.abc'
# list = [[abc],[bcd],[efg]]
# tri_counts = defaultdict(int)
#
# for i in range(len(list)):
#     for j in list[i]:
#         print(list[i][j])

word = input("Enter a word: ")
list = []
print("\nHere's each letter in your word:")
for letter in word:
    list.append(letter)
print(list)
list =