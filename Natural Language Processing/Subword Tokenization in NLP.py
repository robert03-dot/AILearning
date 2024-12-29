"""
Subword Tokenization is a NLP technique in which a word is split into
subwords and these subwords are known as tokens.This technique is used
in any NLP task where a model needs to maintain a large vocabulary and
complex word structures.The concept behind this,frequently occurring words
should be in the vocabulary whereas rare words are split into frequent
subwords.For example,the wood "unwanted" might  be split into “un”, “want”, and “ed”.
The word “football” might be split into “foot”, and “ball”.

Subword Tokenization in NLP
To implement the subword tokenization,we need text data to apply or we can
give a small string input for the base test case.First,we tokenize the given sentences,
for example, "Geeksforgeeks!is-best,for.@geeks don't".If we simplify it
by the space then we see that it creates  "is-best" which includes a
special character that exploits our vocabulary so we need to fix it.And every special
character will also be part of out vocabulary.So the expected vocabulary
will be [“geeksforgeeks”,”!”,”is”,”-“,”best”,”for”,”@”,”geeks”].Here we
also need to convert it into lowercase.
"""
import re
from collections import OrderedDict

test_str = """
GeeksforGeeks is a fantastic resource for geeks 
who are looking to enhance their programming skills, 
and if you're a geek who wants to become an expert programmer, 
then GeeksforGeeks is definitely the go-to place for geeks like you.
"""
print("The original string is:"+str(test_str))
test_str = test_str.lower()
res = re.findall(r'\w+|[^\s\w]+', test_str)
print("The converted string:\n"+str(res))

"""
Since we are taking each word. it creates a large dictionary and because of this,
word tokenization can have an exploding vocabulary problem. 
To get rid of this problem we use tokenization on characters. 
Character tokens solve this large vocabulary problem. For that, 
we need to create a dictionary that has the frequency of each word
in the sentence after the word tokenization and separate each word by space. 
The below code is the implementation of the above process.
"""
res_dict = OrderedDict()
for i in res:
    new_strings = ' '.join(char for char in i)
    if new_strings in res_dict:
        res_dict[new_strings] += 1
    else:
        res_dict[new_strings] = 1
print(res_dict)