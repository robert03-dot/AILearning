"""
Many words used in the phrase are insignificant and hold no meaning.For example - English is a subject
Here,'English' and 'subject' are the most significant words and 'is','a' are almost useless.
English subject and subject holds the same meaning even if we remove the insignificant words - ('is','a').
Using the nltk,we can remove the insignificant words by looking at their part-of-speech tags.
For that we have to decide which POS tags are significant
"""

def filter_insignificant_words(chunk,
                               tag_suffixes = ['DT', 'CC']):
    good=[]

    for word,tag in chunk:
        ok=True

    for suffix in tag_suffixes:
        if tag.endswith(suffix):
            ok=False
            break

        if ok:
            good.append((word,tag))

    return good

print("Significant words: \n",
      filter_insignificant_words([('the' 'DT'),
                                 ('terrible','JJ'),
                                  ('movie','NN')]))