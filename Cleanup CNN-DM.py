import os
from gensim import utils

class MyCorpus(object):
    def __iter__(self):

        directory = r'C:\Users\Eric\Documents\COG 403\Project\Reddit\Data\CNN\cnn\stories'
        story_count = 0
        word_count = 0

        for filename in os.listdir(directory):
            if word_count >= 35000000:
                break
            name = directory + '\\' + filename
            f = open(name, encoding='utf-8')
            f.readline()
            line = f.readline()
            while line == '\n':
                line = f.readline()
            while line[0] != '@':
                processed_line = utils.simple_preprocess(line)
                word_count += len(processed_line)
                line = f.readline()
                while line == '\n':
                    line = f.readline()
                yield processed_line
            story_count += 1
        print('Number of stories in corpus: {}'.format(story_count))
        print('Number of total words in corpus: {}'.format(word_count))

corpus = MyCorpus()
# Save corpus as a line_sentence for a word2vec model to be made from
utils.save_as_line_sentence(corpus, r"C:\Users\Eric\Documents\COG 403\Project\Reddit\Data\CNN\cnn_corpus.txt")