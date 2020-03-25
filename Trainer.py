import gensim

# Select which corpus file to the build model from
corpus = r'RC_2019-09-mil.txt'
# Build model from corpus
model = gensim.models.Word2Vec(corpus_file=corpus, size=300)
# Save model
model.save("word2vec-mil-second.model")
