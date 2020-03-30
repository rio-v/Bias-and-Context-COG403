import gensim

# Select which corpus file to the build model from
corpus = r'C:\Users\Eric\Documents\COG 403\Project\Reddit\Data\CNN\cnn_corpus.txt'
# Build model from corpus
model = gensim.models.Word2Vec(corpus_file=corpus, size=250)
# Save model
model.save(r'C:\Users\Eric\Documents\COG 403\Project\Reddit\git\News\CNN\word2vec-cnn.model')
print(len(model.wv.vocab))