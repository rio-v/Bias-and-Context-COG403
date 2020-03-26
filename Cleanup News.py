import io
from gensim import utils
import json
import zstandard

class MyCorpus(object):
    def __iter__(self):
        # Select .zst file to read from
        # These .zst files of reddit comments and their metadata are taken from
        # https://files.pushshift.io/reddit/comments/
        # Note these files are incredibly large
        # DO NOT ATTEMPT TO READ THE WHOLE FILE!
        with utils.open(r'C:\Users\Eric\Documents\COG 403\Project\Reddit\Data\Comments\RC_2019-09.zst', 'rb') as f:
            # Initialize file readers and decompressors
            dctx = zstandard.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
            # Initialize which comments to remove (i.e. all comments that have been removed or deleted) and initialize
            # which subreddits to read from, comment out line to read from all subs
            remove = ['[deleted]', '[removed]']
            subs = ['politics', 'worldnews', 'news']
            # Initialize count of total number of comments and words included in corpus
            comment_count = 0
            word_count = 0
            for line in text_stream:
                # If under a set number of comments, then include next comment
                if comment_count < 1000000:
                    line = json.loads(line)
                    post = line['body']
                    sub = line['subreddit']
                    # Uncomment if only reading from specific subreddits
                    if sub in subs:
                        if post not in remove:
                            # Process post as required for a word2vec corpus
                            processed_post = utils.simple_preprocess(post)
                            # Increase word and comment counts
                            comment_count += 1
                            word_count += len(processed_post)
                            # Return
                            yield processed_post
                else:
                    # Print word count upon completion
                    print(word_count)
                    break

corpus = MyCorpus()
# Save corpus as a line_sentence for a word2vec model to be made from
utils.save_as_line_sentence(corpus, r"C:\Users\Eric\Documents\COG 403\Project\Reddit\Data\Comments\RC_2019-09-news.txt")

