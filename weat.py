from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
import pandas as pd
import nltk
import numpy as np
import pickle
from itertools import combinations
import math

modelname = "INSERT MODEL FILENAME HERE"

model = Word2Vec.load(modelname)

def set_fix(model, words):
    #removes any words that are not in the model from a set of words
    return [w for w in words if w in model]

def association(model,w,a,b):
    tmp1 = np.array([model.similarity(w, a_i) for a_i in a])
    tmp2 = np.array([model.similarity(w, b_i) for b_i in b])
    return np.mean(tmp1) - np.mean(tmp2)

def weat(model,x,y,a,b):
    tmp1 = np.array([association(model,x_i,a,b) for x_i in x])
    tmp2 = np.array([association(model,y_i,a,b) for y_i in y])
    return np.sum(tmp1) - np.sum(tmp2)

def weat_p(model,x,y,a,b):
    #Warning: function is very computationally expnsive; slows down significantly after 6 target words
    base = weat(model,x,y,a,b)
    count = 0
    pool = x+y
    comb_len = math.factorial(len(pool)) / math.factorial(len(x)) / math.factorial(len(pool)-len(x))
    for xi in combinations(pool, len(x)):
        yi = [i for i in pool if i not in xi]
        count = count + (weat(model,xi,yi,a,b) > base)
    return count/comb_len

def effect_size(model,x,y,a,b):
    pool = x+y
    tmp1 = np.array([association(model,x_i,a,b) for x_i in x])
    tmp2 = np.array([association(model,y_i,a,b) for y_i in y])
    tmp3 = np.array([association(model,w,a,b) for w in pool])
    return (np.mean(tmp1) - np.mean(tmp2))/np.std(tmp3)

#Target and function words from original WEAT study for pleasantness vs unpleasantness of plants vs. bugs:
target1 = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose", "bluebell", "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet", "carnation", "gladiola", "magnolia", "petunia", "zinnia"]
target2 = ["ant", "caterpillar", "flea", "locust", "spider", "bedbug", "centipede", "fly", "maggot", "tarantula", "bee", "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket", "hornet", "moth", "wasp", "blackfly", "dragonfly", "horsefly", "roach", "weevil"]
att1 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter", "paradise", "vacation"]
att2 = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", "rotten", "vomit", "agony", "prison"]

#f1 = set_fix(model, target1)
#f2 = set_fix(model, target2)
#f2 = [f2[i] for i in range(len(f1))]
#f3 = set_fix(model, att1)
#f4 = set_fix(model, att2)


