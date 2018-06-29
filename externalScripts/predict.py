
import pickle
import argparse
import nltk
from nltk.corpus import  movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode

# ===============
# Argument Parser
# ===============
parser = argparse.ArgumentParser(description="Movie Review Sentiment Analysis")
parser.add_argument("paragraph", help="Give the string to analysize", type=str)
# Add argument to check the accuracy
args = parser.parse_args();

# ===========
# Class
# ===========
class VoteClassifier(ClassifierI):
  def __init__(self, classifiers):
    self.classifiers = classifiers

  def classify(self, featureset):
    votes = []
    for classifier in self.classifiers:
      v = classifier.classify(featureset)
      votes.append(v)
    return mode(votes)

  def confidence(self, featureset):
    votes = []
    for classifier in self.classifiers:
      v = classifier.classify(featureset)
      votes.append(v)
    choiceVotes = votes.count(mode(votes))
    conf = choiceVotes/len(votes)
    return conf

f=open("./externalScripts/classifiers/wordFeatures.list.pickle","rb")
wordFeatures=pickle.load(f)
f.close()

file = open("./externalScripts/classifiers/votedClassifier.pickle","rb");
classifier = pickle.load(file)
file.close()

# # list of all the words
# allWords=[]
# for w in movie_reviews.words():
#   allWords.append(w.lower())

# allWords = nltk.FreqDist(allWords)

# # Selecting 3000 most common words as the word features
# wordFeatures = list(allWords)[:1000]



# Convert the review into a featureset
def findFeatures(review):
  words = set(review)
  features={}
  for w in wordFeatures:
    features[w] = w in words
  return features

print(classifier.classify(findFeatures(word_tokenize(args.paragraph))), end='')