# ================
# Author: Shreyansh Saha
# Description: A script to predict if the review is positive or negetive
# ================


# ========
# IMPORTS
# ========

import nltk
import random
import pickle
import argparse

from nltk.tokenize import word_tokenize

from nltk.corpus import movie_reviews

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

import os

from nltk.classify import ClassifierI
from statistics import mode

# ===============
# Argument Parser
# ===============
# parser = argparse.ArgumentParser(description="Movie Review Sentiment Analysis")
# parser.add_argument("paragraph", help="Give the string to analysize", type=str)

# args = parser.parse_args();

# ==================
# Voted Classifier
# ==================
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

# Tuple which contains all the reviews
reviewsCollection = [(list(movie_reviews.words(fileid)), category)
                      for category in movie_reviews.categories()
                     for fileid in movie_reviews.fileids(category)]

# Shuffle the data
random.shuffle(reviewsCollection)

# list of all the words
allWords=[]
for w in movie_reviews.words():
  allWords.append(w.lower())

allWords = nltk.FreqDist(allWords)

# Selecting 3000 most common words as the word features
wordFeatures = list(allWords)[:1000]

# Convert the review into a featureset
def findFeatures(review):
  words = set(review)
  features={}
  for w in wordFeatures:
    features[w] = w in words
  return features

# All data of reviews converted to features
featureSets = [(findFeatures(rev), category) for (rev, category) in reviewsCollection]

def saveBestClassifier(classifier, fileName, featureSet, testRatio, sklean=False):
  rows = len(featureSet)
  print("Training %s 100 times..." %(fileName))

  if sklean:
    classifierBest=nltk.SklearnClassifier(classifier)
  else:
    classifierBest = classifier

  maxAcc=0
  for i in range(100):
    random.shuffle(featureSet)

    # Spliting into train and test set
    trainingSet = featureSet[:1900]
    testingSet = featureSet[1900:]

    if sklean:
      classifierCurr = nltk.SklearnClassifier(classifier).train(trainingSet)
    else:
      classifierCurr=classifier.train(trainingSet)

    currAcc = nltk.classify.accuracy(classifierCurr, testingSet)
    if currAcc > maxAcc:
      classifierBest = classifierCurr
      maxAcc = currAcc
      print("New best accuracy for %s:" %(fileName), maxAcc)
    print("Done: %.2f" %(i/100))
  print("Saving %s with max accuracy:" %(fileName), maxAcc)

  file = open("./classifiers/"+fileName+".pickle", "wb")
  pickle.dump(classifierBest, file)
  file.close()

# Check if classifiers are saved or not
onlyfiles = [f for f in os.listdir("./externalScripts/classifiers/")]
if len(onlyfiles)<0:
  print("No classifiers found in directory ./classifiers! Making the new classifiers!")
  saveBestClassifier(nltk.NaiveBayesClassifier, "naiveBayes", featureSets, 0.1)
  saveBestClassifier(MultinomialNB(), "multinomialNaiveBayes", featureSets, 0.1, True)
  saveBestClassifier(BernoulliNB(), "bernoulliNaiveBayes", featureSets, 0.1, True)
  saveBestClassifier(LogisticRegression(), "logisticRegression", featureSets, .1, True)
  saveBestClassifier(SGDClassifier(), "SGDClassifier", featureSets, 0.1, True)
  saveBestClassifier(LinearSVC(), "linearSVC", featureSets, 0.1, True)
  saveBestClassifier(NuSVC(), "nuSVC", featureSets, 0.1, True)
  print("Classifiers saved! Exititng")
  exit(0)
else:
  print("[!] Classifiers found! Using these classifiers: ",onlyfiles)
  pass

# Make the list of all classifiers
classifiers=[]
for i in onlyfiles:
  path="./externalScripts/classifiers/"+i
  file=open(path, "rb")
  classifierTemp = pickle.load(file)
  file.close()
  classifiers.append(classifierTemp)
print("[+] All classifiers loaded to array 'classifiers'!")

# make the voted classifier
print("[+] Making Voted Classifier!")
votedClassifier = VoteClassifier(classifiers)

file = open("./externalScripts/classifiers/votedClassifier.pickle","wb");
pickle.dump(votedClassifier, file);
file.close();

print("[!] Saved voted classifier!")
print("\n[!] Testing Accuracy")

# acc=[]
# for i in range(100):
#   random.shuffle(featureSets)
#   # Spliting into train and test set
#   testingSet = featureSets[1900:]
#   acc.append(nltk.classify.accuracy(votedClassifier,testingSet))
#   print("Done: ",i/100)

# print("Voted Classifier Average Accuracy: ", sum(acc)/len(acc))
# print("Voted Classifier Best Accuracy: ",max(acc))
# print("Voted Classifier Least Accuracy: ", min(acc))
print(featureSets[1123][1])
print(votedClassifier.classify(featureSets[1123][0]), end='');
