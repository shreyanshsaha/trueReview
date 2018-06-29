# trueReview
A Movie review system based on the comments of the users and not the rating. Made using NLP and Semtiment Analysis

### How to use
Clone the folder and type `npm install` to install all modules

## How is the machine learning part handled
The machine learning and prediction is done by *python* using the NLTK library for natural language processing. 
There are more than one classifiers at play. The classifiers are:
1. Multinomial Naive Bayes
2. Bernoulli Naive Bayes
3. Logistic Regression
4. SGD Classifier
5. SVC
6. Linear Support Vector Classifier
7. NuSVC

All of these classifier are combined under a voted classifier class:
```
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
```

Here:
1. *_classify(self, featureset)_* is responsible to classify the given      input. It does this by taking the mode of the most predicted value.
2. *_confidence(self, featureset)_* gives us the confidence or               reliability of our predicted data. I.e it tells us how many of our        classifiers predicted the same result.

The classifiers are saved using pickle in *./classifiers* to avoid training them again and again. Also the voted classifier is saved along with them.

## Incorporation in Node
Node simple calls the python script and gives the paragraph as an argument. The script returns the prediction to _stdout_.

`exec('python "./externalScripts/predict.py" "'+review+'"', callback(){})`

The average time in predictions of 25 reviews consisting of multiple paragraphs is 15 Seconds.

#### Accuracy
*Highest:* 97%
*Lowest:* 83%
*Average:* 90.06%
*Mode:* 93%