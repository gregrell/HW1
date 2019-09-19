import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!
        
        ###########################


        # YOUR CODE HERE
        #RELL - Numpy Working
        print 'Classes: %s, Docs: %d, Words: %d' %(classes,n_docs,n_words)
        # initialize the array of size number of classes

        theCat, num_cat =np.unique(y,return_counts=True)

        print 'the cat ', theCat
        print 'the cat count ',num_cat
        numClass=[0] * n_classes

        # for each class, find how many matches there are in the y array, increment the numClass array at that
        # class index.

        for myClass in classes:
            for i in y:
                if y[i][0]==myClass:
                    numClass[myClass]+=1
        print numClass

        #Set the prior parameters
        prior[0]=float(numClass[0])/n_docs
        prior[1]=float(numClass[1])/n_docs

        print 'Prior is ',prior

        #print x.shape[1] # numpy shape = returns the size of an array by dimension array.shape[0] rows, array.shape[1]
        # columns
        # the x,y data is represented as follows: all the documents were dissected into a bag of words.
        # The volcabulary is 13989 words and there are 1600 total documents. These documents are either class
        # positive or negative and in those 1600 rows they are mixed at random.
        # Each row represents a document, and each column represents the number of times that particular word exists
        # in that document. That's it. It's a 2D array of statistics.

        countOfWordInPosClass=np.zeros((n_words,n_classes))
        i=0
        for word in range(x.shape[1]):
            for doc in range(x.shape[0]):
                countOfWordInPosClass[word, y[doc][0]] += x[doc, word]
                i+=1
                if i%1000000==0:
                    print float(i)/(n_words*n_docs) * 100, '% done'
        totalWordsClass=[0,0]


        totalWordsClass[0]=np.sum(countOfWordInPosClass[:,0])
        totalWordsClass[1]=np.sum(countOfWordInPosClass[:,1])
        print totalWordsClass[0]
        print totalWordsClass[1]

        # Targets: Accuracy on training set: 0.985625, on test set: 0.687500.

        for word in range(x.shape[1]):
            likelihood[word, 0] = float(countOfWordInPosClass[word, 0])/(totalWordsClass[0]+n_words)
            likelihood[word, 1] = float(countOfWordInPosClass[word, 1])/(totalWordsClass[1] + n_words)










        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
