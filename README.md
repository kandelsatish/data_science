## Introcuction to the Machine Learning
The soul purpose of this repository is to learn the basics of the machine learning with out diving into the details of how
each of the machine learning algorithms work. While working in this repositoty I will be focusing in how to solve the real world 
problems with out diving into the underlying mathematics involved in each of the machine learning algorithm. I will be comprehending the knowledge that i learn from the book: "Introduction to Machine Learning with Python"


## What is machine learning and why is it used?
Machine learning is the research field at the intersection of Statistics, Artificial Intelligence and Computer Science, that is used to extract the knowlegde from the data. Unlike the traditional intelligent system, which were created with the if else rules
which we used to think they are intellegent. However, this is only feasible, when we know every thing about the system, and all the 
conditions with in that system. This usually don't happen in the large application with large data. Having intelligent system made with the hard coded condition, i.e codeing the system with traditional if..eles, has major disadvantages:
* The system developed will be able to solve the problem for single domain
* While desigining the system, designer should have deep understanding about how the decesions are made.

Therefor, machine learning can solve these problem by identifying the patterns with in the data and can make the prediction about unseen data. What we have  to do is simply present the machine learning algorithm or the programm to the large collection of data.


## Types of macine learning algoriths-Supervised and Unsupervised  learning
* Supervised learning: An algrorithms that generalized from the known examples are the supervised learning algorithms. In this types of learning user feed the algorithms with the input and the desired output then the algorithms learns how to find the output.
As a result of which algorith becomes able to produce the desired output for the input it has never seen before.

* Unsupervised learning: The algorithms for which users provides the input with out the desired output. When huge number of input data are provided to the algorithms, it identifies the pattern with in the data with out human supervision. Then, it becomes able to predict the output for the unseen data sets. This types of algorithms are called unsupervised learning.

## Know your data
The most important step in any data project is to know well about the data that we are going to deal with and the problem that we are trying to solve. Building model using random algorith is not of use, since there are different types of algorithms that best work for specefic types of data and the condition. We should be able to know out data clearly so that we can choose the algorithm to solve the problem.

There are the several questions that we need to keep in mind while building any machine learning model:
1. What is the problem that we are trying to solve?
2. Do i have the enough data to get started?
3. What features did we choosed and are these features enough to predict the output?
4. How we are going to measure the success of the model?
5. How this model will contribute to other part of our research?

## scikit-learn
scikit-learn is the open source liberary, which provides the different types of algorithms like classification, regression, clustering etc.This package depends on the other two liberaries Numpy and Scipy. 

## installing necessary liberaries and settingup environment
     pip install numpy scipy matplotlib ipython scikit-learn pandas


## Sparse matrix
The most important part of the SciPy is scipy.sparse, which gives us sparse matrix, which is one of many data representation. Sparse matrix is used whenever we want to store a 2D array that mostly contains zeros. Sparse matrix will contains the most element zero and only few non zero elements.

## mglearn
The book provides the code that for each of the examples that are shown in this book and also provides the utilities functions. These codes function can be found in the following github repository:
https://github.com/amueller/introduction_to_ml_with_python


## Classification algorithms

### k-Nearest Neighbors
With this, the model finds the point in the training data set that is close to the new data point for which we want to make the prediction. After, it assign the label of the training data to the new data point.

k is a number of neighbours that model should consider while making prediction.


## Estimator classes
all the machine learning models are implemented in their own classes which are called the Estimator classes.




