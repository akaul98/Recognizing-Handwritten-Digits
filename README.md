In this project, I have uploaded the data-set and tested for the hypothesis which is "The Digits data set of the sci-kit-learn library provides numerous data-sets.
Some Scientist claims that it predicts the digit accurately 95% of the time.  
I performed data analysis to accept or reject this Hypothesis and finally put-forth my conclusion. 



Recognizing Handwritten Digits





In this project, I have uploaded the data-set and tested for the hypothesis which is "The Digits data set of the sci-kit-learn library provides numerous data-sets.Some Scientist claims that it predicts the digit accurately 95% of the time.  I performed data analysis to accept or reject this Hypothesis and finally put-forth my conclusion. 



The sci-kit-learn library (HTTP://scikit-learn.org/) enables helps you to approach this type of project. The data to be analyzed is closely related to numerical values or strings, but can also involve images and sounds. The problem  faced in this  project is to  predict a numeric value, and then reading and interpreting an image that uses a handwritten font. So even in this case I  have used an estimator with the task of learning through a fit() function, and once it has reached a degree of predictive capability (a model sufficiently valid), which will produce a prediction with the predict() function. Then I have created two sets the training set and validation set, created this time from a series of images.



An estimator that is useful in this case is sklearn.svm.SVC, which uses the technique of Support Vector Classification (SVC). Thus, I have to import the SVM module of the scikit-learn library and created an estimator of SVC type and then choose an initial setting, assigning the values C and gamma generic values. These values can then be adjusted in a different way during the course of the analysis. 



from sklearn import svm

 clf = svm.SVC(gamma=0.001, C=100.)



I have also imported all the necessary header files in this project which are as follows:



import numpy as np

import pandas as pd

import matplotlib.pyplot as pt

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

import matplotlib.pyplot as plt



The Digits Data-set



The scikit-learn library provides numerous data-sets that are useful for testing many problems of data analysis and prediction of the results. Also, in this case, there is a data-set of images called Digits. This data-set consists of 1,797 images that are 8x8 pixels in size. Each image is a handwritten digit in the grey-scale, as shown in Figure 1.





Figure 1. One of 1,797 handwritten number images that make up the data-set digit



Thus, I  loaded the Digits data-set into my Jyputer  Notebook.



 from sklearn import datasets

 digits = datasets.load_digits()



 After loading the data-set, I analyzed the content. First, I read lots of information about the data-set by calling the DESCR attribute.



 print(digits.DESCR) 



For a textual description of the data-set, the authors who contributed to its creation and the references will appear as shown in Figure 2.







Figure 2. Each data-set in the scikit-learn library has a field containing all the information.



The images of the handwritten digits are contained in a digit. images array. Each an element of this array is an image that is represented by an 8x8 matrix of numerical values that correspond to a grey-scale from white, with a value of 0, to black, with the value 15.



digits.images[0]



 This will be following result: 

array([[ 0., 0., 5., 13., 9., 1., 0., 0.],

 [ 0., 0., 13., 15., 10., 15., 5., 0.],

 [ 0., 3., 15., 2., 0., 11., 8., 0.],

 [ 0., 4., 12., 0., 0., 8., 8., 0.],

 [ 0., 5., 8., 0., 0., 9., 8., 0.], 

[ 0., 4., 11., 0., 1., 12., 7., 0.],

 [ 0., 2., 14., 5., 10., 12., 0., 0.],

 [ 0., 0., 6., 13., 10., 0., 0., 0.]]) 



One can visually check the contents of this result using the matplotlib library and typing the following code:



plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')



This will be following result: 



Figure 3. One of the 1,797 handwritten digits



Learning and Predicting 





Now that I have loaded the Digits data-sets into my notebook and have defined an SVC estimator, I can start learning. You should be knowing that, once you define a predictive model, you must instruct it with a training set, which is a set of data in which you already know the belonging class. Given the large number of elements contained in the Digits data-set, you will certainly obtain a very effective model, i.e., one that’s capable of recognizing with good certainty the handwritten number. This data-set contains 1,797 elements, and so you can consider the first 1,794 as a training set and will use the last four as a validation set. You can see in detail these four handwritten digits by using the matplotlib library:



plt.subplot(321)

plt.imshow(digits.images[1791], cmap=plt.cm.gray_r,

interpolation='nearest')

plt.subplot(322)

plt.imshow(digits.images[1792], cmap=plt.cm.gray_r,

interpolation='nearest') 

plt.subplot(323)

plt.imshow(digits.images[1793], cmap=plt.cm.gray_r,

interpolation='nearest')

plt.subplot(324)

plt.imshow(digits.images[1794], cmap=plt.cm.gray_r,

interpolation='nearest')







Figure 4 shows the output of the following code





Figure 4: The four digits of the validation set





Now you can train the clf estimator that you defined earlier.



clf.fit(digits.data[1:1790], digits.target[1:1790])


Now you have to test your estimator, making it interpret the four digits of the validation set.


clf.predict(digits.data[1791:1795])



 You will obtain these results: array([4, 9, 0, 8])





If you compare them with the actual digits, as follows:


digits.target[1791:1795]

You will obtain these results: array([4, 9, 0, 8]



conclusion



One can choose a smaller training set and a different range for validation. In the above case, we have got 100% accurate predictions, but this may not be the case at all times. That is because we have taken less number of test sets. There is a possibility that the prediction may decrease as the number of tests increases.

For more Infomation read my blog https://adarshprojects.blogspot.com/2019/07/recognising-handwritten-digits.html


