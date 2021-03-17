# Project: Can you recognize the emotion from an image of a face? 
<img src="figs/CE.jpg" alt="Compound Emotions" width="500"/>
(Image source: https://www.pnas.org/content/111/15/E1454)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2021

+ Team 2
+ Team members
	+ Anh Thu Doan
	+ Hao Hu
	+ Yibai Liu
	+ Feng Rong
	+ Siyuan Sang

+ Project summary: In this project, we created a classification engine for facial expressions. The baseline gradient boosting machine (gbm) model had accuracy of 81.67% and AUC of 0.564, which uses pairwise distances between X and Y coordinates of fiducial points. We further explored the data to extract the top 100 most important attributes and developed 15 machine learning models, mostly using the `sklearn` package in Python, including fast GBM, Logistic Regression, Adaboost with base estimator BaggingClassifier, Multi-layer Perceptron Classifier, LDA, SVM, Guissian Naive Bayes, Bagging with base estimator ExtraTreesClassifier, SGD, XGboost, Random Forest, K Nearest Neighbors, PCA + SVM, Voting Classifier, and Tensorflow Deep Neural Networks. Among these candidates, Multi-layer Perceptron Classifier (MLP) has the best performance in terms of accuracy, AUC area, time complexity, and balanced accuracy. As a more advanced model, MLP got a claimed accuracy of 84.17% and AUC of 0.689. 
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) 
 - Anh Thu Doan : Originally did knn, xgboost, svm and logistic. Later on i tuned knn and xgboost.
 - Hao Hu: Did pca+svm in R.
 - Feng Rong: Did LDA in Python.
 - Siyuan Sang: Did Random Forest model in R.
 - Yibai Liu: Did baseline GBM, fast GBM, adaBoost, Multi-layer Perceptron Classifier, Stochastic Gradient Descent, Guissian Naive Bayes, Bagging Classifier, VotingClassifier, SVM, Tensorflow Deep Neural Networks. Also generated main.ipynb and other python scripts, created reduced and resampled feature sets, and incorporated others' work into the main notebook.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
