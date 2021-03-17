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

+ Project summary: In this project, we created a classification engine for facial emotion recognition. In this project, we use 15 models. The baseline model uses gradient boosting machine (gbm) with the accuracy of 83.9%. We have also evaluated other machine learning models(fast GBM, Logistic Regression, adaboost(base estimator BaggingClassifier), Multi-layer Perceptron Classifier, LDA, SVM, Guissian Naive Bayes, Bagging(base estimator ExtraTreesClassifier), SGD,  xgboost, RandomForestClassifier, KNN, PCA + SVM, VotingClassifier,  Tensorflow Deep Neural Networks) and chose the best one (adaboost) based on predictive performance and computation efficiency. Our final advanced model uses adaboost and accuracy is 92.5% AUC is 97.6%. 
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) 
Anh Thu Doan : Originally did knn, xgboost, svm and logistic. Later on i tuned knn and xgboost.
Hao Hu: Did pca+svm in R.
Feng Rong: Did LDA in Python.
Siyuan Sang: Did Random Forest model in R.
Yibai Liu: Did baseline GBM, fast GBM, adaBoost, Multi-layer Perceptron Classifier, Stochastic Gradient Descent, Guissian Naive Bayes, Bagging Classifier, VotingClassifier, SVM, Tensorflow Deep Neural Networks.
Also generated main.ipynb and other python scripts, created reduced and resampled feature sets, incorporated others' work into the main notebook.

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
