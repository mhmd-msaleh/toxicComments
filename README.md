# Toxic Comment Classification 
##### Machine Learning Enagineering Nanodegree Capstone Project

### Proposal review link: 
https://review.udacity.com/#!/reviews/3385924


## Project Overview
### Project Domain
Nowadays, the majority of individuals are active on at least one social networking platform. Not just adults, but even teenagers use them. Nonetheless, some people abuse these platforms to vent their wrath and rage via bullying and the use of filthy language. That is why it is critical to avoid this sort of behavior on the platform by banning anyone who use this language. It will be a tiresome task to force humans to read each remark and determine whether or not it is offensive. That is where Machine Learning comes in help; we can train a model to determine which comment is harmful.

### Dataset and Input
The dataset was obtained for free from a Kaggle competition page. It contains a large number of comments taken from Wikipedia talk page. 
The dataset is divided into three csv files: 
 * train.csv: containing 150000 comments for training. 
 * test.csv: containing another 150000 comments for testing (without labels). 
 * test_labels.csv: containing labels for test.csv entries. 

### Problem Statement
This project will design a machine learning model that will evaluate a huge set of comments and then utilize the trained model to predict the level of toxicity of a given comment. The level of toxicity will fall into one or more of the following types: 
 * `toxic`
 * `severe_toxic`
 * `obscene`
 * `threat`
 * `insult`
 * `identity_hate`

## External Libraries Used Throughout the Project: 
 * `Pandas`
 * `Sklearn`

## Model Algorithems used Throughout the Project: 
 * `Naive Bayes MultinomialNB`
 * `Logistic Regression`
 * `Linear Support Vector Machines`



