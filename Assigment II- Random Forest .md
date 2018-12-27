###  Assigment II: Random Forest 
----

Objective: In this excercise you will train a random forest on Titanic Data. 

Due Date: January 14th

Dataset: You can download data from [kaggle website](https://www.kaggle.com/c/titanic)


Packages you need (suggestion): **matplotlib, numpy, sklearn, pandas**

You can read about RF class in sklearn [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

-----


#### Step 1: 
Load the training data in pandas, check for missing values and apply relevent diagnostics.






#### Step 2: One hot encoding


When dealing with categforical data, it is important to convert them to a form of numerical representations so the computer systems can understand them better. OPne of the most basic but popular ways of this conversion is known as one-hot-encoding ( you can read more [here](https://medium.com/@michaeldelsole/what-is-one-hot-encoding-and-how-to-do-it-f0ae272f1179). 


![One Hot Encoding](https://cdn-images-1.medium.com/max/1600/0*T5jaa2othYfXZX9W.)

In python, you can use sklearns prerocessing toolbox for this purpose, however we need to use LabelEncoder ( convert categories to numbers e.g 1,2,3 first and then encode them to vectors )

```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

le = LabelEncoder()
X = le.fit_transform(....)

ohe = OneHotEncoder(categorical_features = ...)
X = ohe.fit_transform(X).toarray()

```



Alternatively, you can also use [Pandas.get_dummies] (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) to convert categorical variables into dummy/ indicator variables. 

Encode the categorical variables if needed.


#### Step 3: Baseline model 

Train a Random= Forest Classifier on your data. Whats the AUC value for your model? Plot the ROC Curve and interpret your results!




#### Step 4: Number of Estimators


n_estimators represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data. However, adding a lot of trees can slow down the training process considerably, therefore lets do a parameter search to find the sweet spot. Calculate the AUC score for different number of trees.


You can use the code below to plot AUC Score vs n_estimators using the code below:


```
# train_results and test_results are lists of AUC scores for different number of estimators in train and test sets.
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(n_estimators, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘n_estimators’)
plt.show()


```


#### Step 5: Tuning max_depth

max_depth represents the depth of each tree in the forest. The deeper the tree, the more splits it has and it captures more information about the data. 


visualize the effect of max_depth for test and train set data AUC scores. What are your findings?



#### STEP 6: min_samples _split

min_samples _split represents the minimum number of samples required to split an internal node. 

This can vary between considering at least one sample at each node to considering all of the samples at each node. When we increase this parameter, each tree in the forest becomes more constrained as it has to consider more samples at each node. Lets try and experiment with different values of this parameter from 10% to 100%.





#### STEP 7: min__samples _leaf

min_samples_leaf is The minimum number of samples required to be at a leaf node. This parameter is similar to min_samples_splits, however, this describe the minimum number of samples of samples at the leafs, the base of the tree. We will continue experimenting different values using below calculation:


```
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

```


##### STEP 8: max_features

max_features represents the number of features to consider when looking for the best split. Lets experiment different values using below line of code:


```

max_features = list(range(1,train.shape[1]))
```



#### STEP 9: Putting ALL together!!!


Using a searching mechanism find the optimal parameters in one block of code! Do the optimal values for each of the above steps match to your findings? 






















