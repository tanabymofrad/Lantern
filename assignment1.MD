### Assignment One
-----------------------------
# K Nearest Neighbor Algorithm classifiction for Glass dataset


* Deadline for submission: December 23rd 
* Submission: push your solution in your github page, send me the link to your code via email
* email: tanaby.mofrad@gmail.com
* How to use Git: There are various tutorials available for GIT setup and use. 

[Here](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners) is one tutorial which might be helpul


 ---
### Glass Identification Dataset:

Attribute Information:

   	1. Id number: 1 to 214
   	2. RI: refractive index
   	3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as 
                  are attributes 4-10)
   	4. Mg: Magnesium
   	5. Al: Aluminum
   	6. Si: Silicon
   	7. K: Potassium
   	8. Ca: Calcium
   	9. Ba: Barium
  	10. Fe: Iron
  	11. Type of glass: (class attribute)
      -- 1 building_windows_float_processed
      -- 2 building_windows_non_float_processed
      -- 3 vehicle_windows_float_processed
      -- 4 vehicle_windows_non_float_processed (none in this database)
      -- 5 containers
      -- 6 tableware
      -- 7 headlamps
      
You can download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/glass+identification)

#### Procedure:

1. Load libraries/ data

	- Load and explore the shape of the dataset

2. Summarize Data

	- Descriptive statistics
	- Correlation matrix
	- Univariate and bivariate analysis and visualizations
	
	**`What did you understand about data?`**

3. Data Preparation:

	- Data Cleaning: Do you need to clean the data?
	- Split-out validation dataset
	- Data transformation

4. Baseline model: train a KNN classifier to serve as the baseline. 

5. Parameter tuning with cross validation:

	Obviously, the best **K** is the one that corresponds to the **lowest test 	error rate**, so let’s suppose we carry out repeated measurements of the test 	error for different values of K. 

	However, what we are doing is using the test set as a training set! This means that we are underestimating the true error rate since our model has been forced to fit the test set in the best possible manner. 
	
	Our model is then incapable of generalizing to newer observations, a process known as **overfitting**. So, touching the test set is out of the question and must only be done at the very end of our pipeline.
	
	`Using the test set for hyperparameter tuning can lead to overfitting.`
	
	**An alternative and smarter approach involves estimating the test error rate by holding out a subset of the training set from the fitting process.**

	**This subset, called the validation set, can be used to select the appropriate level of flexibility of our algorithm!**
	
	**There are different validation approaches that are used in practice, and we will be exploring one of the more popular ones called k-fold cross validation.**
	
	![cross-validation](https://kevinzakka.github.io/assets/k_fold_cv.jpg)
	
	As seen in the image, k-fold cross validation (the k is totally unrelated to K) involves randomly dividing the training set into k groups, or folds, of approximately equal size. 
	
	The first fold is treated as a validation set, and the method is fit on the remaining k−1 folds. The misclassification rate is then computed on the observations in the held-out fold. 
	
	This procedure is repeated k times; each time, a different group of observations is treated as a validation set. This process results in k estimates of the test error which are then averaged out. The code in below will help you to calculate the scores using k-fold cross validation.
		
	```
	from sklearn.model_selection import cross_val_score
	# creating odd list of K for KNN
	 k_list= list(range(1,50))
	
	# subsetting just the odd ones
	neighbors = filter(lambda x: x % 2 != 0, k_list)
	
	# empty list that will hold cv scores
	cv_scores = []
	
	# perform 10-fold cross validation
	for k in neighbors:
	    knn = KNeighborsClassifier(n_neighbors=k)
	    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
	    cv_scores.append(scores.mean())
	
	```

	As we can see above, scikit-learn comes in handy with its cross_val_score() method. We specifiy that we are performing 10 folds with the cv=10 parameter and that our scoring metric should be accuracy since we are in a classification setting.
	
	Finally, we plot the misclassification error versus K (Use the code below). 
	
	
	```
	# changing to misclassification error
	MSE = [1 - x for x in cv_scores]
	
	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	print "The optimal number of neighbors is %d" % optimal_k
	
	# plot misclassification error vs k
	plt.plot(neighbors, MSE)
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('Misclassification Error')
	plt.show()
	
	```
	** What would be the best value of K to choose??? Why? **




6. Apply Diagnostics:

 - A simple and effective way to remedy skewed class distributions is by implementing weighed voting. The class of each of the K neighbors is multiplied by a weight proportional to the inverse of the distance from that point to the given test point. This ensures that nearer neighbors contribute more to the final vote than the more distant ones. How can you do this in Sklean Knearestneighbor class?
 
- Changing the distance metric for different applications may help improve the accuracy of the algorithm. Which of other distance measures can you try? Do hem improve your model?

- Rescaling your data makes the distance metric more meaningful. For instance, given 2 features height and weight, an observation such as x=[180,70] will clearly skew the distance metric in favor of height. One way of fixing this is by column-wise subtracting the mean and dividing by the standard deviation. Scikit-learn’s normalize() method can come in handy. would this improve the results? How can you do this in Sklearn?

- Use Dimensionality reduction techniques like PCA:

	[Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
		
	```
	from sklearn.decomposition import PCA
	```

-  should be executed prior to appplying KNN and help make the distance metric more meaningful.


After applying and testing for these diagnostics, how the model performance has improved? What is the final confusion matric that you can share? What is overall accuracy of the model?



	


 
 
 
