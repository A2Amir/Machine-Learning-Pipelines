#  Machine Learning  Pipelines
 
In this Repo, I will build pipelines to automate my machine learning workflows that train on the data and you will learn about:

* The advantages of using machine learning pipelines

* Using Scikit-learn's Pipeline and Feature Union classes

* Grid search over the entire workflow.


In this Repo, I will automate my machine-learning workflows with pipelines using [the dataset of corporate messaging](https://github.com/A2Amir/Machine-Learning-Pipelines/blob/master/dataset/corporate_messaging.csv) as a case study. This corporate message data is from one of the free datasets provided on the [Figure Eight Platform](https://appen.com/resources/datasets/),  Each row of the dataset contains information about a social media post from different corporations and contains a column for the category of the post. In the category column, we can see that **each post is classified as information (objective statements about the company or its activities), action (such as messages that ask for votes or ask users to click on links), dialogue (replies to users) or exclude (a miscellaneous column).** 


## Tokenization

Before I can classify any posts, I'll need to clean and tokenize the text data. Use what I learned from [the last Repo on NLP](https://github.com/A2Amir/NLP-and-Pipelines) to implement the function `tokenize`. Check out [this jupyter notebook](https://github.com/A2Amir/Machine-Learning-Pipelines/blob/master/Code/1_clean_tokenize.ipynb) to tokenize the text data.


## Machine Learning Workflow without pipelines
Now I 've cleaned and tokenized the text data, it's time to complete the rest of my Machine Learning workflow. In [this jupyter notebook](https://github.com/A2Amir/Machine-Learning-Pipelines/blob/master/Code/2_ml_workflow.ipynb) I first build out the normal way (without pipelines) to create a classifier then test my classifier. After buildin and testing my classifier I am going to refactor these steps into the two functions.

## Machine Learning Workflow with pipelines

Below is a simple example, where we generate features from text data using **CountVectoriser and TfidfTransformer** and then fit it to a **RandomForestClassifier**.

~~~python

      
    # Instantiate transformers and classifier (initialize step) 
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    clf =  RandomForestClassifier()

    # Fit and/or transform each to the data (training step)

    X_train_count = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_count)
    clf.fit(X_train_tfidf, y_train)
    
    # Transform test data (test step)
    
    X_test_count = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_count)
~~~

Each of these three objects (**CountVectoriser, TfidfTransformer and RandomForestClassifier**) is called an **estimator**, which [scikit-learn states](https://scikit-learn.org/stable/tutorial/statistical_inference/settings.html), is any object that learns from data, whether it's a classification, regression or clustering algorithm, or a transformer that extracts or filter is useful features from raw data. Because it learns from data, every estimator must have a fit method that takes a dataset. So, each of these three estimators have a fit method.  Additionally, **CountVectoriser and TfidfTransformer**, are a specific type of estimator called a **transformer**, meaning it has a transform method. 

 The final estimator(**RandomForestClassifier**) is a predictor, which although it doesn't have a transform method,has a predictor method. 
 
 
In machine learning tasks, it's pretty common to have a very specific sequence of transformers to fit to data before applying a final estimator.  As above, we'd have to initialize all the tree estimators (initialize step) , fit and transform the training data for each of the transformers and then fit to the final estimator (training step). In the test step we  have to call transform for each transformer again to the test data and finally call predict on the final estimator. 


We could actually automate all of this fitting, transforming and predicting by chaining these estimators together into one single estimator object. That single estimator would be **scikit-learns pipeline**. 



To create this pipeline, we just need a list of key value pairs. Where the key is a string containing what you want to name the step and the value is the estimator object. 

~~~python
#(initialize step) 
pipline = Pipline(['vect' : CountVectorizer(),
                  'tfidf' : TfidfTransformer(),
                  'clf' : RandomForestClassifier()]
~~~

By fitting our pipeline to the training data, we're accomplishing exactly what we would by fitting and transforming each of the objects of the training step to our training data one by one. 


~~~python
#(training step)
pipline.fit(X_train)
~~~

Similarly, when we call predict on our pipeline to our test data, we're accomplishing what we would by calling transform on each of
our transformer objects of the test step to our test data and then calling predict on our final estimator. 


~~~python
#(test step)
pipline.predict(X_test)
~~~

Not only does pipline make our code is so much shorter and simpler, it has other great advantages
