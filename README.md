#  Machine Learning  Pipelines
 
In this Repo, I will build pipelines to automate my machine learning workflows that train on the data and you will learn about:

* The advantages of using machine learning pipelines

* Using Scikit-learn's Pipeline and Feature Union classes

* Grid search over the entire workflow.


In this Repo, I will automate my machine-learning workflows with pipelines using [the dataset of corporate messaging](https://github.com/A2Amir/Machine-Learning-Pipelines/blob/master/dataset/corporate_messaging.csv) as a case study. This corporate message data is from one of the free datasets provided on the [Figure Eight Platform](https://appen.com/resources/datasets/),  Each row of the dataset contains information about a social media post from different corporations and contains a column for the category of the post. In the category column, we can see that **each post is classified as information (objective statements about the company or its activities), action (such as messages that ask for votes or ask users to click on links), dialogue (replies to users) or exclude (a miscellaneous column).** 


## Tokenization

Before I can classify any posts, I'll need to clean and tokenize the text data. Use what I learned from [the last Repo on NLP](https://github.com/A2Amir/NLP-and-Pipelines) to implement the function `tokenize`. Check out [this jupyter notebook](https://github.com/A2Amir/Machine-Learning-Pipelines/blob/master/Code/1_clean_tokenize.ipynb) to tokenize the text data.
