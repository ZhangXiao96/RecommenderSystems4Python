# Recommender Systems for Python

In this repository, we implement several models applied in recommender systems, including traditional ones and deep ones.
This repository references to the book ["Deep Learning Recommender System" by Zhe Wang (《深度学习推荐系统/王喆》)](https://item.jd.com/12630209.html?cu=true&utm_source=link.zhihu.com&utm_medium=tuiguang&utm_campaign=t_1001542270_12769_0_2045956751&utm_term=9d367a3329bb44f3b64dfa03ccb73eae) and you can treat this repository as a study notes of recommender systems.

The dataset we test on is [MovieLens/ml-latest-small](https://grouplens.org/datasets/movielens/latest/). More detailed information can be found in [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/).

If you have any question, please send email to *xiao_zhang@hust.edu.cn*.

## Traditional Recommender Systems

### [Collaborative Filtering](./TraditionalRecommenderSystems/CFDemo.py)

We implement **ItemCF** and **UserCF** based on the co-occurrence matrix instead of graph, which is much faster but less memory-friendly.

#### Parameters
|Model|nb_similar_user|nb_similar_item|Fill NA|Similarity|Top K|
|----|----|----|----|----|----|
|UserCF|20|None|0|cos|10|
|ItemCF|20|None|0|cos|10|

#### Results

|Model|Test MSE|Precision(%)|Recall(%)|
|----|----|----|----|
|UserCF|7.11|17.7|18.6|
|ItemCF|4.00|16.0|16.9|

#### Tips
1. In practice, the number of users is usually much larger than items, which means ItemCF is usually more friendly because the item similarity-matrix is much smaller than the user-similarity matrix.
2. The behavior matrix of users is usually highly sparse, hence accurately searching the similar users can be hard.
3. The basic idea of UserCF is that similar people share similar interests, so it is usually utilized in some situations with social properties, such as news recommendation systems. UserCF is good at tracking hot spots.
4. ItemCF is usually applied in the situation when the interests of users are stable in a while, such as e-commerce and video recommendation.
5. The basic ItemCF and UserCF do not efficiently utilize some other information, such as the information of users or the descriptions of items.
6. The topK of ItemCF is based on Scores calculated instead of the predict rating.

### [Matrix Factorization](./TraditionalRecommenderSystems/MFDemo.py)

We implement Matrix Factorization based on PyTorch.

#### Parameters

|nb_factor|Optimizer|lr|Weight Decay|Epochs|Batch Size|Drop Rate|Top K|
|----|----|----|----|----|----|----|----|
|80|Adam|1e-3|1e-6|80|64|0.2|10|

#### Results
|Model|Train MSE|Test MSE|Precision(%)|Recall(%)|
|----|----|----|----|----|
|Base MF| 0.20| 0.87|0.23|0.07|

#### Tips
1. Using Matrix Factorization can be more memory-friendly because the features of every user and item can be represented as a latent vector, which is usually much smaller than the rating matrix.
2. The test MSE loss of Matrix Factorization is much less than CF.
3. Note: here the recall and precision of the recommended movies is not based on the rating but whether the user rates them.

### [Logistic Regression](./TraditionalRecommenderSystems/LRDemo.py)
We construct simple feature vectors and implement the recommender system based on the Logistic Regression model in Sklearn.

#### Parameters

|Optimizer|lr|Weight Decay|Epochs|Batch Size|Top K|
|----|----|----|----|----|----|
|||||||

#### Results
|Model|Train MSE|Test MSE|Precision(%)|Recall(%)|
|----|----|----|----|----|
|Logistic Regression|||||

#### Tips
The key to Logistic Regression is the construction of feature vectors, which directly decides the performance of the model. Since the construction of feature vectors varies according to datasets, here we only provide a simple example for building feature vectors which will be sent to Logistic Regression in sklearn.

### Factorization Machine (TODO)

## Deep Recommender Systems (TODO)