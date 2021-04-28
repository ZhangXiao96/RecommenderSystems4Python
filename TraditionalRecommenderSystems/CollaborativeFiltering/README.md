# Collaborative Filtering (CF)

We implement ItemCF and UserCF based on the co-occurrence matrix instead of graph, which is much faster but less memory-friendly.
## Tips
1. In practice, the number of users is usually much larger than items, which means ItemCF is usually more friendly because the item similarity-matrix is much smaller than the user-similarity matrix.
2. The behavior matrix of users is usually highly sparse, hence accurately searching the similar users can be hard.
3. The basic idea of UserCF is that similar people share similar interests, so it is usually utilized in some situations with social properties, such as news recommendation systems. UserCF is good at tracking hot spots.
4. ItemCF is usually applied in the situation when the interests of users are stable in a while, such as e-commerce and video recommendation.
5. The basic ItemCF and UserCF do not efficiently utilize some other information, such as the information of users or the descriptions of items.