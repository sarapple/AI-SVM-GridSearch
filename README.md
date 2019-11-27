# SVM

## Setup

- Split data into training (60%) and testing (40%).
- Use stratified sampling (same ratio of positive and negative in training and datasets)
- Use cross validation (folds k = 5)
- Train-test splitting and cross validation are in sklearn

## Kernels

Use SVM with different kernels to build a classifier

### SVM with Linear Kernel
Search for a good setting of parameters to obtain high classification accuracy. Specifically, try values of C = [0.1, 0.5, 1, 5, 10, 50, 100]

### SVM with Polynomial Kernel
Try values of C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 0.5].

### SVM with RBF Kernel
Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10].

### Logistic Regression
Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100].

### k-Nearest Neighbors.
Try values of n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60].

### Decision Trees
Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].

### Random Forest
Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].

# Run the program

Run the program with:
```
python3 problem3.py input3.csv output3.csv
```

# Output

The output is a `output3.csv` csv in the following format:

svm_linear,best_score,test_score
svm_polynomial,best_score,test_score
svm_rbf,best_score,test_score
logistic,best_score,test_score
knn,best_score,test_score
decision_tree,best_score,test_score
random_forest,best_score,test_score

# Run

To run, execute the following:

```
python3 problem3.py input3.csv output3.csv
```

