# loss function: are we doing better or worse? we want a low number
# numpy: matrix language
# scikit not good with statistics, good with prediction, instead use statsmodels
# always care about out-of-sample performance (scikit), out of sample testing > in sample performance
# r^2 proportion of variance that our model is explaining
# but adjusted still better because than normal r^2, which makes overfitting look good
# bootstrap: take data, create a few smaller samples with replacement
# random forest model does not like missing data, we can drop missing values, change missing to mean, bad idea but for larger datasets won't make much difference
# precision allows for false positives
# recall allows for false negatives (check 0.3 random_forest
# precision = if I make a positive prediction, how often do I get it right
# recall = how often is the result of interest brought back
# cross validation very costly on large datasets, but do if you can to ensure our model gives us what is should 


# good way of knowing models: test by generating random sample data
np.random.seed(0)
N = 500
X = 2 * np.random.rand(N, 1)
y = 4 + 3 * X + np.random.randn(N, 1)

# LinearRegression is a class, this is a standard procedure
model = LinearRegression()
fit = model.fit(X, y)

# mean when value of x is 0
intercept = 3.885506032525154



# Hyperparameter Optimisation
# Why: To get a better fitting model, every model has hyperparameters
# How: optimise important ones, keep others at default to save time
# Trick; to determine which ones are important

# GridSearchCV = Cross Validation
scoring='f1' # average
n_jobs=4 # 4 cores on a computer, perform 4 jobs at the same time

# Extreme boosted trees (XGBoost)
# Boosting model with lower bias, more more models to try to learn what the previous model missed
# Bagging best fitting model, more likely to overfit, out of sample performance critical


# Most important parameter, learning rate
learning_rate': hp.loguniform('learning_rate', -5, -2)
