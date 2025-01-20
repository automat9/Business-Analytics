# loss function: are we doing better or worse? we want a low number
# numpy: matrix language
# scikit not good with statistics, good with prediction, instead use statsmodels
# always care about out-of-sample performance (scikit), out of sample testing > in sample performance
# r^2 proportion of variance that our model is explaining
# but adjusted still better because than normal r^2, which makes overfitting look good



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

