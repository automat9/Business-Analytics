from sklearn.feature_extraction.text import CountVectorizer

corpus = ["The cat sat on the mat.", "The dog chased the cat.", "The dog chased a ball with his dog friends."]

# Vectorize the corpus
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
print(X.toarray())

# Prints how many times words occured in a body of text

# Output:
['ball' 'cat' 'chased' 'dog' 'friends' 'his' 'mat' 'on' 'sat' 'the' 'with']
[[0 1 0 0 0 0 1 1 1 2 0]
 [0 1 1 1 0 0 0 0 0 2 0]
 [1 0 1 2 1 1 0 0 0 1 1]]
