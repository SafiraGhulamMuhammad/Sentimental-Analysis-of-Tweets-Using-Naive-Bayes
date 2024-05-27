# Sentiment Analysis Using Naive Bayes Rule

This project performs sentiment analysis using the Naive Bayes Rule. Although Logistic Regression works well for this task, Naive Bayes is chosen for its ability to handle conditional probabilities. For instance, it can focus on tweets containing specific words like "happy." 

## Steps Involved

### Data Preprocessing
1. **Remove Retweets, Hashtags, and URLs:** Clean the data by removing unnecessary elements.
2. **Convert to Lowercase:** Ensure consistency.
3. **Tokenize:** Split text into individual words.
4. **Remove Stopwords and Punctuation:** Eliminate irrelevant words.
5. **Stemming:** Reduce words to their root form (e.g., 'running' to 'run').

### Applying Naive Bayes
1. **Calculate Class Probabilities:**
   - \( P(D_{pos}) \): Probability that the document is positive.
   - \( P(D_{neg}) \): Probability that the document is negative.
2. **Log Prior:** Calculate the ratio of positive to negative probabilities and rescale using log.
3. **Laplacian Smoothing:** Prevent zero probabilities by adjusting word frequency calculations.
4. **Log Likelihood:** Compute the likelihood of words given each class.

### Prediction
- **Implement `naive_bayes_predict`:** Sum the log likelihoods of words in a tweet and add the log prior to predict sentiment.
- **Test Predictions:** Use the function to classify tweets based on the trained model.
