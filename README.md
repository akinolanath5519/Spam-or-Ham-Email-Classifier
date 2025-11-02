# Spam/Ham Email Classifier

This project demonstrates a simple **Natural Language Processing (NLP) pipeline** to classify emails/messages as **spam** or **ham** (not spam). It uses Python, scikit-learn, and standard text preprocessing techniques.

---

## **Steps Taken**

1. **Dataset Loading**
   - Loaded the spam/ham dataset containing 5,572 messages.
   - Checked for missing values and cleaned the dataset.

2. **Text Preprocessing**
   - **Lowercasing:** Convert all text to lowercase for consistency.
   - **Remove special characters:** Keep only letters for clean text.
   - **Tokenization:** Split text into individual words (tokens).
   - **Stopword removal:** Remove common words that do not add meaning.
   - **Lemmatization:** Reduce words to their base form (e.g., "running" → "run").
   - Converted the tokens back into a single cleaned string for vectorization.

3. **Feature Extraction**
   - **Bag-of-Words (BoW):** Represented text as word counts.
   - **TF-IDF:** Represented text with term frequency–inverse document frequency to weigh important words higher.

4. **Train/Test Split**
   - Split data into **80% training** and **20% testing** sets.

5. **Model Training**
   - Used **Multinomial Naive Bayes** classifier.
   - Trained separate models for **BoW** and **TF-IDF** features.

6. **Handling Class Imbalance**
   - Dataset was imbalanced: Ham = 4825, Spam = 747.
   - Oversampled the minority class (Spam) to balance the dataset for better performance.

7. **Evaluation**
   - Measured **accuracy, precision, recall, and F1-score** for both models.
   - Tested with new example messages to check generalization.

---

## **Results**

### Bag-of-Words
- **Accuracy:** 98%  
- **Spam Recall:** 0.89 → Caught 89% of all spam messages.  
- **Spam Precision:** 0.96 → Most predicted spam were correct.  
- **Conclusion:** BoW performed very well and is slightly better for real-world spam detection because it catches more spam.

### TF-IDF
- **Accuracy:** 96.7%  
- **Spam Recall:** 0.75 → Missed 25% of spam messages.  
- **Spam Precision:** 1.00 → All predicted spam were correct.  
- **Conclusion:** TF-IDF is very precise but misses some spam, making it safer but less sensitive than BoW.

---

## **Interactive Testing**
- Users can input custom messages and the model predicts **spam** or **ham**.
- Example messages:
  - Ham: `"Hey, are we still meeting for lunch today?"`
  - Spam: `"Congratulations! You have won a $1000 gift card. Click here to claim now."`

---

## **Conclusion**
- **Preprocessing steps** (lowercase, remove special characters, stopwords removal, lemmatization) improved text quality for modeling.
- **Bag-of-Words** slightly outperforms TF-IDF in spam detection due to higher recall.
- The trained model can generalize well to new messages and is suitable for **real-world spam filtering**.
- Handling **class imbalance** is important for accurate spam detection.
- This project demonstrates a simple yet effective **end-to-end NLP pipeline** for text classification.

---

## **Next Steps / Improvements**
- Experiment with **Word2Vec, FastText, or BERT embeddings** for better semantic understanding.
- Deploy as a **real-time spam filter** in email or chat applications.
- Use **cross-validation** and **hyperparameter tuning** to further improve performance.
