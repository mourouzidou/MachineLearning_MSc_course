# Naïve Bayes Classifier 
## Implementation of `train` and `prediciton` functions - Model Assesment and Hyperparameter Tuning

### Overview

Implementation of a Naïve Bayes Classifier (NBC) for both categorical and continuous data. The classifier is trained on an input dataset to predict class labels for new samples.
Key Components:

  Training Function: Calculates probabilities for classification, using Laplace smoothing for categorical data and Gaussian assumptions for continuous data.
  Prediction Function: Predicts class labels based on the trained model by evaluating the likelihood of each sample belonging to each class.
  Model Evaluation: Assesses classifier accuracy over multiple iterations with different data subsets to ensure reliable performance.

### Conclusion

Impact of Hyperparameter Tuning (Laplace Smoothing)

The choice of the Laplace smoothing parameter L significantly affects the performance of the Naïve Bayes Classifier in categorical data classification. When L is small (e.g., L=1), the model tends to fit closely to the training data, resulting in higher accuracy on known data but increasing the risk of overfitting. This means the model may not perform well on unseen data.

In contrast, increasing L (e.g., L=1000) enhances the model's ability to generalize by smoothing the probability estimates, thus reducing overfitting. However, this also leads to a decrease in accuracy on the training data as the model becomes too simplistic, potentially underfitting the data.

It is important to choose an appropriate value of L to balance the trade-off between overfitting and underfitting, ensuring the model performs well on both the training data and new, unseen datasets.
