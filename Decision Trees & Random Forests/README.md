# Random Forest vs Decision Trees 

## Overview

Implementation of a Random Forest classifier and its comparison to individual Decision Trees. The main goal is to explore how the ensemble method of Random Forest improves prediction accuracy compared to single decision trees, particularly under different configurations of the min_samples_leaf hyperparameter.

## Methodology

TrainRF Function: 
Trains a Random Forest model by growing multiple decision trees, each trained on a bootstrap sample of the data. The min_samples_leaf parameter controls the minimum number of samples required to split an internal node.

### PredictRF Function: 
Uses the trained Random Forest model to predict class labels for new data by averaging the predictions of individual trees.

### Model Evaluation: 
The accuracy of each individual decision tree is compared to the overall accuracy of the Random Forest. Multiple models are trained with different min_samples_leaf values to observe their impact on model performance.

### Bonus Analysis - Observations - Conclusions: 
Evaluates the probability of a single decision tree achieving accuracy similar to or better than the Random Forest by training a large number of individual trees and comparing their accuracies.

## Conclusion - Discussion

Ensemble Advantage: The Random Forest method significantly outperforms individual decision trees in terms of accuracy. This is because Random Forest combines the predictions of multiple decision trees, which helps to balance out the errors that might arise from overfitting or underfitting in any single tree. By averaging the results from many trees, Random Forest provides a more stable and reliable prediction.

### Impact of min_samples_leaf:

When min_samples_leaf is set to a smaller value (e.g., 1), individual decision trees tend to have higher variability in their accuracy. This variability results from the trees overfitting to the training data, capturing noise rather than the underlying patterns. However, Random Forest mitigates this issue by pooling the predictions, leading to a more accurate overall model.

With a larger min_samples_leaf (e.g., 10), trees become simpler, with less overfitting. This reduces variability, but it also risks underfitting, where the model may miss some important details in the data. Despite this, Random Forest still outperforms individual trees by providing a better generalization through ensemble learning.

### Probability of Single Tree Matching Forest Accuracy: 
Fig1 shows the accuracy comparison between individual decision trees and the Random Forest for different min_samples_leaf settings:

* min_samples_leaf=1 (left): Individual trees show more variability in accuracy, with some performing poorly. This is because smaller leaf sizes lead to overfitting. However, the Random Forest significantly outperforms these individual trees by averaging their predictions.

* min_samples_leaf=10 (right): With a higher min_samples_leaf, individual tree accuracies are more consistent, leading to a smaller gap between the average tree accuracy and the Random Forest accuracy. Even so, the Random Forest still achieves better overall performance, demonstrating its robustness in handling different configurations.

![image](https://github.com/user-attachments/assets/29fe787d-4394-404f-bd78-4ae35a4d1f94)


As shown in Fig2, the accuracy of individual decision trees varies widely, with some performing much better than others. However, the overall Random Forest accuracy consistently surpasses the average accuracy of individual trees. Fig2 shows the distribution of accuracies for 1000 individual decision trees compared to the Random Forest's accuracy. The histogram highlights that while individual tree accuracies vary widely, the Random Forest's accuracy is consistently higher, demonstrating the advantage of using an ensemble approach.

![image](https://github.com/user-attachments/assets/c8f6edda-cf32-4f05-8f2d-a2ece4542c5c)

Fig3 illustrates that the probability of a single tree matching or exceeding the accuracy of the entire Random Forest is very low. This emphasizes the strength of the Random Forest approach, as it leverages the collective wisdom of multiple trees to produce more consistent and higher accuracy predictions. Fig3 presents the probability distribution of achieving an accuracy similar to or better than the Random Forest with a single decision tree across 100 iterations. The results indicate that it is quite unlikely for any single tree to match the Random Forest's accuracy, further confirming the benefits of ensemble learning.

![image](https://github.com/user-attachments/assets/401258f9-55ff-443b-921e-b346625a9a1f)
