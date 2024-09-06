
# Model Selection Project

This project focuses on model selection and evaluation, applying various machine learning algorithms to a dataset with both continuous and categorical features. The goal was to determine the best model configuration by evaluating different hyperparameter settings and then assess the model's performance on unseen data.

## Project Overview

## 1. Data Exploration and Feature Classification

The idea is to classify the features of the dataset as either categorical or continuous, given that no prior information was available regarding the data types. All features consisted of integer values, which added complexity to the task of determining their nature. To address this, we adopted a systematic approach based on the distribution characteristics of each feature, as illustrated in Figures 1 and 2.

#### Figure 1: Feature Distribution Analysis

Each histogram in Figure 1 visualizes the distribution of values for a specific feature, highlighting the number of unique values and the variance. These characteristics were crucial in classifying the features:

* Binary Categorical: Features with exactly two unique values were classified as binary categorical.
*  Continuous: Features with smooth, bell-shaped distributions and higher variance were classified as continuous.
*   Categorical: Features with distinct clusters or a limited number of unique values were classified as categorical, especially those with low variance.

This detailed analysis allowed us to move beyond simple assumptions and classify features based on their inherent characteristics.

![image](https://github.com/user-attachments/assets/50c21abc-c1ff-40a2-99ed-0362ae5eff17)


#### Figure 2: Threshold and Distribution Shape Analysis

In Figure 2, we explore the relationship between the number of unique values and the distribution shape across different thresholds. This scatterplot analysis was pivotal in determining the optimal threshold for classifying features as either categorical or continuous. We empirically tested various thresholds and found that a threshold of 9 unique values provided the highest agreement between the distribution-based classification and the unique value count.

Conclusion:

* Features with fewer than 9 unique values were classified as categorical.
* Features with 9 or more unique values were classified as continuous.

This method ensured that our feature classification was both data-driven and robust, allowing for more accurate modeling in subsequent analyses.

![image](https://github.com/user-attachments/assets/7cae3887-c8b8-4a3c-9957-a7d517a536a4)

### Data Exploration - Dependence Between Features and Target Variable Y

In this phase of the analysis, we aimed to explore the relationship between the features and the target variable Y. The approach varied based on whether the features were continuous or categorical.

#### Continuous Features (Figure 3)

To determine the correlation between continuous features and the target variable Y, we computed the pairwise correlation coefficients using the `corrwith` method. The strength of these correlations was categorized as follows:

* Weak correlation: |correlation coefficient| between 0.1 and 0.3.
* Moderate correlation: |correlation coefficient| between 0.3 and 0.5.
* Strong correlation: |correlation coefficient| greater than 0.5.

As shown in Figure 3, most continuous features exhibited weak correlations with the target variable Y. Notably:

* Feature 4 showed a moderate positive correlation with Y (correlation coefficient = 0.29908).
* Feature 7 had a moderate negative correlation with Y (correlation coefficient = -0.21082).
* 
The remaining features had extremely weak correlations, indicating minimal linear relationships with the target variable.

![image](https://github.com/user-attachments/assets/260fdb8b-1382-4d97-bfca-70575b40ea9f)

#### Categorical Features (Figure 4)

To assess the dependence of categorical features on the target variable Y, we performed the Chi-squared test, setting a significance threshold of p<0.05. Features with p-values below this threshold were considered dependent on Y.

Figure 4a depicts the distribution of dependent features. The count plots, enhanced with Kernel Density Estimation (KDE) lines, show clear differences in the feature distributions between the two classes of Y. These variations suggest that these features are closely related to the target variable.

Figure 4b illustrates the distribution of independent features. Here, the KDE lines are nearly identical between the two classes, indicating that these features do not significantly differ between the classes and, thus, are less likely to be related to Y.

#### Conclusion:

Continuous Features: Most showed weak correlations with Y, with a few exceptions demonstrating moderate relationships.
Categorical Features: Dependent features displayed distinct distribution patterns across different Y classes, highlighting their potential significance in predicting Y. In contrast, independent features showed no such distinction, implying a lack of connection to Y.

These insights are crucial for feature selection, helping to identify which variables are most likely to contribute to predictive modeling.


![image](https://github.com/user-attachments/assets/f5f20445-7fcb-427b-b1e2-582b7068601e)


### Data Exploration - Interesting Feature Selection
In this section, we carefully selected a subset of features from the dataset based on their statistical properties and their relationship with the target variable Y. The goal was to choose features that would enhance the model's ability to recognize patterns and make accurate predictions.

#### Categorical Features

We focused on categorical features that demonstrated a strong dependence on the target variable Y, as indicated by the Chi-square test (p-value < 0.05). These features exhibited significant differences in distribution across different classes of Y, as seen in their respective histograms:

    Feature 5: This feature shows substantial variability in its class distribution and significant differences between the Kernel Density Estimation (KDE) curves for different classes of Y, making it a strong candidate for capturing class-dependent patterns.
    Feature 8: Similar to Feature 5, Feature 8 exhibits distinct class patterns, with KDE lines that differ noticeably between Y=1 and Y=2, indicating its relevance in distinguishing between these classes.
    Feature 13: This feature shows high variability among classes and clear distinctions between the KDEs of Y=1 and Y=2. Its dependence on Y suggests that it can effectively capture distinct class patterns.

#### Continuous Features

For continuous features, we selected those with meaningful distribution characteristics:

    Feature 43: Exhibits a normal distribution pattern, with a moderate number of unique values and a reasonable variance. This feature could contribute to good predictive accuracy, especially in models that assume normal distributions.
    Feature 21: Displays a uniform distribution, which is less sensitive to outliers and extreme values. This feature adds robustness to the model, providing a different perspective on the data and complementing the normal distribution of Feature 43.

Conclusion: 
By selecting these features—three categorical and two continuous—we aim to create a balanced mix that captures the diversity in the data. The categorical features are expected to enhance the model's ability to identify distinct patterns associated with the target variable Y, while the continuous features contribute smooth and robust information. This strategic selection is intended to improve the overall accuracy and effectiveness of the model in recognizing patterns and making predictions.

![image](https://github.com/user-attachments/assets/13bd1114-c690-4553-980f-0fa0b765a6ff)
![image](https://github.com/user-attachments/assets/cc7f8405-b719-41c2-ab74-7a811797a33b)

## Model Selection

### Three classifiers were selected: 

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest.

Various hyperparameters were tested for each classifier to build a diverse range of model configurations.

* Logistic Regression: Tested with different regularization strengths (C), penalties (l2 or None), and iterations.
* KNN: Evaluated using different numbers of neighbors (n_neighbors) and weighting methods (uniform, distance).
* Random Forest: Tested with varying numbers of trees (n_estimators) and tree depths (max_depth).

The models were trained and validated using cross-validation with the ROC AUC score as the performance metric. The best-performing model configuration was selected based on the highest average ROC AUC score across the validation folds.


![image](https://github.com/user-attachments/assets/23a3b179-0f44-4ddd-8cf8-645073233a34)

Bars with purple color correspond to Logistic Regression classifier, bars with blue color correspond to KNN classifier and bars with green color correspond to Random Forest classifier. In the body of each bar are shown the hyperparameter combinations. The best model configuration is highlighted with a black outline. 

## Out-of-Sample Performance 
* The best model was then evaluated on a hold-out test set to assess its out-of-sample performance. The ROC curve was generated and compared against a naive classifier that always predicts the majority class, as well as a random classifier.
* The ROC AUC score of the model on the test set was slightly lower than the cross-validation score, indicating a small degree of overfitting but generally robust performance.
  
#### Figure 6
The ROC curve of the best model (solid line) achieved an AUC of 0.87, demonstrating strong predictive accuracy. This model significantly outperformed the random classifier (dashed line, AUC = 0.5) and the naive classifier (dot, with no predictive value).

![image](https://github.com/user-attachments/assets/44ff728c-2036-4a7c-a260-514e99abdb17)


## Bootstrap Confidence Intervals

Bootstrap sampling was performed on the test set to estimate the 95% confidence intervals of the model's ROC AUC score. This analysis provided insights into the stability and reliability of the model's performance. By evaluating the model's performance on these bootstrapped datasets, we can gain insight into how the model would perform on unseen data.

#### Figure 7

Figure 7 illustrates the distribution of ROC AUC scores obtained from 1,000 bootstrapped samples. The histogram provides a clear visual representation of the variability in the model's performance. The dashed lines mark the 2.5th and 97.5th percentiles, which define the 95% Confidence Interval for the ROC AUC score.

    Lower Bound (2.5th percentile): This indicates the score below which only 2.5% of the bootstrapped ROC AUC scores fall. It represents a conservative estimate of the model's performance.

    Upper Bound (97.5th percentile): This indicates the score above which only 2.5% of the bootstrapped ROC AUC scores fall. It represents an optimistic estimate of the model's performance.

The results show that the model's ROC AUC score is consistently high, with a 95% Confidence Interval that suggests a stable and reliable performance. The confidence interval provides reassurance that the model's true performance lies within this range. 


![image](https://github.com/user-attachments/assets/467e013f-12e6-440a-9311-ab0fd3f0a4d1)

This analysis underscores the model's ability to generalize well to new, unseen data, which is critical for its practical application.















