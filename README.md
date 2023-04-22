# **Mini project for SC1015, Group 1**

## About our team and contributions
- Tan Chuan You *(Data Cleaning, Data Visualization/EDA, Logistic Regression, Decision Tree)*
- Divya Gupta *(Dataset selection, Logistic Regression, README file, Presentation Slides)*
- Tan Jun Liang *(Data Cleaning, Data Visualization/EDA, README file, Video creation and editing)*

## Deliverables

### Dataset
A dataset from kaggle that where we extract our data from.

### Presentation Slides
A summarised content towards our progress in the project.

### Jupyter Notebook #1: Data Extraction, Data Cleaning
Imports all necessary libaries. We take a general view (such as min-max values,and distribution of data using boxplot) at our uncleaned data to spot anomalies or null values. From there, we clean out the anomalies and outliers to improve the accuracy of our predictors by narrowing the spread of data.

### Jupyter Notebook #2: EDA/Visualisations/Correlations
We utilise the module "plotly" in creating more detailed graphs that help us observe trends and data patterns better. We display and compare percentages, looking at the distribution of diabetics among data groups to spot trends and patterns that can bring us useful insights. As we evaluate our first set of graphs, we realised that our data was imbalanced at 0.9:9.1 (Diabetics:Non-Diabetics. Hence, we split the data into train and test sets of ratio 8:2, use oversampling techniques (SMOTE + TOMEK) on the train dataset ONLY to 3:7 (Diabetics:Non-Diabetics). This reduction in imbalance allows us to better train our model and gain better accuracy while also retaining the reliability of our data. By examining correlations and graphs from the sampled data, we choose the best predictors to train our machine-learning models. 

`Chosen Predictors: age, HbA1c_level, bmi and blood_gluclose_level`

### Jupyter Notebook #3: Machine Learning Model 1 - Logistic Regression & Fine-tuning
We chose Logistic Regression to learn something new. The outcome derived from the model is binary, which fits our response variable 'diabetes' with the values 0 and 1. Logistic Regression has a performance matric that accounts for imbalance data issue which is preferred for our data as there still exists minor imbalance. Furthermore, this model can evaluate both linear and non-linear data which allows for more complex relationships to be captured. First, we fine-tune the model with gridsearch that returns the best hyperparameters needed for our train datasets. Secondly, we train the model with our train and test dataset to find the coeeficients of each variable and plot the confusion matrix based on the Logistic Regression Model. Lastly, we evaluate the model's accuracy through Receiver Operating Characteristic(ROC) which takes into account of imbalance data. We also further evaluate through information such as accuracy, precision, F1-score, support and recall generated in a classification report.

### Jupyter Notebook #4: Machine Learning Model 2 - Decision Tree & Fine-tuning
Applying our knowledge in our syllabus, we explore the use of Decision Trees for our predictors. A big reason is that decision tree captures non-linear relationships between predictor variables and outcome, evaluating and presenting complex relationships between our predictors and response variable. With our huge data of 124k rows, decision tree is suitable as it can be scaled to handle larger data.  First, we fine-tune the model with gridsearch that returns the best hyperparameters needed for our train datasets. Secondly, we plot a binary tree with our train and test dataset to find the coeeficients of each variable and plot the confusion matrix based on the Logistic Regression Model. Lastly, we evaluate the model's accuracy through Receiver Operating Characteristic(ROC) which takes into account of imbalance data. We also further evaluate through information such as accuracy, precision, F1-score, support and recall generated in a classification report.

### Jupyter Notebook #5: Gradient Boosting Classifier
This model allows us to further test on the non-linear aspect of our data as the non-linear Decision Tree gave us a better result for our data.

## Problem Definition
- What are the top 3 predictor for the response variable 'diabetes'?
- What "out of syllabus" techniques can we apply to our dataset?
- What models can best fit our data?
- 
## About dataset
### Title: diabetes_prediction_dataset
> 9 Columns - gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes

1) gender *[Male / Female]*
2) age *[0.08-80]* 
3) hypertension *[0,1] (0 - False, 1 - True)*
    - `(Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated.)`
4) heart_disease *[0,1] (0 - False, 1 - True)*
5) smoking_history *[No info, never, not current, former, ever, current]* 
    - `No info - No information regarding smoking history`
    - `Never - Never smoked before`
    - `Not current - Quit smoking for more than 12 months`
    - `Former - Quit smoking within the past 12 months`
    - `Ever - Has a smoking history regardless of whether current smoking or not`
    - `Current - Currently smoking`
6) bmi *[10.01-95.69]* 
    - `(BMI (Body Mass Index) is a measure of body fat based on weight and height.)`
7) HbA1c_level *[3.5-9.0]* 
    - `(HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months)`
8) blood_glucose_level *[80-300]* 
    - `(Blood glucose level refers to the amount of glucose in the bloodstream at a given time. )`
9) diabetes *[0,1]* 
    - `(Diabetes is the target variable being predicted)`

## Algorithms/Libraries used
**In syllabus**
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Decision Tree
- Sklearn

**Out of syllabus**
- Plotly
- SMOTE (Oversampling)
- Logistic Regression
- GrideSearch (FineTuning Hyper parameters)
- load_iris (Fine Tuning)


## Conclusion of insights
- The top 3 predictors of the response variable are " ", " ", and " "
- We were able to identify the best predictor as ""
- The "what" machine learning model provided a better and more accurate fit for our data compared to "what"
- Our fine-tuning methods were effective/uneffective/not that effective

## What has the team learnt from the project?
1) There are many available algorithms and techniques for us to utilise and it highly depends on many factors of the dataset such as size, skew, type of variables, and others
2) Previously, our team had the idea that any machine learning model would work fine for any dataset due to the lack of experience in building one ourselves. This project has corrected our wrong perceptions and given us insightful lessons.
3) We can apply the insights from our findings in our lives that one should always be careful of the fluctuations HbA1c_level. Being negligent in maintaining the levels can result in high possibilities of one developing diabetes. Despite HbA1c_level being the best predictor, we believe that Blood glucose level and BMI are factors that one should take into consideration as well.

## References
- s
