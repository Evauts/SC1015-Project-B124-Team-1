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
In further understanding our cleaned data, we utilise the module "plotly" in creating graphs to help us observe trends and data patterns better. Taking advantage of data such as percentages, we look at the distribution of diabetics among data groups with the to spot trends and habits in bringing us useful insights. By comparing correlations and visual data from the graphs,  we choose the best predictors that we can use to train our machine-learning models. 

### Jupyter Notebook #3: Machine Learning Model 1 - Logistic Regression & Fine-tuning
We chose Logistic Regression to learn something new. Additionally, the outcome derived from the model is binary, which fits our response variable 'diabetes' with the values 0 and 1. As our dataset is imbalance, this machine learning model has a performance matric that accounts for this issue. Furthermore, this model can take in both linear and non-linear data which allows us to use the different variables. To explain our workflow, we split our predictor data into two parts, numerical and categorical. Next, we find the coeeficients of each variable, plot the confusion matrix based on the trained logistic model. Lastly, evaluate the models accuracy through the accuracy, precision, F1-score, support and recall generated in the classification report.

### Jupyter Notebook #4: Machine Learning Model 2 - Decision Tree & Fine-tuning
We explore the use of Decision Trees as a machine learning model for our predictors which contain both categorical and numerical data. The process of training the machine learning model is similar where we split our data into two parts, create a confusion matrix, fine tune the hyper parameters then plot a binary tree. Lastly, we compare coefficient value of each variable and finally evaluating the the model base on the accuracy, precision, F1-score, support and recall generated in the classification report.


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
