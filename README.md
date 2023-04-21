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
s
### Jupyter Notebook #2: EDA/Visualisations/Correlations
s
### Jupyter Notebook #3: Machine Learning Model 1 - Logistic Regression & Fine-tuning
s
### Jupyter Notebook #4: Machine Learning Model 2 - Decision Tree & Fine-tuning
s

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
3) Outside of the project, we can derive from the findings from at least 89 thousand data that one should always be careful of their HbA1c_level. Being negligent in maintain the levels can result high possibilities of one developing diabetes 

## References
- s
