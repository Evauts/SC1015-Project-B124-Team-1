# **Mini project for SC1015, Group 1**

## About our team and contributions
- Tan Chuan You *(Data Cleaning, Data Visualization/EDA, Logistic Regression, Decision Tree)*
- Divya Gupta *(Dataset selection, Logistic Regression, README file, Presentation Slides)*
- Tan Jun Liang *(Data Cleaning, Data Visualization/EDA, README file, Video creation and editing)*

## Problem Definition
- What is the top predictor for the response variable 'diabetes'?
- What "out of syllabus" techniques can we apply to our dataset?
- What models can best fit our dataset?

## Deliverables

### Dataset
A dataset from Kaggle where we extract our data from. https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

### Presentation Slides
A summarised content and resources for our presentation of the project.

### Jupyter Notebook #1: Data Extraction, Data Cleaning
Imports all necessary libaries. We take a general view (such as min-max values and distribution of data using boxplot) at our uncleaned data to spot anomalies or null values. From there, we clean out the anomalies and outliers to improve the accuracy of our predictors by narrowing the spread of data.

### Jupyter Notebook #2: EDA/Visualisations/Correlations
We utilise the module "plotly" in creating more detailed graphs that help us observe trends and data patterns better. We display and compare percentages, looking at the distribution of diabetics among data groups to spot trends and patterns that can bring us useful insights. 

>**PROCESS**
>
>1) Plot and evaluate our first set of graphs
>> Realised that data was imbalanced at 0.9 : 9.1 (Diabetics : Non-Diabetics)
>
>2) Hence, split the data into train and test sets of ratio 8 : 2
>
>3) Use a hybrid technique on the train dataset ONLY for oversampling and undersampling (SMOTE + TOMEK) to 3 : 7 ratio (Diabetics : Non-Diabetics)
>>  This reduction in imbalance allows us to better train our models and gain better accuracy while also retaining the reliability of our data. 
>
>4) By examining correlations and graphs from the sampled data, we choose the best predictors to train our machine-learning models and justify the drop of the other predictors.

`Chosen Predictors: age, HbA1c_level, bmi and blood_glucose_level`

### Jupyter Notebook #3: Machine Learning Model 1 - Logistic Regression & Fine-tuning
Logistic Regression evaluates binary outcomes, which fits our response variable 'diabetes' with the values 0 and 1. This model has a performance metric that accounts for imbalance data issue which is preferred for our data as there still exists minor imbalance after cleaning. 

>**PROCESS**
>
>1) Create and fit data into the Gradient Boosting Classifier model
>
>2) Pick out best hyperparameters needed for train dataset with gridsearch 
>
>3) Fine-tune hyperparameters to create improved model
>
>4) Train model with test and train dataset 
>
>5) Compare coeeficients of each variable.
>
>6) Plot the confusion matrix based on the Logistic Regression Model. 
>
>7) Evaluate the model's accuracy through the Area Under Curve (AUC) from the Receiver Operating Characteristic(ROC) graph which takes into account of imbalance data. 
>
>8) Further evaluate through information such as accuracy, precision, F1-score, support and recall generated in a classification report.

### Jupyter Notebook #4: Machine Learning Model 2 - Decision Tree & Fine-tuning
Applying our knowledge in our syllabus, we explore the use of Decision Trees for our predictors. A big reason is that the decision tree captures non-linear relationships between predictor variables and outcome, evaluating and presenting complex relationships between our predictors and response variable. With our huge data of 124k rows, the decision tree is suitable as it can be scaled to handle larger data.  
>**PROCESS**
>
>1) Create and fit data into the Gradient Boosting Classifier model
>
>2) Pick out best hyperparameters needed for train dataset with gridsearch 
>
>3) Fine-tune hyperparameters to create improved model
>
>4) Train model with test and train dataset
>
>5) Plot the confusion matrix to see the general prediction of our data
>
>6) Plot a decision tree to find the importance of each variable. 
>
>7) Evaluate the model's accuracy through the Area Under Curve (AUC) from the Receiver Operating Characteristic(ROC) graph which takes into account imbalance data.
>
>8) Further evaluate through information such as accuracy, precision, F1-score, support and recall generated in a classification report.

### Jupyter Notebook #5: Gradient Boosting Classifier
As the non-linear Decision tree gave us a better accuracy, we used Gradient Boosting Classifier as it improves on weak prediction models such as decision trees to create a stronger predictive model. With the benefits such as handling complex data, high-dimensional data, imbalanced class distributions and providing interpretable results. We found this model relevant and useful in analysing this dataset. 

>**PROCESS**
>1) Create and fit data into the Gradient Boosting Classifier model
>
>2) Pick out best hyperparameters needed for train dataset with gridsearch 
>
>3) Fine-tune hyperparameters to create improved model
>
>4) Train model with test and train dataset
>
>5) Plot the confusion matrix to see the general prediction of our data
>
>6) Evaluate the model's accuracy with information such as accuracy, precision, F1-score, support and recall generated in a classification report. 
>
>7) Look at the Area Under Curve (AUC) from the Receiver Operating Characteristic(ROC) graph which takes into account imbalance data
>
>8) Evaluate the importance of predictors.


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
- SMOTE and Temok (Hybrid, oversampling + undersampling)
- Logistic Regression
- GridSearch (FineTuning Hyper parameters)
- Gradient Boosting Classification
- Receiver Operating Characteristic(ROC) Graph


## Conclusion of insights
- We were able to identify the **best** predictor as **"HbA1c_level"** for the response variable **"diabetes"**
- The **Gradient Boosting Classifier** machine learning model provided the most accuracy of 0.97 fit for our data compared to Logistic Regression and Decision Trees
- Our fine-tuning methods such as gridsearch were **effective** as it improved our machine learning models.

## What has the team learnt from the project?
1) Due to our lack of experience in building a machine learning model, our team had the naive idea that any machine learning model would work for any dataset. This project has corrected our wrong perceptions and given us insightful lessons. It is important for us to examine our dataset properly before choosing from the many available algorithms and techniques. To pick a suitable machine learning model, it is crucial to consider the influential factors of dataset such as size, distribution, types of variable, and the advantageous that the machine learning model can bring to your project.
2) Cooperation and communication with group mates. It was quite challenging for us to keep track of the updates and improvements made by each of us at our own time. We had a few miscommunication here and there but gradually set procedures to update one another whenever making changes to the group project. As well as assigning one person to be in charge so that everybody can stay on the right track.
3) With the vast amount of data that we had, we were able to learn many new techniques such as oversampling and undersampling, gridsearch to fine tune our models as well as ROC graph to evaluate the performance of different predictors. There were also many machine learning models that we learnt outside of what we have applied in this project from the process of choosing suitable models.
4) With the increasing amount of youths developing diabetes, it is important for us to apply the insights from our findings into our lives that one should always be careful of the fluctuations in their HbA1c levels which is the average blood sugar levels over the past 3 months. Being negligent in maintaining the levels can result in high possibilities of one developing diabetes. Despite HbA1c_level being the best predictor, we believe that Blood glucose level and BMI are factors that one should take into consideration as well due to their moderate correlations to diabetes.
5) As we all say, `"better safe than sorry"` and `"better to prevent than to cure"`

## References
- https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
- https://sph.nus.edu.sg/2016/05/young-singaporean-diabetes-on-the-rise/#:~:text=Research%20studies%20conducted%20by%20the,by%20the%20age%20of%2065
- https://realpython.com/logistic-regression-python/#logistic-regression-in-python-with-scikit-learn-example-1https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
- https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/
- https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
