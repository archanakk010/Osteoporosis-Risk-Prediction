
# Osteoporosis Risk Prediction


## Introduction

Osteoporosis is a common medical condition that causes bones to become weak and more likely to fracture. Early detection and intervention are essential for effective treatment and management. In this project, we develop a machine learning model to predict the risk of osteoporosis using various health and lifestyle factors. This predictive model aims to help healthcare providers identify individuals at risk, thereby improving patient outcomes through timely intervention and appropriate treatment strategies.


## Problem Statement
Osteoporosis often goes undiagnosed until a fracture happens, leading to severe complications and reduced quality of life. The challenge is to use patient data to predict osteoporosis risk accurately. By analyzing factors like age, gender, hormonal changes, family history, and lifestyle choices, we aim to create a reliable model that can serve as a predictive tool in clinical settings.


## Data Collection
The dataset for this project has 16 columns covering various health and lifestyle factors, along with the osteoporosis status of the individuals.The dataset is Sourced from Kaggle, the dataset includes features like age, gender, hormonal changes, family history, race/ethnicity, body weight, calcium intake, vitamin D intake, physical activity, smoking, alcohol consumption, medical conditions, medication, prior fractures, and osteoporosis status.


## Data Description

ID: Unique identifier for each individual

Age: Age of the individual

Gender: Gender of the individual

Hormonal Changes: Indicates whether the individual has experienced hormonal changes

Family History: Family history of osteoporosis

Race/Ethnicity: Race or ethnicity of the individual

Body Weight: Weight of the individual

Calcium: Calcium intake

Vitamin D: Vitamin D intake

Physical Activity: Level of physical activity

Smoking: Smoking habits

Alcohol Consumption: Alcohol consumption habits

Medical Conditions: Existing medical conditions

Medication: Medication usage

Prior Fracture: History of prior fractures

Osteoporosis: Target variable indicating osteoporosis status

## Exploratory Data Analysis (EDA)

During Exploratory Data Analysis (EDA), we analyzed  both numerical and categorical columns to understand the data distribution and relationships between variables. We used histograms, box plots, scatter plots, correlation matrices, and Power BI visualizations for this analysis. We also checked for missing values and duplicates.

### Visualizations included:

#### 1..Distribution of Osteoporosis Status: A bar plot showing the distribution of patients with and without osteoporosis.

#### 2.Age Distribution: A histogram showing the age distribution of patients.

#### 3.Gender Distribution: A bar plot showing the distribution of genders in the dataset.

#### 4.Correlation Matrix: A heatmap displaying the correlation between numerical features and the target variable.

#### 5.Feature Relationships: Scatter plots and box plots showing the relationship between each independent feature and the target variable (osteoporosis status).

#### 6.Power BI Visualization for Feature Study

To enhance the understanding of the data and its impact on osteoporosis prediction, a Power BI dashboard can be created. This dashboard will include various visualizations to explore the distribution and relationship of features with osteoporosis. Below is a detailed plan for the dashboard, including descriptions of each visualization.

##### Dashboard Components

##### Page.1

![Screenshot (822)](https://github.com/archanakk010/main_project/assets/132830280/065a76c8-481a-45a2-82c3-5f2a954b4724)

###### 1.Total  Patients,Patients with Osteoporosis and Patients without Osteoporosis details

Visualization:Card

Description: These cards display the total number of patients, the number of patients diagnosed with osteoporosis, and those without osteoporosis.

###### 2.Race/Ethnicity

Visualization:Slicer

Description: This slicer allows filtering the data based on different race/ethnicity groups.

###### 3.Osteoporosis by Age

Visualization:clustered column chart


Description: This chart shows the distribution of patients' ages and highlights how age correlates with osteoporosis incidence, helping to identify age groups that are more at risk.

###### 4. Distribution of Patients with and without Osteoporosis


Visualization:pie chart

Description: This chart displays the count of patients who have osteoporosis versus those who do not, helping to understand the proportion of the population affected by osteoporosis.


##### Page.2

![Screenshot (823)](https://github.com/archanakk010/main_project/assets/132830280/4de5fe3c-de36-4a63-9b6c-669cf41a32b7)

###### 1.Gender and Osteoporosis

Visualization: Stacked Column Chart

Description: This chart breaks down the osteoporosis status by gender, showing the distribution of osteoporosis among males and females.

###### 2.Hormonal Changes and Osteoporosis

Visualization: Stacked Column Chart

Description: This chart shows the relationship between hormonal changes and osteoporosis status, highlighting whether hormonal changes are a significant factor in osteoporosis.

##### Page.3

![Screenshot (824)](https://github.com/archanakk010/main_project/assets/132830280/60e2c367-5e41-4cbb-938c-e36ee10cc372)


###### 1.Family History and Osteoporosis

Visualization: Stacked Column Chart

Description: This chart displays the relationship between family history of osteoporosis and the patients' osteoporosis status, indicating the hereditary aspect of the condition.

###### 2.Race/Ethnicity and Osteoporosis


Visualization: Stacked Column Chart

Description: This chart shows the distribution of osteoporosis across different race/ethnicity groups, helping to identify if certain races/ethnicities are more prone to osteoporosis.

##### Page.4

![Screenshot (825)](https://github.com/archanakk010/main_project/assets/132830280/14e2dd6a-00e4-41d8-a11a-d1b8807a0e9b)

###### 1.Body Weight and Osteoporosis


Visualization: Stacked Column Chart


Description: This chart shows the distribution of body weight for patients with and without osteoporosis, indicating how body weight correlates with osteoporosis risk.

###### 2.Physical Activity and Osteoporosis

Visualization: Stacked Column Chart

Description: This chart shows the correlation between physical activity levels and osteoporosis, indicating whether an active lifestyle helps in preventing osteoporosis.

##### Page.5

![Screenshot (826)](https://github.com/archanakk010/main_project/assets/132830280/32b5d5d5-aa22-41d2-b03d-9e7643c99db5)

###### 1.Nutrition and Osteoporosis


Visualization: Stacked Column Chart

Description: This chart explores the relationship between nutritional intake (Calcium and Vitamin D) and osteoporosis status, highlighting the importance of diet in bone health.

##### Page.6

![Screenshot (827)](https://github.com/archanakk010/main_project/assets/132830280/15560d84-cd91-4aa8-b34b-65eb80f4ac84)

###### 1.Medical Conditions, Medications, and Osteoporosis


Visualization: Stacked Column Chart

Description: This chart displays the relationship between existing medical conditions, medication usage, and osteoporosis status, showing the impact of other health factors and treatments on osteoporosis risk.

##### Page.7

![Screenshot (828)](https://github.com/archanakk010/main_project/assets/132830280/af102d83-5dd5-4cc3-b39a-2683e981fc5c)

###### 1.Smoking and Alcohol Consumption and Osteoporosis


Visualization: Stacked Column Chart

Description: This chart displays the impact of smoking and alcohol consumption on osteoporosis status, highlighting lifestyle choices that may affect bone health.


##### Page.8

![Screenshot (829)](https://github.com/archanakk010/main_project/assets/132830280/67cde12c-5ead-49a9-a89c-4d17ea6bab10)


###### 1.Prior Fracture and Osteoporosis

Visualization: Stacked Column Chart

Description: This chart shows the relationship between prior fractures and current osteoporosis status, indicating whether past fractures are a significant predictor of osteoporosis.


## Data Preprocessing


Since there were no missing values or outliers, we focused on transforming categorical features into numerical ones using Label Encoding and scaling numerical features using MinMaxScaler to ensure consistent input for the models.

## Feature Engineering

Feature Transformation: Categorical features were encoded using Label Encoding, and numerical features were scaled.
Data Splitting
The dataset was divided into training and testing sets with a 70-30 split to ensure that our model's performance could be evaluated on unseen data.

## Model Selection
We explored several machine learning algorithms to identify the best model for predicting osteoporosis risk. The algorithms used include:

1.Logistic Regression

2.Support Vector Machine (SVM)

3.Decision Tree

4.Multi-layer Perceptron (MLP) Classifier

5.Naive Bayes

6.Random Forest

7.K-Nearest Neighbors (KNN)

8.Gradient Boosting Classifier

9.AdaBoost Classifier

## Model Training and Evaluation

Each model was trained on the training dataset and evaluated using the testing dataset. Evaluation metrics included accuracy, precision, recall, and F1-score. Confusion matrices were generated to visualize the performance of each model.

## Hyperparameter Tuning
Hyperparameter tuning was performed using grid search to optimize the models further.

## Results
The performance of the models was compared based on the evaluation metrics. The best-performing model was identified, providing the most reliable predictions for osteoporosis risk.

## Model Deployment
The final model, Gradient Boosting Classifier, was saved and tested with unseen data to ensure its robustness and reliability. The Gradient Boosting Classifier achieved an accuracy of 92.18%, precision of 100%, recall of 84.82%, and an F1-score of 91.78%.

## Limitations
1.Data Quality: The dataset may have limitations in terms of sample size and diversity.

2.Feature Availability: Some potentially relevant features may not be included in the dataset. Including dietary habits, sun exposure, caffeine intake, and bone density as features in osteoporosis prediction models can enhance accuracy by providing a more comprehensive view of bone health factors. These factors contribute to bone density maintenance and fracture risk assessment. However, challenges such as variability in data quality, measurement methods, and access to bone density testing must be carefully addressed for effective model integration.

3.Model Generalization: The model may not generalize well to different populations or clinical settings.

4.Fewer Samples: With fewer samples available for analysis, the model may face challenges in capturing the full spectrum of osteoporosis risk factors and generating reliable predictions


## Conclusion
This project successfully developed a machine learning model for predicting osteoporosis risk. The model can assist healthcare providers in early identification and intervention, potentially improving patient outcomes. Future work could involve collecting more data, exploring additional features, and applying advanced algorithms to enhance the model's accuracy and generalizability.

## Future Work

1.Collect More Data: Increasing the dataset size for better model training.

2.Try Different Algorithms: Exploring deep learning techniques.

3.Model Updating: Regularly updating the model with new data.

4.Resampling: Addressing any class imbalance in the data.

5.Adding More Features: Including more relevant features for improved prediction accuracy.
