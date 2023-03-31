# Tanzanian Water Wells

**Author**: [Matthew Duncan](mailto:mduncan0923@gmail.com)

![initial_plot](/photos/tanzania.jpg)

## Overview
For this analysis, I have analyzed data on almost 60,000 wells across the country of Tanzania. I've utilized descriptive analysis, visualizations, and machine learning methods to provide recommendations on focus points for well repair efforts.



## Business Problem
Tanzania struggles with providing clean water to the people of its country. The goal of this analysis is to use data from existing wells throughout the country to identify the wells in need of repair. 

This analysis is focused on assisting the Tanzanian Government reduce resources spent to identify water wells that are in need of repair to ensure that clean water is available to its citizens. 

![initial_plot](/photos/well.jpg)

## Data

The data was provided by DrivenData.org in partnership with Taarifa who aggregated the data from the Tanzanian Ministry of Water.

## Methods
This analysis uses descriptive analysis, visualizations, and machine learning methods to describe trends for wells in need of repair across the country. Additionally I have provided recommendations on focus points for well repair efforts

To better understand the data, I have reviewed the features and separated them into numerical and categorical feature types. To better define the status of each well for this analysis, I have also set up a binary system for the wells:
- 0 = Functional
- 1 = Needs Repair

About 54% of all of the wells are completely functional and 46% of the wells need repairing.

![initial_plot](/photos/initial_plot.png)

To ensure that I'm reducing the noise in the dataset while simultaneously being cognizant of computing time, I've removed a few columns that are not relevant to this analysis or are duplicated in other features:

- `scheme_name` - Who operates the waterpoint.
- `id` - A unique identifier number for each well.
- `date_recorded` - The date the row was entered.
- `funder` - Who funded the well.
- `recorded_by` - Group entering this row of data.
- `wpt_name` - Name of the waterpoint if there is one.
- `region`, `subvillage`, `ward`, `lga`, and `basin` - Geographic location.
- `quality_group` - The quality of water.
- `quantity_group` - The quantity of water.
- `installer` - Organization that installed the well.
- `source_type` and `source_class` - The source of the water.
- `waterpoint_type_group` - The kind of waterpoint.
- `extraction_type_group` and `extraction_type_class` - The kind of extraction the waterpoint uses.
- `management_group` - How the waterpoint is managed.
- `payment_type` - What the water costs.

To streamline the analysis, reduce the potential for data leakage, and ensure that all preprocessing steps are completed with each model, I have set up a pipeline. The pipeline begins by separating the numerical data from the categorical data, once separate, the two types of data can undergo their respective preprocessing.

- Numerical Data:
    - The columns for numerical data or not missing any values so there is no need to include an imputer.
    - `StandardScaler` is being utilized to standardize and scale all numerical data.

- Categorical Data:
    - `SimpleImputer` is being utilized to fill missing data in the categorical columns based on the most frequent value within that column. A 'missing' indicator is in place to help identify that the data was not originally found within the dataset.
    - `OneHotEncoder` is being utilized to convert the categorical information to a binary system for modeling. 




## Results
### Visualizing the Data
#### `waterpoint_type` Visualization
When we look at the functionality of wells by the type of water point, we can clearly see that `communal_standpipe_multiple` wells have a lower functionality rate than the other types with 63% of wells needing repair. 

Repairing these types of wells would also provide a huge benefit to the communities they’re in and should be a focus for repair efforts. These well types have multiple spigots and are designed for large groups of people to have access at once.

![initial_plot](/photos/type_percent.png)

#### `quantity` Visualization
When we look at the functionality of wells by the quantity of available water, we can see that almost all wells that are dried out are in need of repair. While these `dry` waterpoints have the highest percentage of wells that need repairing, they should not be a focus since there would be no benefit to repairing the well.

Even though only 34% of the wells that provide `enough` water are in need of repair, they should be the primary focus of any repair efforts. 

![initial_plot](/photos/quantity_percent.png)

#### `extraction_type_class` Visualization
When we look at well functionality based on the `extraction_type_class` of the well we can clearly see that automatic (not man-powered) wells have a lower functionality rate than the manual powered wells. 

62% of all wells extracted by motor and 57% of those extracted by wind power are in need of repair. It makes sense that the more complex wells would need repairs more often and should be a focus for inspections. 

![initial_plot](/photos/extraction_percent.png)

#### `region` Visualization
Looking at the 5 `region` percentages with the most wells in need of repair. `Lindi` and `Mtwara` are the two regions with the highest percentage of wells in need of repair.

According to the 2012 national census, the regions have a combined population of over 2,000,000.

70% of all wells in Lindi and Mtwara are in need of repair. Efforts should be made to focus on well repairs in these regions first. 

![initial_plot](/photos/region_percent.png)

### Machine Learning
I have used the following Machine Learning methods in conjunction with `GridSearchCV` during this analysis:
   - Logistic Regression
   - K-Nearest Neighbors
       - K-Means
   - Decision Trees
   - Random Forests
   - C-Support Vector Classification (SVC)
   - Nu-Support Vector Classification (NuSVC)
   - AdaBoost
   - XGBoost
   - Stack Classification Methods
   
#### Logistic Regression
Using Logistic Regression, I was able to achieve accuracy of 77% and an ROC-AUC score of 84% on test data. The accuracy and ROC-AUC scores line up relatively well between train and test data indicating that there might be a slight overfitting but not significant overfitting.
#### K-Nearest Neighbors & K-Means
Using K-Nearest Neighbors, I was able to achieve accuracy of 80% and an ROC-AUC score of 88% on test data. The accuracy and ROC-AUC scores of the training data are much higher than the that of the test data, indicating overfitting of the model during training.

I also employed K-Means Clustering in an attempt to improve my K-Nearest Neighbors model. I have a loop to create a table utilizing 1-7 clusters, then separate the predictor data based on the cluster results. I then run K-Nearest Neighbors on each of the cluster groupings to determine if clustering helps the dataset. When the number of clusters is set to 1, it is the same as running K-Nearest Neighbors without clustering. 

The results of the model show that it is more beneficial to skip K-Means Clustering altogether. 

![initial_plot](/photos/Kmeans_results.png)

#### Decision Tree
Using Decision Trees, I was able to achieve accuracy of 78% and an ROC-AUC score of 79% on test data. The accuracy and ROC-AUC scores of the training data are much higher than the that of the test data, indicating overfitting of the model during training.
#### Random Forest
Using Random Forest, I was able to achieve accuracy of 82% and an ROC-AUC score of 90% on test data. The accuracy and ROC-AUC scores of the training data are much higher than the that of the test data, indicating overfitting of the model during training.
#### C-Support Vector Classification (SVC) 
Using SVC, I was able to achieve accuracy of 79% and an ROC-AUC score of 86% on test data. The accuracy and ROC-AUC scores line up relatively well between train and test data indicating that there might be a slight overfitting but not significant overfitting. This model takes about 20 minutes to fit, due to time constraints, I was not able to complete a gridsearch to hypertune it.
#### Nu-Support Vector Classification (NuSVC)
Using NuSVC, I was able to achieve accuracy of 79% and an ROC-AUC score of 86% on test data. The accuracy and ROC-AUC scores line up relatively well between train and test data indicating that there might be a slight overfitting but not significant overfitting. This model takes about 30 minutes to fit, due to time constraints, I was not able to complete a gridsearch to hypertune it.
#### AdaBoost
Using AdaBoost, I was able to achieve accuracy of 76% and an ROC-AUC score of 83% on test data. The accuracy and ROC-AUC scores line up relatively well between train and test data indicating that there might be a slight overfitting but not significant overfitting.
#### XGBoost
Using XGBoost, I was able to achieve accuracy of 82% and an ROC-AUC score of 90% on test data. The accuracy and ROC-AUC scores of the training data are much higher than the that of the test data, indicating overfitting of the model during training.
#### XGBoost Random Forest
Using XGBoost Random Forest, I was able to achieve accuracy of 73% and an ROC-AUC score of 81% on test data. The accuracy and ROC-AUC scores line up relatively well between train and test data indicating that there might be a slight overfitting but not significant overfitting.
#### Stacking Classifier
I have created a stacking classifier using the best models that were previously defined. I have decided to exclude XGBoost Random Forest due to the poor results, SVC and NuSVC due to their fit and runtime length, and Decision Trees due to the vast overfitting. 

I also used the baseline model for K-Nearest Neighbors because the final model was very overfit. Stacking Classifier benefits from worse performing models but is hindered by models that are overfit.

Using Stacking Classifier, I was able to achieve accuracy of 83% and an ROC-AUC score of 90% on test data. The accuracy and ROC-AUC scores of the training data are much higher than the that of the test data, indicating overfitting of the model during training.

![initial_plot](/photos/final_plot.png)


The results table lets us clearly visualize the performance of the models run in this analysis. Overall, the Stacking Classifier has the best results with an accuracy of 82.5% and an ROC-AUC score of 90.3% on the test set. That being said, it took roughly 55 times longer to fit than the Random Forest Final Model.

When looking at time-to-fit in addition to pure performance metrics, the Stacking Classifier took a little over 6 minutes to fit. It is recommended to use the Stacking Classifier for future predictions of wells needing repair in Tanzania.

![initial_plot](/photos/results_table.png)

## Conclusions

This report found four areas to focus on:
- Automatically powered wells rather than man-power wells 
- Wells that provide enough water rather than dry or seasonal wells 
- Efforts should be focused on multiple communal standpipe wells as these will have the greatest impact to the communities
- Efforts should be focused in the Lindi and Mtwara Regions

Next steps would be to implement the model and to continue to fine tune it as new data is made available. As new data is introduced, the model can continue to learn and grow to make better predictions.

Expectations include some level setting as well. The model is not 100% perfect and there will be some inaccurate predictions but the goal here is to supplement current identification methods and to improve on the current levels of detection for wells needing repair. 

Utilizing this model would greatly improve the detection rate of wells needing repair, leading to a reduction to the cost of well maintenance and a reduction in the manpower needed for inspections.

In addition to cutting costs and reducing manpower needs, the citizens of Tanzania would greatly benefit by having increased access to clean drinking water. 

### Future Considerations
Further Analysis could yield additional insights to improve predictions of wells in need of repair. If given more time and resources:

- I would look to further reduce overfitting in several of the models. This would enhance the performance of the individual models and the Stacking Classifier model. 

- Both the SVC and NuSVC models have fairly good performance but take too long to run. I would like to be able to hypertune them to see if I can reduce run time.

- I would like to spend more time cleaning the data, with so many datapoints and limited time, I feel like the initial data analysis could be expanded before moving on to modeling. 


## For More Information

See the full analysis in the [Jupyter Notebook](./Water_Well_Project_Notebook.ipynb) or review this [presentation](./Water_Well_Presentation.pdf).

**For additional info, contact:**
- Matthew Duncan: mduncan0923@gmail.com

![initial_plot](/photos/closing_well.jpg)

## Repository Structure

```
├── data
│   ├── target_data.csv
│   ├── test_data.csv
│   ├── X_data.csv
├── photos
│   ├── accuracy.jpg
│   ├── closing_well.jpg
│   ├── extraction_count.png
│   ├── extraction_percent.png
│   ├── final_plot.png
│   ├── initial_plot.png
│   ├── KMeans_results.png
│   ├── quantity_count.png
│   ├── quantity_percent.png
│   ├── region_percent.png
│   ├── results_table.png
│   ├── roc.jpg
│   ├── tanzania.jpg
│   ├── type_count.png
│   ├── type_percent.png
│   └── well.jpg
├── Water_Well_Project_Notebook.ipynb
├── modelruns.py
├── Water_Well_Presentation.pdf
└── README.md
```
