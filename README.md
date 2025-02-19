https://www.nature.com/articles/s41598-024-75876-2

betterfit: https://github.com/alexhubbe/image_classification/blob/main/reports/images/betterfit.jpg

mobilenet summary: https://github.com/alexhubbe/image_classification/blob/main/reports/images/mobilenet_summary.png

nasnet summary: https://github.com/alexhubbe/image_classification/blob/main/reports/images/nasnetmobile_summary%20.png

overfit: https://github.com/alexhubbe/image_classification/blob/main/reports/images/overfit.jpg



# Convolutional Neural Network   

This repository contains code and notebooks showcasing key techniques for improving performance when working with **convolutional neural networks (CNN)**, using the **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)** dataset—a well-known collection of 60,000 32×32 color images across 10 different classes.


<p align="center">
<img src="https://github.com/alexhubbe/image_classification/blob/main/reports/images/example_images.jpg" width="100%" alt="Readme Image">
</p>  

<br>

CNNs have revolutionized the field of computer vision, enabling applications ranging from image segmentation and object recognition to more complex tasks like detecting diseases such as cancer in medical images ().

My main goal was to improve the score on the test dataset by preprocessing the data and optimizing hyperparameters using Optuna.  

The evaluation metric used was `Average Precision`, as it is well-suited for highly imbalanced datasets, such as credit card fraud detection (Borgne et al., 2022). The machine learning methods applied were `Logistic Regression`, `Random Forest`, and `XGBoost`. Logistic Regression was chosen for its simplicity, while Random Forest and XGBoost are well-suited for this type of analysis (Borgne et al., 2022).  

## Key Findings  
- The choice of preprocessing strategy improved the test dataset score by **1.1% to 30%**, depending on the machine learning method (Figure 1).  
- Hyperparameter optimization improved the test dataset score by **1.1% to 2.1%**, depending on the machine learning method (Figure 2).  

<div style="display: flex; justify-content: center; gap: 20px;">
    <img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/preprocessor.jpg" width="48%" alt="Preprocessing Results">
    <img src="https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/reports/images/hp_optimization.jpg" width="48%" alt="Hyperparameter Optimization Results">
</div>  

<br>

Below is a succinct description of the steps taken in this project.

## [Exploratory Data Analysis and Data Engineering](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/01_ah_eda.ipynb)  
At this stage, the following procedures were performed: 

1. **Sanity Check**  
   - Conducted an initial inspection for duplicate entries and missing values across features.

2. **Numeric Features**  
   - Inspected the normality of the features, pairwise correlations, and the presence of outliers.

3. **Feature-Target Relationships**  
   - Examined the association between features and the target variable.  

4. **Target Variable Analysis**  
   - Observed that fraud cases were relatively evenly distributed over time.  
   - Decided to use the first day's data for training the models and the second day's data for testing.

5. **Time Feature**  
   - Created a binary feature representing periods of low (hours 1–6) and high (hours 0, 7–23) transaction amounts. This improved the test dataset score by **0.7%** ([see details](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/ah_appendix.ipynb)).  

6. **Feature Transformation**  
   - Explored whether PowerTransformer or QuantileTransformer would be the best transformation strategy.  

## [Machine Learning](https://github.com/alexhubbe/credit_card_fraud_detection_kaggle/blob/main/notebooks/02_ah_model.ipynb)  
In this phase, I implemented the following steps:  

1. **Preprocessing**  
   - Determined the best preprocessing strategy for each machine learning method.

2. **Handling Imbalance**  
   - Used the hybrid over- and under-sampling **SMOTE-TOMEK** method to confirm that Average Precision is robust against class imbalance.

3. **Hyperparameter Optimization**  
   - Utilized the **Optuna** framework to optimize hyperparameters, improving the test dataset scores over the default **Scikit-Learn** and **XGBoost** hyperparameters.  

## Tools and Technologies  
- **Libraries**: Imbalanced-learn, Matplotlib, NumPy, Optuna, Pandas, Seaborn, Scikit-Learn, XGBoost  

## Project organization

## Project organization

```
├── .gitignore                                <- Files and directories to be ignored by Git
|
├── LICENSE                                   <- License type 
|
├── README.md                                 <- Main README explaining the project
|
├── environments                              <- Requirements files to reproduce the analysis environments
|   ├── autogluon_environment.yml             <- Environment for running '02_ah_autogluon.ipynb' 
|   ├── deep_learning_environment.yml         <- Environment for running '01_ah_model.ipynb', 'ah_appendix.ipynb'
|
├── models                                    <- Trained and serialized models, model predictions, or model summaries
|   ├── best_model.keras                      <- Stores the best-performing model during training based on validation sparse categorical accuracy (used in MobileNet Model II.5)
|   └── kt_mobilenet_tuning                   <- Directory to store Keras hyperparameter optimization results
|       ├── mobilenet_bayesian                <- Directory for trials from the optimization procedure
|
├── notebooks                                 <- Jupyter notebooks
|   ├── 01_ah_model.ipynb                     <- Image Classification with Transfer Learning
|   ├── 02_ah_autogluon.ipynb                 <- Image Classification with Autogluon`s AutoMM
|   ├── ah_appendix.ipynb                     <- Model Summary Diagram
|
├── references                                <- Data dictionaries, manuals, and other explanatory materials
|   ├── data_contextualization.md             <- Description of the dataset
|
├── reports                                   <- Generated analyses in HTML, PDF, LaTeX, etc., and results
│   └── images                              <- Images used in the project
|      ├── betterfit.jpg                    <- Figure with a better fit in the train and validation loss curve
|      ├── example_images.jpg               <- Figure to Illustrate the Pictures within Each Class
|      ├── mobilenet_summary.png            <- MobileNet Models Summary Diagram
|      ├── nasnetmobile_summary .png        <- NasNetMobile Models Summary Diagram
|      ├── overfit.jpg                      <- Figure depicting overfit in the train and validation loss curve
```

## Contributing
All contributions are welcome!

### Issues
Submit issues for:
- Recommendations or improvements
- Additional analyses or models
- Feature enhancements
- Bug reports

### Pull Requests
- Open an issue before starting work.
- Fork the repository and clone it.
- Create a branch and commit your changes.
- Push your changes and open a pull request for review.

## References

Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection—Practical Handbook. Université Libre de Bruxelles.




















```
├── .gitignore                              <- Files and directories to be ignored by Git
|
├── LICENSE                                 <- License type 
|
├── README.md                               <- Main README explaining the project
|
├── data                                    <- Project data files
|   ├── ames_dataset.csv                    <- Original dataset from R's tidymodels
|   ├── ATNHPIUS11180Q.csv                  <- Data for Ames' House Price Index
|   ├── clean_data.csv                      <- Dataset prepared for machine learning analysis
|   ├── CSUSHPINSA.csv                      <- Data for the USA's Case-Shiller Index
|   ├── original_plus_lat_lon.csv           <- Kaggle dataset with added longitude and latitude from tidymodels
|   ├── test.csv                            <- Original test dataset from Kaggle
|   ├── train.csv                           <- Original train dataset from Kaggle
|
├── environments                            <- Requirements files to reproduce the analysis environments
|   ├── autogluon_environment.yml           <- Environment for running '03_ah_MODEL.ipynb' 
|   ├── eda_environment.yml                 <- Environment for running '01_ah_merging_datas.ipynb', '02_ah_EDA.ipynb', and 'ah_appendix_2.ipynb' 
|   ├── maps_environment.yml                <- Environment for running 'ah_appendix_1.ipynb'
|
├── models                                  <- Trained and serialized models, model predictions, or model summaries
|
├── notebooks                               <- Jupyter notebooks
|   ├── 01_ah_merging_datas.ipynb           <- Adding latitude and longitude information to Kaggle's dataset
|   ├── 02_ah_EDA.ipynb                     <- Exploratory data analysis
|   ├── 03_ah_MODEL.ipynb                   <- Machine learning approach
|   ├── ah_appendix_1.ipynb                 <- Creating maps
|   ├── ah_appendix_2.ipynb                 <- Exploring an alternative dataset transformation
|   └── src                                 <- Source code used in this project
|       ├── __init__.py                     <- Makes this a Python module
|       ├── auxiliaries.py                  <- Scripts to compute nearest houses and median sale prices
|       ├── config.py                       <- Basic project configuration
|       ├── eda.py                          <- Scripts for exploratory data analysis and visualizations
|
├── references                              <- Data dictionaries, manuals, and other explanatory materials
|   ├── data_description.txt                <- Description of the dataset as presented on Kaggle
|
├── reports                                 <- Generated analyses in HTML, PDF, LaTeX, etc., and results
|   ├── best_kaggle_prediction.csv          <- Best prediction from Kaggle competition
│   └── images                              <- Images used in the project
|      ├── betterfit.jpg                    <- Figure with a better fit in the train and validation loss curve
|      ├── example_images.jpg               <- Figure to Illustrate the Pictures within Each Class
|      ├── mobilenet_summary.png            <- MobileNet Models Summary Diagram
|      ├── nasnetmobile_summary .png        <- NasNetMobile Models Summary Diagram
|      ├── overfit.jpg                      <- Figure depicting overfit in the train and validation loss curve
```

## Contributing
All contributions are welcome!

### Issues
Submit issues for:
- Recommendations or improvements
- Additional analyses or models
- Feature enhancements
- Bug reports

### Pull Requests
- Open an issue before starting work.
- Fork the repository and clone it.
- Create a branch and commit your changes.
- Push your changes and open a pull request for review.