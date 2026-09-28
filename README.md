#  Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project is a **Heart Disease Prediction** system built using Machine Learning classification techniques. The model uses patient health-related features to predict the presence or absence of heart disease.

The project demonstrates a complete machine learning workflow, including data cleaning, feature engineering, feature selection, data balancing, feature scaling, model training, evaluation, hyperparameter tuning, and deployment as a web application.

### Key Features

* Data manipulation using **Pandas**
* Data cleaning and duplicate removal
* Variable transformation
* Outlier handling
* Feature engineering
* Feature selection
* Class balancing using **SMOTE**
* Feature scaling using **StandardScaler**
* Classification model training
* Model evaluation
* Hyperparameter tuning using **GridSearchCV**
* Model serialization using **Pickle**
* Flask-based web application
* Responsive HTML/CSS frontend
* Deployment using **Render**

---

## 🛠️ Technologies Used

| Technology | Purpose |
| --- | --- |
| **Python** | Programming language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical operations |
| **SciPy** | Statistical analysis and transformations |
| **Scikit-learn** | Machine learning, preprocessing and evaluation |
| **Imbalanced-learn** | Data balancing using SMOTE |
| **Matplotlib** | Data visualization |
| **Flask** | Web application and backend |
| **HTML/CSS** | Frontend development |
| **Pickle** | Saving and loading the trained model |
| **Render** | Application deployment |

---

## 📊 Dataset

The dataset contains patient health information along with a target variable representing the presence or absence of heart disease.

The patient health attributes are used as **independent variables**, while the heart disease target is used as the **dependent/target variable**.

### Data Processing

**Pandas** and other Python libraries were used for:

* Loading the dataset
* Inspecting the dataset
* Removing duplicate records
* Resetting indexes
* Separating independent and dependent variables
* Splitting the dataset into training and testing data
* Preparing the data for machine learning

---

## 🧹 Data Cleaning

Data cleaning was performed before training the machine learning models.

The cleaning process included:

* Checking the dataset structure
* Removing duplicate records
* Resetting the dataset index
* Preparing clean data for further preprocessing

Removing unnecessary duplicate records helps prevent repeated observations from unnecessarily influencing the model.

---

## 🔧 Feature Engineering

Feature engineering and variable transformation techniques were applied to prepare the features for machine learning.

### Yeo-Johnson Transformation

The **Yeo-Johnson transformation** was applied to numerical features.

It was used to transform the feature distributions before further preprocessing.

### Outlier Handling

Outliers were handled using the **Interquartile Range (IQR)** approach.

The IQR was calculated as:

```text
IQR = Q3 - Q1
```

The lower and upper limits were determined using:

```text
Lower Limit = Q1 - 1.5 × IQR

Upper Limit = Q3 + 1.5 × IQR
```

Extreme values were handled based on these calculated boundaries.

---

## 🔍 Feature Selection

Feature selection techniques were applied to identify useful features and remove unnecessary features from the dataset.

The following methods were used:

### Constant Feature Removal

Features having no variation were identified using:

```text
VarianceThreshold(threshold = 0.0)
```

Constant features do not provide useful information for distinguishing between classes.

### Quasi-Constant Feature Removal

Features with very low variance were checked using:

```text
VarianceThreshold(threshold = 0.1)
```

This helped remove features containing very little variation.

### Correlation with Hypothesis Testing

**Pearson correlation** and hypothesis testing were used to analyze the relationship between individual features and the target variable.

The significance level used was:

```text
α = 0.05
```

Features were evaluated using their p-values to determine whether their relationship with the target was statistically significant.

---

## ⚖️ Data Balancing

The training dataset was balanced using **SMOTE (Synthetic Minority Over-sampling Technique)**.

SMOTE generates synthetic observations for the minority class to reduce class imbalance.

```text
Training Data
      ↓
    SMOTE
      ↓
Balanced Training Data
```

Balancing the training data helps the classification model learn from both target classes.

---

## 📏 Feature Scaling

Feature scaling was performed using **StandardScaler**.

StandardScaler transforms features based on their mean and standard deviation.

The standardization formula is:

```text
z = (x - μ) / σ
```

Where:

* `x` = Original feature value
* `μ` = Mean of the feature
* `σ` = Standard deviation
* `z` = Standardized value

Feature scaling is particularly useful for distance-based machine learning algorithms.

---

## 🔄 Machine Learning Workflow

```text
                 Dataset
                    ↓
              Data Cleaning
                    ↓
            Train-Test Split
                    ↓
         Variable Transformation
                    ↓
            Outlier Handling
                    ↓
           Feature Engineering
                    ↓
            Feature Selection
                    ↓
              Data Balancing
                 (SMOTE)
                    ↓
             Feature Scaling
           (StandardScaler)
                    ↓
              Model Training
                    ↓
             Model Evaluation
                    ↓
         Hyperparameter Tuning
             (GridSearchCV)
                    ↓
          Final Model Selection
                    ↓
          Model Serialization
                 (Pickle)
                    ↓
         Flask Web Application
                    ↓
                Deployment
```

---

## 🤖 Machine Learning Models

Classification algorithms were trained and evaluated to identify a suitable model for heart disease prediction.

The models were compared based on their performance on the dataset.

The classification workflow follows:

```text
Training Data
      ↓
Classification Model
      ↓
Model Training
      ↓
Prediction
      ↓
Performance Evaluation
```

---

## 🎛️ Hyperparameter Tuning

**GridSearchCV** was used for hyperparameter tuning.

GridSearchCV tests different combinations of hyperparameters and evaluates them using cross-validation to identify a suitable parameter combination for the model.

For K-Nearest Neighbors, parameters such as the following can be explored:

* Number of neighbors (`n_neighbors`)
* Weight function (`weights`)
* Distance metric (`metric`)
* Minkowski power parameter (`p`)

Example workflow:

```text
Parameter Grid
      ↓
GridSearchCV
      ↓
Cross Validation
      ↓
Compare Parameter Combinations
      ↓
Best Parameters
      ↓
Final Model
```

Hyperparameter tuning helps select model settings based on cross-validation performance rather than manually choosing parameter values.

---

## 📈 Model Evaluation

The classification models were evaluated using multiple performance metrics.

### Evaluation Metrics

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report**
* **ROC Curve**
* **AUC Score**

### Accuracy

Accuracy represents the proportion of correctly classified observations.

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

### Confusion Matrix

The confusion matrix provides information about:

```text
True Positive
True Negative
False Positive
False Negative
```

### Classification Report

The classification report provides metrics such as:

* Precision
* Recall
* F1-score
* Support

### ROC-AUC

ROC-AUC was used to analyze the classification performance of the models across different classification thresholds.

---

## 🩺 Input Features

The deployed web application accepts patient information required by the trained model.

| Feature | Description |
| --- | --- |
| **Age** | Age of the patient |
| **Sex** | Sex of the patient |
| **CP** | Chest pain type |
| **Thalach** | Maximum heart rate achieved |
| **Oldpeak** | ST depression induced by exercise |
| **Slope** | Slope of peak exercise ST segment |
| **Thal** | Thal-related feature |

These values are processed and passed to the trained machine learning model to generate a prediction.

---

## 🌐 Web Application

A **Flask web application** was developed to provide a user-friendly interface for heart disease prediction.

Users can enter the required patient information through the frontend, and the Flask backend processes the values and passes them to the trained machine learning model.

### Application Flow

```text
User enters patient details
            ↓
         HTML Form
            ↓
           Flask
            ↓
       Data Processing
            ↓
       Feature Scaling
            ↓
    Trained ML Model
            ↓
         Prediction
            ↓
 Display Prediction Result
```

The frontend was developed using **HTML and CSS** with a responsive interface for desktop and mobile devices.

---

## 🚀 Live Deployment

The application is deployed using **Render**.

### 🔗 Live Demo

**[Heart Disease Prediction – Live Application](https://ml-mini-project-e5gx.onrender.com)**

You can enter the required patient information and generate a heart disease prediction directly through the web application.

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── main.py
├── heart.csv
├── model.pkl
├── scaled.pkl
├── column_selection.py
├── variable_transformation_data.py
├── all_models.py
├── log_file.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── viha.jpeg
│
└── README.md
```

### File Description

| File/Folder | Description |
| --- | --- |
| `app.py` | Flask application and prediction logic |
| `main.py` | Main machine learning workflow |
| `heart.csv` | Dataset used for model development |
| `model.pkl` | Saved trained classification model |
| `scaled.pkl` | Saved feature scaler |
| `column_selection.py` | Feature selection operations |
| `variable_transformation_data.py` | Variable transformation and outlier handling |
| `all_models.py` | Classification model training and comparison |
| `log_file.py` | Logging configuration |
| `templates/index.html` | Frontend interface |
| `static/viha.jpeg` | Vihara Tech logo |
| `README.md` | Project documentation |

---

## ▶️ How to Run Locally

### 1. Clone the Repository

```bash
git clone <your-github-repository-link>
```

### 2. Navigate to the Project Directory

```bash
cd Heart-Disease-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib flask
```

### 6. Run the Flask Application

```bash
python app.py
```

### 7. Open the Application

After running the application, open the local address displayed in the terminal, usually:

```text
http://127.0.0.1:5000/
```

---

## 🎯 Project Objective

The main objective of this project is to develop a machine learning application capable of predicting the presence of heart disease using patient health-related features.

The project provides practical experience with the complete machine learning lifecycle:

**Data → Cleaning → Feature Engineering → Feature Selection → Balancing → Scaling → Training → Evaluation → Hyperparameter Tuning → Deployment**

It also demonstrates how a trained machine learning model can be integrated with a web application and made accessible to users through deployment.

---

## 🔮 Future Improvements

* Experiment with additional classification algorithms
* Improve feature engineering techniques
* Perform more extensive hyperparameter tuning
* Compare different data balancing techniques
* Add additional model evaluation metrics
* Add model explainability techniques
* Improve the frontend user experience
* Build a complete preprocessing pipeline for deployment
* Add probability-based prediction output
* Improve deployment and model monitoring

---

## ⚠️ Disclaimer

This project is developed for **educational and machine learning practice purposes only**.

The prediction generated by this application should **not be considered medical advice or a medical diagnosis**. Medical decisions should always be made with guidance from qualified healthcare professionals.

---

## 👨‍💻 Author

**Macha G R P Himavantha Reddy**

CSE Student | Machine Learning Enthusiast

Developed as part of Machine Learning training at **Vihara Tech**.

---

## 📄 License

This project is created for **educational and learning purposes**.
