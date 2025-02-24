# 🔬 Glucose Prediction using Machine Learning

This project implements a machine learning pipeline to predict the likelihood of diabetes based on clinical parameters. It involves data preprocessing, exploratory data analysis (EDA), feature scaling, and model training using various classification algorithms. Advanced techniques like model stacking and blending are also employed to enhance performance.

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Features and Exploratory Data Analysis (EDA)](#features-and-exploratory-data-analysis-eda)
4. [Modeling Approach](#modeling-approach)
5. [Results](#results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Technologies Used](#technologies-used)
9. [Directory Structure](#directory-structure)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

---

## 🌟 Project Overview

This project aims to build an accurate and interpretable machine learning model for diabetes prediction using the PIMA Indians Diabetes Dataset. The pipeline includes:

- Exploratory Data Analysis (EDA) with Seaborn and Plotly.
- Data Cleaning and Imputation for missing values.
- Feature Engineering and Scaling.
- Model training using PyCaret, including:
  - Logistic Regression
  - Random Forest
  - CatBoost
  - Gradient Boosting
  - Linear Discriminant Analysis (LDA)
- Advanced techniques like model stacking and blending.
- Model evaluation using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

---

## 📊 Dataset Description

The dataset used is the **PIMA Indians Diabetes Dataset**, commonly used for binary classification tasks. It contains the following features:

| Feature              | Description                                              |
|----------------------|----------------------------------------------------------|
| Pregnancies          | Number of times the patient has been pregnant             |
| Glucose              | Plasma glucose concentration                             |
| BloodPressure        | Diastolic blood pressure (mm Hg)                         |
| SkinThickness        | Triceps skinfold thickness (mm)                          |
| Insulin              | 2-Hour serum insulin (mu U/ml)                          |
| BMI                  | Body mass index (weight in kg/(height in m)^2)           |
| DiabetesPedigreeFunc | Diabetes pedigree function (genetic predisposition)      |
| Age                  | Age of the patient (years)                               |
| Outcome              | Binary target variable (1 = Diabetes, 0 = No Diabetes)   |

---

## 🔍 Features and Exploratory Data Analysis (EDA)

Key steps in EDA include:

1. **Missing Value Analysis:**  
   - Visualized using `missingno` matrix and imputed with feature means for zero values.
   
2. **Class Distribution:**  
   - Donut pie chart showing the ratio of diabetic vs. non-diabetic patients.

3. **Feature Distribution:**  
   - Histograms for each feature, segmented by diabetes outcome.

4. **Correlation Heatmap:**  
   - Showcasing feature correlations to identify multicollinearity.

5. **UMAP for Dimensionality Reduction:**  
   - 2D and 3D UMAP projections for visualizing separability between classes.

---

## 🤖 Modeling Approach

The project implements the following machine learning techniques:

1. **Baseline Models:**  
   - Logistic Regression, Random Forest, LDA, Gradient Boosting, CatBoost.

2. **Model Tuning:**  
   - Hyperparameter optimization using PyCaret’s `tune_model` function.

3. **Model Stacking:**  
   - Combining top 5 models using stacking for improved performance.

4. **Model Blending:**  
   - Soft and hard blending for robust predictions.

5. **Calibration:**  
   - Probability calibration using PyCaret’s `calibrate_model` method.

6. **Final Model:**  
   - The best-performing model finalized and evaluated.

---

## 📈 Results

Performance metrics for the final model include:

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 0.87   |
| Precision   | 0.85   |
| Recall      | 0.83   |
| F1-Score    | 0.84   |
| ROC-AUC     | 0.91   |

Confusion matrices for stacking, soft blending, and hard blending are plotted using Seaborn heatmaps.

---

## ⚙️ Installation

Ensure Python 3.8+ is installed. Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Required packages include:**  
`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `pycaret`, `plotly`, `umap-learn`, and `missingno`.

---

## 🚀 Usage

1. **Clone the Repository:**

```bash
git clone https://github.com/yourusername/glucose-prediction.git
cd glucose-prediction
```

2. **Run the Script:**

```bash
python glucose_pred.py
```

3. **Expected Output:**

- Visualizations for EDA and Feature Correlations.
- Model Comparison and Evaluation Metrics.
- ROC Curves, Precision-Recall Curves, and Confusion Matrices.
- Final Model Prediction for unseen test data.

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3.8+  
- **Libraries:**  
  - Data Handling: `pandas`, `numpy`  
  - Visualization: `seaborn`, `matplotlib`, `plotly`  
  - Machine Learning: `scikit-learn`, `pycaret`, `catboost`  
  - Dimensionality Reduction: `umap-learn`  
  - Data Cleaning: `missingno`

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

1. Fork the repository.  
2. Create a new branch (`git checkout -b feature-branch`).  
3. Commit changes (`git commit -m 'Add new feature'`).  
4. Push to the branch (`git push origin feature-branch`).  
5. Open a Pull Request.

---
