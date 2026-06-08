# Cook County Housing Price Prediction

A two-part data science project that builds a machine learning pipeline to predict residential property sale prices in Cook County, Illinois using real-world assessor data. Completed as part of CSCI 3022 (Data Science) at the University of Colorado Boulder.

---

## Project Overview

Cook County, Illinois — home to Chicago — relies on property assessments to determine tax obligations for hundreds of thousands of homeowners. Inaccurate assessments can create regressive tax burdens, disproportionately affecting lower-income residents. This project uses a dataset from the [Cook County Assessor's Office (CCAO)](https://datacatalog.cookcountyil.gov/Property-Taxation/Archive-Cook-County-Assessor-s-Residential-Sales-D/5pge-nu6u) to explore, clean, engineer features from, and ultimately build a predictive model for residential home sale prices.

The project is divided into two parts: an exploratory analysis and feature engineering phase, followed by a linear regression modeling and evaluation phase.

---

## The Dataset

The dataset contains over **500,000 residential property records** from Cook County, Illinois, split into a training set (~204,000 records) and a test set (~68,000 records). Each record represents a single property transaction and includes **62 features** spanning:

| Category | Example Features |
|---|---|
| Physical characteristics | Building Square Feet, Bedrooms, Roof Material, Wall Material, Basement Type, Attic Type, Fireplaces, Garage Size |
| Location | Latitude, Longitude, Neighborhood Code, Town Code, Census Tract, Road Proximity, Floodplain |
| Sale information | Sale Price, Sale Year, Sale Month, Deed Number |
| Property condition | Age, Repair Condition, Construction Quality, Site Desirability |

The target variable is **Sale Price** (log-transformed to `Log Sale Price` for modeling). A full variable description is available in `codebook.txt`.

---

## Part 1: Exploratory Data Analysis & Feature Engineering

**Notebook:** [Project Part 1/ProjPart1.ipynb](Project%20Part%201/ProjPart1.ipynb)

### Goals
Understand the structure and content of the Cook County housing dataset, identify data quality issues, and engineer new features in preparation for predictive modeling.

### What Was Done

**1. Data Contextualization**
- Analyzed the granularity of the dataset (each row = one property transaction)
- Explored the social and governmental context of property tax assessment data
- Formulated analytical questions about Cook County housing trends

**2. Exploratory Data Analysis (EDA)**
- Visualized the distribution of `Sale Price` using histograms and box plots
- Applied a log transformation to `Sale Price` to address right-skew, producing `Log Sale Price`
- Filtered out non-market sales (e.g., $1 transfers between family members) by removing records under $500
- Examined the relationship between `Log Building Square Feet` and `Log Sale Price` using joint plots and linear regression overlays, confirming a strong positive linear association

**3. Outlier Removal**
- Wrote a reusable `remove_outliers()` function that filters a DataFrame to a specified numeric range for any given column
- Applied it to `Building Square Feet` to remove extreme values that skewed visualizations

**4. Feature Engineering**
- **Bedrooms**: Used regular expressions (regex) to extract the total number of bedrooms from unstructured text in the `Description` column
- **Neighborhood Analysis**: Identified the top 20 most populous neighborhoods and visualized `Log Sale Price` distributions across them using box plots and count plots
- **Expensive Neighborhood Indicator**: Ranked neighborhoods by median `Log Sale Price` and created a binary `in_expensive_neighborhood` indicator variable for the top 3
- **Roof Material Decoding**: Replaced numeric codes in the `Roof Material` column with human-readable string labels (e.g., `1` → `"Shingle/Asphalt"`)
- **One-Hot Encoding**: Used Scikit-learn's `OneHotEncoder` to encode the categorical `Roof Material` column into binary indicator columns (e.g., `Roof Material_Shingle/Asphalt`, `Roof Material_Tar&Gravel`) without dropping the original column

---

## Part 2: Linear Regression Modeling & Evaluation

**Notebook:** [Project Part 2/ProjPart2.ipynb](Project%20Part%202/ProjPart2.ipynb)

### Goals
Build and iteratively improve a linear regression model to predict `Log Sale Price`, evaluate model performance using RMSE and residual analysis, and critically examine the fairness and ethical implications of algorithmic property assessment.

### What Was Done

**1. Ethics & Human Context Analysis**
- Investigated the history of property tax inequity in Cook County as reported by the Chicago Tribune
- Analyzed how model errors (residuals) translate to over- or under-taxation for individual homeowners
- Evaluated different assessment bias scenarios (regressive vs. progressive taxation) and identified which residual plot structures correspond to each

**2. Data Preprocessing Pipeline**
- Built a feature engineering pipeline that reproduces all transformations from Part 1 in a single reusable function, ensuring consistent preprocessing across training and validation data

**3. Model 1 — Simple Linear Regression**
- Fit a baseline linear regression model:
  `Log Sale Price = θ₁ × (Log Building Square Feet) + θ₀`
- Used Scikit-learn's `LinearRegression` with `fit_intercept=True`
- Evaluated using Root Mean Squared Error (RMSE) on both training and validation sets
- Performed 5-fold cross-validation to verify that validation RMSE was representative
- Visualized residual plots (residuals vs. predicted values, and residuals vs. actual values) to identify bias patterns

**4. Model 2 — Adding Roof Material**
- Extended Model 1 by adding one-hot encoded `Roof Material` features as predictors
- Observed only a marginal RMSE improvement and investigated why (insufficient variation in rare roof material categories)

**5. Model 3 — Custom Feature Addition**
- Selected and justified an additional quantitative feature to improve model accuracy
- Analyzed RMSE and residual plots for improvement over Models 1 and 2
- Discussed remaining bias patterns and proposed next steps

**6. Model Fairness Evaluation**
- Connected model accuracy to equitable property tax assessment
- Identified regressive bias patterns in residual plots (overestimating low-value homes, underestimating high-value homes)
- Discussed what it means for a predictive model to produce "fair" tax assessments in practice

---

## Skills Demonstrated

This project demonstrates a range of skills relevant to data science, analytics, and machine learning roles:

### Programming & Tools
- **Python** — pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Jupyter Notebooks** — end-to-end data science workflow in an interactive environment
- **Regular Expressions (regex)** — extracting structured data from unstructured text

### Data Engineering
- Data cleaning and preprocessing (handling missing values, filtering outliers, encoding categoricals)
- Feature engineering (log transformations, indicator variables, one-hot encoding, text extraction)
- Building reusable preprocessing pipelines

### Statistical Analysis & Machine Learning
- Exploratory Data Analysis (EDA) and statistical summarization
- Linear regression modeling (simple and multivariate)
- Model evaluation: RMSE, cross-validation, residual analysis
- Bias detection and model fairness assessment

### Data Visualization
- Distribution plots (histograms, KDE, box plots)
- Scatter/joint plots with regression overlays
- Residual plots for model diagnostics
- Categorical comparisons across groups

### Critical Thinking & Communication
- Interpreting model outputs in a real-world policy context
- Communicating findings and limitations clearly
- Ethical reasoning around algorithmic decision-making in government systems

---

## Project Structure

```
CSCI3022DataScienceProject/
├── Project Part 1/
│   ├── ProjPart1.ipynb       # EDA and feature engineering notebook
│   ├── ProjPart1.pdf         # PDF export of notebook
│   ├── cook_county_train.csv # Training dataset
│   ├── cook_county_test.csv  # Test dataset
│   └── data/
│       └── codebook.txt      # Variable descriptions
├── Project Part 2/
│   ├── ProjPart2.ipynb           # Modeling and evaluation notebook
│   ├── ProjPart2.pdf             # PDF export of notebook
│   ├── cook_county_train_val.csv # Training/validation dataset
│   ├── cook_county_contest_test.csv # Competition test dataset
│   ├── codebook.txt              # Variable descriptions
│   ├── ds100_utils.py            # Utility functions
│   └── feature_func.py           # Feature engineering functions
└── README.md
```

---

## Course Context

- **Course**: CSCI 3022 — Introduction to Data Science (University of Colorado Boulder)
- **Dataset Source**: [Cook County Assessor's Office](https://datacatalog.cookcountyil.gov/Property-Taxation/Archive-Cook-County-Assessor-s-Residential-Sales-D/5pge-nu6u)
