# XAI Bias Detection: SHAP vs. DiCE

## Overview
This repository contains the code demonstration for the Explainable AI assignment. It explores the research question: *To what extent do counterfactual explanations provide more effective detection of algorithmic bias for denied loan applicants compared to the feature-attribution insights provided by SHAP?*

The code trains a Random Forest model on a credit risk dataset and applies two explanation methods:
1. **SHAP (Global Feature Attribution):** To understand the overall impact of features like `Age` and `Sex` on loan approval.
2. **DiCE (Counterfactual Explanations):** To perform a targeted bias audit on denied female applicants, testing if isolating and flipping protected attributes changes the model's decision.

## Prerequisites
Ensure you have Python 3.8 or higher installed. You will also need Jupyter Notebook to open and run the file. The following libraries are required:
* `pandas`
* `matplotlib`
* `scikit-learn`
* `shap`
* `dice-ml`
* `jupyter`

You can install all dependencies via your terminal using:
```bash
pip install pandas matplotlib scikit-learn shap dice-ml jupyter
```
## Dataset
The project uses the [German Credit Data](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk/data) dataset. Ensure that the dataset file `german_credit_data.csv` is located in the exact same directory as the Jupyter Notebook before running.

## Usage
To run the code demonstration and reproduce the results:
1. Open your terminal or command prompt and launch Jupyter Notebook:
``` bash
jupyter notebook
```
2. In the browser window that opens, navigate to and open the `code.ipynb` file.
3. Run the cells sequentially, or click **Cell > Run All** (depending on your Jupyter environment) to execute the entire demonstration.

## Expected Output
Running the notebook cells will display the outputs inline:
1. Model Training: Prints the baseline model performance (Accuracy and Classification Report) for the Random Forest classifier.
2. SHAP Global Analysis: Generates a global SHAP summary plot showing feature impacts.
3. DiCE Targeted Audit: Executes the bias audit using DiCE. It prints the progress of generating counterfactuals and outputs the final audit results, showing the percentage of denied decisions that could be flipped by changing only 'Sex' or 'Age'.
