# Bias Detection in Credit Scoring: SHAP vs. DiCE
This repository contains the code demonstration for the Explainable AI assignment. It explores the research question: _To what extent do counterfactual explanations provide more effective detection of algorithmic bias for denied loan applicants compared to the feature-attribution insights provided by SHAP?_

## Overview
The code trains a Random Forest model on a credit risk dataset and applies two explanation methods:
1. **SHAP (Global Feature Attribution):** To understand the overall impact of features like `Age` and `Sex` on loan approval.
2. **DiCE (Counterfactual Explanations):** To perform a targeted bias audit on denied female applicants, testing if isolating and flipping protected attributes changes the model's decision.

## Requirements and Installation
Ensure you have Python 3.8 or higher installed. You will also need Jupyter Notebook to open and run the file.

1. Clone the repository and enter the directory
``` bash
git clone git@github.com:saragarzou/xai-bias-detection-credit.git && cd xai-bias-detection-credit
```
2. Create a virtual environment in your project directory:
``` bash
python3 -m venv xai-venv
```
3. Activate the virtual environment:
* On Windows: `xai-venv\Scripts\activate`
* On macOS/Linux: `source xai-venv/bin/activate`
  
4. Install all required dependencies using the requirements file:
``` bash
pip install -r requirements.txt
```
5. Register the environment as a Jupyter Kernel:
``` bash
python3 -m ipykernel install --user --name xai-venv --display-name "XAI Bias Detection"
```

Key libraries included in the requirements:
* `pandas`
* `matplotlib`
* `scikit-learn`
* `shap`
* `dice-ml`
* `jupyter`

## Dataset
The project uses the [German Credit Data](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk/data) dataset. Ensure that the dataset file `german_credit_data.csv` is located in the exact same directory as the Jupyter Notebook before running.

## Usage
To run the code demonstration and reproduce the results:
1. Ensure your virtual environment is activated.
2. Open your terminal or command prompt and launch Jupyter Notebook:
``` bash
jupyter notebook
```
3. In the browser window that opens, navigate to and open the `code.ipynb` file.
4. Make sure your kernel is selected: **Kernel > Change Kernel > XAI Bias Detection**
5. Run the cells sequentially, or click **Cell > Run All** (depending on your Jupyter environment) to execute the entire demonstration.

## Expected Output
Running the notebook cells will display the outputs inline:
1. Model Training: Prints the baseline model performance (Accuracy and Classification Report) for the Random Forest classifier.
2. SHAP Global Analysis: Generates a global SHAP summary plot showing feature impacts.
3. DiCE Targeted Audit: Executes the bias audit using DiCE. It prints the progress of generating counterfactuals and outputs the final audit results, showing the percentage of denied decisions that could be flipped by changing only `Sex` or `Age`.
