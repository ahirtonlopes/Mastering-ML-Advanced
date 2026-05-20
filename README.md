# Mastering Machine Learning Advanced

Open courseware covering advanced Machine Learning topics — Feature Engineering, Model Evaluation, Time Series, NLP, and MLOps · Scikit-learn · Python

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What you'll learn

- Transform raw data into high-quality features with real-world datasets
- Evaluate models correctly using cross-validation and proper metrics
- Forecast time series with classical and deep learning approaches
- Apply NLP techniques from TF-IDF to BERT fine-tuning
- Deploy and monitor ML models in production

## Prerequisites

| Topic | Level |
|---|---|
| Python | Intermediate |
| Pandas & NumPy | Intermediate |
| Scikit-learn basics | Basic |
| [Mastering Machine Learning](https://github.com/ahirtonlopes/Mastering-Machine-Learning) | Recommended |

---

## Contents

| # | Notebook | Topics |
|---|---|---|
| 01 | `Aula1_Feature_Engineering.ipynb` | Missing values, outliers, encoding, scaling, feature creation, feature selection, pipeline |
| 02 | `Aula2_CrossValidation_Hyperparameter_Tuning.ipynb` | K-Fold, Stratified K-Fold, metrics (Precision/Recall/AUC-ROC), learning curves, Grid Search, Randomized Search, Optuna |
| 03 | `Aula3_Series_Temporais.ipynb` | Time series decomposition, stationarity, ADF test, ARIMA, Prophet, LSTM, model comparison |
| 04 | `Aula4_NLP_Aplicado.ipynb` | Text preprocessing, TF-IDF, Naive Bayes, SVM, word embeddings, BiLSTM, DistilBERT fine-tuning |
| 05 | `Aula5_MLOps_Basico.ipynb` | MLflow experiment tracking, model serialization, FastAPI serving, Evidently data drift monitoring |

---

## ML Engineer Exercises

Hands-on programming exercises for those who want to build ML systems, not just call APIs. Each notebook is self-contained with skeleton code, assertions, and validation cells.

| # | Notebook | Skills |
|---|---|---|
| Ex01 | `Exercicios/Ex01_Gradient_Descent.ipynb` | Vanilla GD, Momentum, Adam optimizer from scratch |
| Ex02 | `Exercicios/Ex02_Algoritmos_do_Zero.ipynb` | KNN, K-Means, Decision Tree from scratch |
| Ex03 | `Exercicios/Ex03_Custom_Sklearn_API.ipynb` | Custom transformers, stateful fitting, Pipeline + GridSearchCV |
| Ex04 | `Exercicios/Ex04_Backpropagation.ipynb` | Backprop, forward/backward pass, training loop from scratch |
| Ex05 | `Exercicios/Ex05_Regularizacao.ipynb` | L1/L2 regularization, Dropout, Early Stopping |
| Ex06 | `Exercicios/Ex06_Debugging_ML.ipynb` | Data leakage, wrong metrics, target leakage, CV bugs |

---

## Getting Started

**Option 1 — Google Colab (recommended)**

Open any notebook directly on GitHub and click **Open in Colab** at the top of the file.

**Option 2 — Local**

```bash
git clone https://github.com/ahirtonlopes/Mastering-ML-Advanced.git
cd Mastering-ML-Advanced
python -m venv .venv && source .venv/bin/activate
pip install scikit-learn pandas numpy matplotlib seaborn scipy jupyter
jupyter notebook
```

---

## Author

**Prof. Dr. Ahirton Lopes** · [LinkedIn](https://linkedin.com/in/ahirtonlopes) · [Google Scholar](https://scholar.google.com/citations?user=1SQDVrwAAAAJ)

Contributions are welcome — open an issue or submit a pull request.

## License

[MIT](LICENSE)
