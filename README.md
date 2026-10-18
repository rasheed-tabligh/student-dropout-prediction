# Student Dropout Prediction

Predicting whether a student will graduate, drop out, or remain enrolled, using the UCI Student Dropout and Academic Success dataset.

## About the Project

Universities lose students to dropout every year, and by the time it is obvious, it is usually too late to help. This project asks a practical question: can an institution identify students who need support early enough to change the outcome? We frame it as a 3-class prediction problem (Graduate, Dropout, Enrolled) on real data from a Portuguese higher education institution, published by the UCI Machine Learning Repository. Built as a team project during iXperience Data Science 2026.

## About the iXperience Program

iXperience is an immersive summer program where university students work through an intensive data science curriculum and apply it to a real project in small teams. This repo is the output of our group's project for the Data Science 2026 cohort.

## Dataset

- UCI ML Repository, dataset ID 697
- About 4,400 students, 36 features (demographic, academic, socioeconomic)
- Target: `Dropout`, `Enrolled`, or `Graduate`
- Classes are imbalanced: roughly 50% Graduate, 32% Dropout, 18% Enrolled

## Approach

EDA showed that academic performance in the first two semesters carries the strongest signal, especially the gap between units enrolled and units approved. We engineered an approval ratio feature (approved units divided by enrolled units) plus financial risk flags, since tuition status and debtor status turned out to be strong predictors on their own. Each team member then built and evaluated a model: Logistic Regression, Random Forest, XGBoost, and SVM.

## What's in This Repo

- `notebooks/eda.ipynb` is the exploratory data analysis, by Karim Ayoub
- `notebooks/logistic_regression.ipynb` is the Logistic Regression model, by Rashid Mikidadi
- The Random Forest, XGBoost, and SVM models were built by other team members and are presented in the slide deck under `docs/`, not as code in this repo

## My Logistic Regression Model

A multiclass Logistic Regression on scaled features, with a stratified 80/20 train/test split.

- Overall accuracy: 76.8%
- Dropout recall: 0.77 (the model catches about 3 out of 4 actual dropouts)
- Dropout AUC: 0.91, so it ranks at-risk students very reliably
- Graduate: recall 0.93, AUC 0.92. Enrolled is the hardest class (recall 0.32, AUC 0.80)

Because Logistic Regression is interpretable, the coefficients show what drives dropout. The strongest protective factors are passing the units you enroll in (2nd semester approved units and the approval ratio) and keeping tuition fees up to date. Risk rises with units enrolled but not passed and with older age at enrollment. The notebook ends by ranking students by predicted dropout probability, which is the list an institution would use to decide who to reach out to first.

## Full Team Results Overview

- Random Forest (binary, dropout vs not): high accuracy, catches most dropouts
- XGBoost (3-class): Macro F1 0.717, Graduate recall 0.91, Dropout recall 0.75, Enrolled recall 0.46
- XGBoost (binary on resolved students): Macro F1 0.93, Dropout recall 0.93, ROC-AUC 0.972
- SVM (3-class, RBF kernel): 72% accuracy, Dropout AUC 0.91, Graduate AUC 0.92

Full details for all four models are in the slide deck under `docs/`.

## Repository Structure

```
student-dropout-prediction/
├── README.md
├── CLAUDE.md
├── notebooks/
│   ├── eda.ipynb
│   └── logistic_regression.ipynb
├── data/
│   └── data.csv
└── docs/
    └── (slide deck)
```

## How to Run

Requires Python 3 with pandas, numpy, matplotlib, seaborn, and scikit-learn:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then open the notebooks with Jupyter:

```
jupyter notebook notebooks/
```

The notebooks load the dataset from `data/data.csv` (note: it is semicolon-separated).

## Team

- Rashid Mikidadi (Logistic Regression, ROC analysis, dropout risk ranking)
- Karim Ayoub (EDA, feature engineering)
- Tanya Karpinchyk (Random Forest)
- Tayveon Moore (XGBoost)
- Rea Lila (SVM)

## Acknowledgements

Thanks to iXperience and our instructors for the teaching and guidance throughout the program, and to the UCI Machine Learning Repository for hosting the dataset.
