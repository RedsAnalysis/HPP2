# House Price Prediction Intervals: 6th Place Solution

<p align="center">
  <img src="Screenshot 2025-07-28 051058.png">
</p>

This repository contains the code for the final, best-performing model that secured a **6th place finish** in the [House Price Prediction Interval Competition II on Kaggle](https://www.kaggle.com/competitions/prediction-interval-competition-ii-house-price/leaderboard).

The solution presented here is the culmination of extensive experimentation. This notebook represents the champion strategy that emerged after numerous trial-and-error approaches. For a complete history of all experiments, including different feature sets, modeling strategies, and calibration techniques that were tested, please see the full experimentation repository: [RedsAnalysis/Kaggle_Competitions](https://github.com/RedsAnalysis/Kaggle_Competitions).

Our final private leaderboard score was **$298,521.66**, with a cross-validation score of **$291,785.50**. The core of this solution is a sophisticated, multi-level ensembling strategy that combines the predictions of a "Mean+Error" model with two distinct "Direct Quantile" models.

## Table of Contents
1. [Final Solution Architecture](#final-solution-architecture)
2. [The Evaluation Metric: Winkler Score](#the-evaluation-metric-winkler-score)
3. [Methodology in Detail](#methodology-in-detail)
    - [Part 1: Comprehensive Feature Engineering](#part-1-comprehensive-feature-engineering)
    - [Part 2: Pipeline A - The Mean+Error Model](#part-2-pipeline-a---the-meanerror-model)
    - [Part 3: Pipeline B & C - Direct Quantile Models](#part-3-pipeline-b--c---direct-quantile-models)
    - [Part 4: The Final 3-Model Ensemble](#part-4-the-final-3-model-ensemble)
4. [Key Insights and Lessons Learned](#key-insights-and-lessons-learned)
5. [How to Reproduce the Solution](#how-to-reproduce-the-solution)

---

## Final Solution Architecture

The final model is an "Ensemble of Ensembles," a robust strategy that leverages the strengths of different modeling approaches. The architecture is layered, with each level building upon the previous one.

1.  **Level 0: Feature Engineering**: A rich set of features was created, including specialized "volatility features" designed to help predict the width of the price interval.
2.  **Level 1: Base Model Training**: Multiple powerful models were trained for different tasks, with their predictions saved to disk.
    -   **Mean Models (4x)**: XGBoost, CatBoost, LightGBM, and a ResNet were trained to predict the mean sale price. Their predictions were blended to create a highly accurate `ensemble_mean`.
    -   **Error Models (3x)**: XGBoost, CatBoost, and LightGBM were trained to predict the absolute error of the `ensemble_mean`.
    -   **Quantile Models (2x2)**: XGBoost and LightGBM were each trained to directly predict the 5th and 95th percentiles of the sale price.
3.  **Level 2: Interval Generation & Calibration**: The predictions from Level 1 were used to construct three distinct, high-quality prediction intervals:
    -   **Pipeline A (Mean+Error)**: `ensemble_mean ± blended_error * multiplier`
    -   **Pipeline B (XGB Quantile)**: `[lower_xgb, upper_xgb] * multiplier`
    -   **Pipeline C (LGBM Quantile)**: `[lower_lgbm, upper_lgbm] * multiplier`
    Each pipeline was individually calibrated using `scipy.optimize.minimize` to find multipliers that minimized the Winkler score on out-of-fold (OOF) data.
4.  **Level 3: Final Blend**: The three calibrated intervals from Level 2 were blended using an optimizer to find the final weights, resulting in our best-performing submission.

## The Evaluation Metric: Winkler Score

The Winkler Score is designed for evaluating prediction intervals. It uniquely penalizes models based on two criteria:
1.  **Width**: The interval should be as narrow as possible.
2.  **Coverage**: A very large penalty is applied if the true value falls outside the predicted interval.

This forces a trade-off: an interval that is too narrow will be penalized for missing the true value, while an interval that is too wide will be penalized for its lack of precision. Our entire modeling process was designed to optimize this trade-off.

## Methodology in Detail

### Part 1: Comprehensive Feature Engineering

A robust feature set is the foundation of any successful model. The main feature engineering is encapsulated in the `create_comprehensive_features` function and includes:
- **Numerical Interactions**: Brute-force interactions between key numerical features like `grade` and `sqft`.
- **Date Features**: Extraction of `year`, `month`, and `dayofyear` from the sale date.
- **Geospatial Features**: K-Means clustering on latitude and longitude to create `location_cluster` features and distances.
- **Advanced Aggregations**: Grouping by categoricals like `submarket` and `city` to create aggregate statistics (e.g., `mean_price_by_submarket`).

### Part 2: Pipeline A - The Mean+Error Model

This pipeline models the interval indirectly by first predicting the center and then the spread. The training of these base models was performed in separate notebooks.
1.  **Mean Ensemble**: Four powerful models' predictions were loaded and blended to create the `oof_ensemble_mean`.
2.  **Error Ensemble**: Three error models' predictions were loaded.
3.  **Calibration**: The final interval was calculated as `mean ± blended_error * multiplier`. This pipeline achieved a standalone CV score of **$295,017.11**.

### Part 3: Pipeline B & C - Direct Quantile Models

This approach models the interval bounds directly. We built two separate versions using XGBoost and LightGBM.
1.  **Elite Feature Set**: We discovered that quantile models are highly sensitive to noise. The feature set for these models was specially constructed:
    - The top 25 raw features (determined by XGBoost feature importance).
    - The stacked predictions from all 4 mean and 3 error models.
    - **Volatility Features (Key Insight)**: We engineered two features to measure price dispersion in a **leakage-proof** manner inside a K-Fold loop: `price_std_by_submarket` and `price_range_by_grade`. These were crucial for success.
2.  **Tuning & Training**:
    - The XGBoost Quantile model was trained in a separate process and its predictions were loaded.
    - The LightGBM Quantile model was **tuned with Optuna and trained with K-Fold within this notebook** to find the optimal hyperparameters.
3.  **Calibration**: Both pipelines were calibrated to correct for issues like quantile crossing.

The LightGBM Quantile model (**$293,068.04**) proved superior to the XGBoost Quantile model (**$349,311.64**).

### Part 4: The Final 3-Model Ensemble

Our best submission came from blending the three calibrated intervals.
- **Optimizer**: We used `scipy.optimize.minimize` to find the optimal weights to apply to each pipeline's lower and upper bounds.
- **Final Weights**: The optimizer revealed the true strength of each pipeline:
    - **LGBM Quantile (Model C): 62.70%** -> The best single model.
    - **Mean+Error (Model A): 37.30%** -> A highly valuable, diverse contributor.
    - **XGB Quantile (Model B): 0.00%** -> Provided no additional value once A and C were included.

This blend achieved our best CV score of **$291,785.50**, leading to our final 6th place finish.

## Key Insights and Lessons Learned

1.  **Volatility Features are Crucial for Quantile Models**: The performance of our LightGBM Quantile model improved dramatically after adding features that explicitly described price dispersion.
2.  **Quantile Models Require Aggressive Feature Selection**: Direct quantile models are more sensitive to noise than mean models. Reducing the feature set from 200+ to an elite set of ~30 features was key to their success.
3.  **Ensembling Diverse Strategies is Key**: Our final score was not achieved by a single model, but by blending two fundamentally different and high-performing approaches (Mean+Error vs. Direct Quantile).
4.  **Always Trust the CV Score**: The optimizer correctly identified our best two models and their optimal weights, which translated directly to a high leaderboard rank. The final blend of our best pipelines was the key to securing a top position.

## How to Reproduce the Solution

This repository contains the single master notebook (`King_Final_291k_298k.ipynb`) that generates the final, winning submission.

**Prerequisites:**
To run this notebook successfully, you must have the pre-computed prediction files (`.npy`) for the Level 1 models. The training for these models can be found in the main experimentation repository. The required files are:
- **Mean Model Predictions**: `oof_xgb_preds.npy`, `test_xgb_preds.npy`, etc. (for 4 models)
- **Error Model Predictions**: `oof_error_preds_xgb.npy`, `test_error_preds_xgb.npy`, etc. (for 3 models)
- **XGB Quantile Model Predictions**: `oof_lower_preds.npy`, `test_lower_preds.npy`, etc.

**Running the Notebook:**
Once the prerequisites are in place, running the `King_Final_291k_298k.ipynb` notebook from top to bottom will perform the following steps:
1.  **Block 1-2**: Setup and base feature engineering.
2.  **Block 3**: Load all prerequisite `.npy` prediction files.
3.  **Block 4-10**: Construct the elite feature set, tune the LightGBM quantile models with Optuna.
4.  **Block 11**: Train the tuned LightGBM quantile models using K-Fold and save their predictions.
5.  **Block 12**: Load all pipeline predictions, perform final calibration, run the 3-model blend, and generate the final submission file `submission_FINAL_3M_BLEND_291785.csv`.
