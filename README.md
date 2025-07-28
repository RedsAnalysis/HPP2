# House Price Prediction Intervals: 6th Place Solution



This repository contains the code for a **6th place finish** in a house price prediction competition. The goal was to predict a 90% confidence interval for sale prices, evaluated by the **Winkler Score**. Our final private leaderboard score was **$298,521.66**.

The core of this solution is a sophisticated, multi-level ensembling strategy that combines the predictions of a "Mean+Error" model with two distinct "Direct Quantile" models. The final submission is a carefully weighted blend of these three powerful pipelines.

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
2.  **Level 1: Base Model Training**: Multiple powerful models were trained for different tasks:
    -   **Mean Models (4x)**: XGBoost, CatBoost, LightGBM, and a ResNet were trained to predict the mean sale price. Their predictions were blended to create a highly accurate `ensemble_mean`.
    -   **Error Models (3x)**: XGBoost, CatBoost, and LightGBM were trained to predict the absolute error of the `ensemble_mean`. Their predictions were blended to form an `error_prediction`.
    -   **Quantile Models (2x2)**: XGBoost and LightGBM were each trained to directly predict the 5th and 95th percentiles of the sale price.
3.  **Level 2: Interval Generation & Calibration**: The predictions from Level 1 were used to construct three distinct, high-quality prediction intervals:
    -   **Pipeline A (Mean+Error)**: `ensemble_mean ± error_prediction * multiplier`
    -   **Pipeline B (XGB Quantile)**: `[lower_xgb, upper_xgb] * multiplier`
    -   **Pipeline C (LGBM Quantile)**: `[lower_lgbm, upper_lgbm] * multiplier`
    Each pipeline was individually calibrated using `scipy.optimize.minimize` to find multipliers that minimized the Winkler score on our out-of-fold (OOF) data.
4.  **Level 3: Final Blend**: The three calibrated intervals from Level 2 were blended using an optimizer to find the final weights, resulting in our best-performing submission.

## The Evaluation Metric: Winkler Score

The Winkler Score is designed for evaluating prediction intervals. It uniquely penalizes models based on two criteria:
1.  **Width**: The interval should be as narrow as possible.
2.  **Coverage**: A very large penalty is applied if the true value falls outside the predicted interval.

This forces a trade-off: an interval that is too narrow will be penalized for missing the true value, while an interval that is too wide will be penalized for its lack of precision. Our entire modeling process was designed to optimize this trade-off.

## Methodology in Detail

### Part 1: Comprehensive Feature Engineering

A robust feature set is the foundation of any successful model. Our process included:
- **Numerical Interactions**: Brute-force interactions (multiplication) between key numerical features like `grade`, `sqft`, and `area`.
- **Date Features**: Extraction of `year`, `month`, and `dayofyear` from the sale date to capture seasonality.
- **Geospatial Features**: K-Means clustering on latitude and longitude to create `location_cluster` features and calculating the distance to each cluster center.
- **Advanced Aggregations**: Grouping by categorical features like `submarket` and `city` to create aggregate statistics (e.g., `mean_price_by_submarket`).
- **Volatility Features (Key Insight)**: To specifically aid the quantile models, we engineered features that measure price dispersion. These were created in a **leakage-proof** manner inside a K-Fold loop.
    - `price_std_by_submarket`: The standard deviation of sale prices within a property's submarket.
    - `price_range_by_grade`: The difference between the max and min sale price for a property's grade.
    These features gave our quantile models a direct signal about which properties were likely to have higher or lower price uncertainty.

### Part 2: Pipeline A - The Mean+Error Model

This pipeline models the interval indirectly by first predicting the center and then the spread.
1.  **Mean Ensemble**: Four powerful models were trained to predict the mean sale price. Their OOF and test predictions were blended using an optimizer to find the optimal weights, resulting in a highly accurate `oof_ensemble_mean`.
2.  **Error Ensemble**: Three models were trained to predict the absolute error of the mean ensemble (`abs(y_true - oof_ensemble_mean)`). The features for these models included the full raw feature set *plus* the OOF predictions from the mean models (stacking).
3.  **Calibration**: The final interval was calculated as `mean ± error * multiplier`. An optimizer found the best multipliers (`a` and `b`) to minimize the OOF Winkler score. This model achieved a CV score of **$295,017**.

### Part 3: Pipeline B & C - Direct Quantile Models

This approach models the interval bounds directly. We built two separate versions using XGBoost and LightGBM.
1.  **Elite Feature Set**: We discovered that quantile models are highly sensitive to feature noise. Instead of using all 200+ features, we trained them on a smaller, "elite" set consisting of:
    - The top 25 raw features (determined by feature importance).
    - The stacked predictions from all mean and error models.
    - The two new volatility features.
2.  **Tuning**: We used Optuna to perform a comprehensive hyperparameter search (50 trials each) for two separate models: one with `objective='quantile'` and `alpha=0.05` (lower bound) and another with `alpha=0.95` (upper bound).
3.  **Training**: The tuned models were trained using 5-Fold StratifiedKFold to generate robust OOF and test predictions.
4.  **Calibration**: We discovered significant "quantile crossing" (where `lower > upper`). We corrected this by enforcing `lower <= upper` and then used an optimizer to find calibration multipliers.

The LightGBM Quantile model (**$293,068**) proved to be superior to the XGBoost Quantile model (**$349,311**).

### Part 4: The Final 3-Model Ensemble

Our best submission came from blending the three calibrated intervals.
- **Optimizer**: We used `scipy.optimize.minimize` to find the optimal weights to apply to each pipeline's lower and upper bounds.
- **Final Weights**: The optimizer revealed the true strength of each pipeline:
    - **LGBM Quantile (Model C): 62.70%** -> The best single model.
    - **Mean+Error (Model A): 37.30%** -> A highly valuable, diverse contributor.
    - **XGB Quantile (Model B): 0.00%** -> Provided no additional value once A and C were included.

This blend achieved our best CV score of **$291,785**, leading to our final 6th place finish.

## Key Insights and Lessons Learned

1.  **Volatility Features are Crucial for Quantile Models**: The performance of our LightGBM Quantile model improved dramatically after adding features that explicitly described price dispersion.
2.  **Quantile Models Require Aggressive Feature Selection**: Direct quantile models are more sensitive to noise than mean models. Reducing the feature set from 200+ to ~30 elite features was key to their success.
3.  **Ensembling Diverse Strategies is Key**: Our final score was not achieved by a single model, but by blending two fundamentally different and high-performing approaches (Mean+Error vs. Direct Quantile).
4.  **Always Trust the CV Score**: The optimizer correctly identified our best two models and their optimal weights, which translated directly to a high leaderboard rank. The final blend of our best pipelines was the key to securing a top position.

## How to Reproduce the Solution

The solution is structured across several notebooks, designed to be run in sequence. All required prediction artifacts (`.npy` files) are saved at each step.

1.  **Notebook 1 - `01_feature_engineering.ipynb`**: Creates the base feature sets (`X`, `X_test`).
2.  **Notebook 2 - `02_train_mean_error_models.ipynb`**: Trains the 4 mean and 3 error models, saving their OOF and test predictions to disk.
3.  **Notebook 3 - `03_train_quantile_models.ipynb`**: Contains the full pipeline for both XGBoost and LightGBM quantile models, including elite feature creation, Optuna tuning, and K-Fold training. Saves all quantile predictions to disk.
4.  **Notebook 4 - `04_final_ensemble.ipynb`**: Loads all predictions from the previous notebooks, calibrates each pipeline, and runs the final 3-model blend to generate the winning submission file.

The `data` directory should contain the original `dataset.csv` and `test.csv` files. Each notebook will create directories to store the model prediction outputs.