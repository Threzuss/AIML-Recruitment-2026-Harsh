# AI-ML Recruitment 2026 

## 1. Candidate Details
* **Name:** Harsh Kumar
* **College:** SRM KTR
* **Year of Study:** 2nd Year
* **Branch:** CSE CORE

## 2. Tasks Completed
* **Task 1:** Air Quality Forecasting (Time-Series Regression)
* **Task 2:** Neural Network (MNIST Handwritten Digit Classification)

## 3. Problem Statement
* **Task 1:** To analyze historical UCI air-quality measurements and build a machine-learning regression model to predict future Carbon Monoxide (CO) levels, while strictly adhering to time-series forecasting constraints.
* **Task 2:** To design, train, and evaluate a multi-class feedforward neural network capable of classifying handwritten digits (0-9) from the MNIST dataset, and to analyze how architectural changes impact performance.

## 4. Approach
* **Task 1:** Replaced custom missing value placeholders (`-200`) and forward-filled gaps to maintain temporal continuity. Engineered time-based features (hour, day, month) and sequence features (1-hour and 24-hour lags, 6-hour rolling averages). Used a strict chronological 80/20 train-test split to train a Random Forest Regressor.
* **Task 2:** Normalized image pixel values to a `[0.0, 1.0]` scale and flattened the 28x28 matrices. Built a baseline Sequential network with a 128-unit ReLU hidden layer and a 10-unit Softmax output layer. Conducted an architectural experiment by adding a secondary 64-unit hidden layer to compare validation metrics.

## 5. Technologies Used
* **Languages:** Python
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest Regressor, Evaluation Metrics)
* **Deep Learning:** TensorFlow, Keras (Sequential API, Dense Layers)

## 6. Results
* **Task 1 (Air Quality):** The Random Forest Regressor effectively captured diurnal pollution cycles, heavily relying on the 24-hour lag feature. Evaluation yielded strong $R^2$ and minimal RMSE, though extreme anomaly spikes were slightly underestimated. 
* **Task 2 (MNIST):** The baseline model achieved ~98% test accuracy. Adding a second hidden layer marginally improved convergence speed by allowing the network to build higher-level abstract feature representations. The confusion matrix indicated rare misclassifications primarily occurred between visually overlapping digits (e.g., 4 vs. 9).

## 7. Key Learnings
1. **Handling Dataset-Specific Quirks:** Identifying that the UCI dataset utilized `-200` for missing values rather than standard nulls, which required careful replacement prior to standard imputation.
2. **Preventing Temporal Data Leakage:** Learning why standard random shuffling during train-test splitting ruins time-series forecasting, and how chronological splitting prevents the model from "peeking" into the future.
3. **Activation Function Dynamics:** Gaining practical insight into how ReLU hidden layers resolve vanishing gradients, while Softmax translates raw logits into interpretable class probabilities for the output layer.

## 8. Challenges
* **Challenge:** Dealing with sequential data dependencies without introducing future leakage during the feature engineering and modeling phases.
* **Solution:** I explicitly avoided `train_test_split` with `random_state` shuffling. Instead, I used a chronological array slice (`iloc[:split_idx]`) to ensure the Random Forest model was trained strictly on historical data to predict future unseen data.
