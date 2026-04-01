# Master-Thesis
AI and Machine Learning Analysis of Vessel Performance and Fuel Consumption
===========================================================================

Overview
--------
This project analyzes operational data from the vessel "NISSOS SIFNOS" to develop machine learning models for understanding and predicting fuel consumption and vessel performance under varying operating conditions.

The analysis focuses on identifying the relationship between environmental factors, vessel loading conditions, and engine parameters, with the objective of improving operational efficiency and detecting performance degradation.

Dataset Description
-------------------
The dataset consists of high-frequency time-series measurements collected from vessel sensors over multiple operational periods (2020–2023).

Each record includes navigational, environmental, and engine-related variables, such as:

- Vessel speed over water (SOW) and speed over ground (SOG)
- Main engine fuel oil consumption (M/E FOC, tons/day)
- Engine RPM and brake power (kW)
- Wind force (Beaufort scale) and wind direction
- Sea current speed and direction
- Draft measurements (forward, aft, mean)
- Trim and loading condition (ballast vs laden)
- Cargo weight
- Emissions data (CO2, NOx, SO2 flow rates)

The dataset is segmented into multiple time periods and operational modes (e.g., ballast vs laden conditions).

Problem Definition
------------------
The primary objective is to model and analyze fuel consumption behavior (M/E FOC) as a function of:

- Vessel speed and propulsion characteristics
- Environmental conditions (wind, currents)
- Hull condition indicators (e.g., slip)
- Loading condition and draft

This is treated as a supervised learning problem, primarily regression-focused.

Data Processing
---------------
- Filtering of invalid or noisy sensor readings
- Segmentation based on voyage and loading status
- Feature selection to isolate key operational variables
- Handling of missing values and inconsistent measurements
- Normalization of features where required

Special attention is given to separating ballast and laden conditions to ensure consistent model behavior.

Exploratory Analysis
--------------------
The analysis highlights:

- Strong correlation between fuel consumption and engine load (RPM, brake power)
- Sensitivity of fuel consumption to environmental resistance (wind force, currents)
- Observable differences in performance between ballast and laden conditions
- Impact of vessel slip on propulsion efficiency

Visualization is used extensively to examine trends and detect anomalies.

Model Development
-----------------
Multiple machine learning models are implemented to predict fuel consumption:

- Regression Models (scikit-learn)
  Baseline models for interpretability and benchmarking

- Ensemble Methods (e.g., XGBoost)
  Used to capture nonlinear relationships and interactions between operational variables

- Neural Networks
  Applied to model complex dependencies across environmental and engine parameters

Hyperparameter tuning is performed to optimize predictive performance.

Evaluation Strategy
-------------------
Models are evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² (coefficient of determination)

Comparisons are performed across:
- Different model types
- Operational modes (ballast vs laden)
- Time periods

Results
-------
- Ensemble models (XGBoost) provide the most consistent and accurate predictions of fuel consumption across operating conditions.
- Fuel consumption is strongly driven by engine power and vessel speed, with secondary influence from environmental conditions.
- Ballast and laden conditions exhibit distinct performance profiles, validating the need for separate modeling or feature inclusion.
- Neural networks capture complex interactions but require more tuning and offer less interpretability.

Key findings include:
- Increased slip is associated with reduced propulsion efficiency and higher fuel consumption.
- Environmental resistance (wind and currents) introduces measurable deviations from baseline fuel curves.
- Draft and cargo load significantly influence resistance and engine demand.

Conclusion
----------
The project demonstrates the effectiveness of machine learning in modeling vessel fuel consumption and operational efficiency using real-world sensor data.

The results highlight:
- The importance of feature engineering and operational segmentation
- The superiority of ensemble methods for structured maritime data
- The potential for applying such models in performance monitoring and decision support systems

Technology Stack
----------------
- Python
- NumPy, Pandas
- Scikit-learn
- XGBoost
- TensorFlow / Keras or PyTorch
- Matplotlib, Seaborn
- Jupyter Notebook

Execution Instructions
----------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run the notebook:
   jupyter notebook Master_Thesis_AI_ML.ipynb

Project Structure
-----------------
- ΔΙΠΛΩΜΑΤΙΚΗ_AI_ML.ipynb   Main analysis and modeling notebook
- NISSOS SIFNOS analysis.xlsx   Source dataset
- README.txt                Documentation

