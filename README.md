# Climate Change Impact Analysis on Rainfall and Temperature

This project analyzes the relationship between temperature and rainfall to study climate change's effects on agricultural sectors. It focuses on historical climate data to predict rainfall trends based on temperature using a linear regression model.

## Project Overview

The project:
- Processes historical climate data from a CSV file.
- Analyzes annual temperature and rainfall trends.
- Builds a predictive model using linear regression to predict rainfall based on temperature.
- Visualizes actual versus predicted rainfall values.

## Dataset

The dataset, `POWER_Point_Monthly_19810101_20221231_026d20N_092d94E_UTC.csv`, contains monthly climate data from 1981 to 2022 for a specific geographical location. The primary parameters of interest in this analysis are:
- **PRECTOTCORR_SUM**: Total corrected precipitation (rainfall).
- **TS**: Surface temperature.

## Requirements

To run this project, install the necessary libraries:
```bash
pip install pandas numpy matplotlib scikit-learn
