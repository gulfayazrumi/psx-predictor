# Accuracy Tracking Dashboard Implementation Plan

## Goal
Implement a robust Accuracy Tracking Dashboard to monitor the performance of the AI prediction models. This dashboard will display real-time accuracy metrics, historical performance, and detailed pass/fail analysis for each stock.

## User Review Required
> [!IMPORTANT]
> This module relies on the availability of historical prediction logs (`reports/history/predictions_*.csv`) and up-to-date historical market data (`data/raw/historical/*.csv`).

## Proposed Changes

### Analysis Layer
#### [NEW] [accuracy_tracker.py](file:///e:/Ai classes/projects/Claude PSX Project/psx-predictor-complete/psx-predictor/src/analysis/accuracy_tracker.py)
- **Functionality**:
    - Iterate through all historical prediction files.
    - Compare predicted prices/directions with actual market outcomes (fetched from historical CSVs).
    - Calculate key metrics:
        - **Directional Accuracy**: % of times the model correctly predicted UP/DOWN.
        - **Price Error**: Mean Absolute Percentage Error (MAPE) between predicted and actual price.
        - **Confidence Accuracy**: Accuracy broken down by confidence buckets (High, Medium, Low).
    - Generate a JSON report `reports/accuracy_report.json`.

### Data Layer
#### [MODIFY] [data_updater.py](file:///e:/Ai classes/projects/Claude PSX Project/psx-predictor-complete/psx-predictor/src/data_collection/data_updater.py)
- **Functionality**:
    - Ensure historical data is synchronized daily to allow for accurate verification of past predictions.
    - Fix duplicate column issues and ensure robust API fetching.

### Dashboard Layer
#### [MODIFY] [dashboard/app.py](file:///e:/Ai classes/projects/Claude PSX Project/psx-predictor-complete/psx-predictor/dashboard/app.py)
- **Navigation**: Add "🎯 Model Accuracy" to the sidebar.
- **New Page**: Implement the Model Accuracy page.
    - **Summary Metrics**: Big number cards for Overall Accuracy, Avg Price Error, Total Predictions.
    - **Accuracy Trend**: Line chart showing accuracy over time.
    - **Confidence Analysis**: Bar chart showing accuracy vs. confidence level.
    - **Detailed Log**: Searchable table of all past predictions with Pass/Fail status.

## Verification Plan
### Automated Tests
- Run `src/analysis/accuracy_tracker.py` and verify `reports/accuracy_report.json` is generated.
- Run `debug_updater.py` to ensure data synchronization is working.

### Manual Verification
- Launch dashboard (`streamlit run dashboard/app.py`).
- Navigate to "🎯 Model Accuracy".
- Verify that the metrics match the generated report.
- Check if the "Detailed Log" shows correct Pass/Fail status for known outcomes.
