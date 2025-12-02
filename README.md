# Universal EDA Dashboard

A Streamlit web app for exploratory data analysis (EDA) on any CSV / Excel dataset.

## Features
- Upload CSV / XLS / XLSX
- Preview data and dtypes
- Numeric summary (describe)
- Categorical value counts
- Flexible filters (numeric, date, categorical)
- Univariate & bivariate charts (hist, box, bar, line, scatter)
- Correlation heatmap for numeric columns
- Missing values report
- Download full or filtered dataset as CSV

## How to run

```bash
cd universal-eda-dashboard-full
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
