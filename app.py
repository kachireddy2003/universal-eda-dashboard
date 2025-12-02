import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Universal EDA Dashboard",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Universal EDA Dashboard")
st.caption("Upload any CSV / Excel file and explore it like a data analyst.")

st.sidebar.header("1️⃣ Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx", "xls"],
    help="Supported formats: CSV, Excel (.xlsx, .xls)",
)

use_example = st.sidebar.checkbox("Use example dataset (Iris-style)", value=not bool(uploaded_file))

@st.cache_data
def load_example():
    data = {
        "sepal_length": [5.1, 4.9, 6.2, 5.9, 5.5, 6.7, 5.6, 6.3],
        "sepal_width":  [3.5, 3.0, 3.4, 3.0, 2.4, 3.1, 2.5, 2.9],
        "petal_length": [1.4, 1.4, 5.4, 4.2, 3.7, 5.6, 3.9, 5.1],
        "petal_width":  [0.2, 0.2, 2.3, 1.5, 1.0, 2.4, 1.1, 1.8],
        "species":      ["setosa", "setosa", "virginica", "versicolor",
                         "versicolor", "virginica", "versicolor", "virginica"],
    }
    return pd.DataFrame(data)

@st.cache_data
def load_file(file) -> pd.DataFrame:
    name = file.name.lower()

    # Excel files (safe)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)

    # CSV files — auto-detect encoding
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(file, encoding="latin1")
        except Exception:
            return pd.read_csv(file, encoding_errors="ignore")

    # Fallback for unknown types
    try:
        return pd.read_excel(file)
    except:
        return pd.read_csv(file, encoding_errors="ignore")


df = None
if uploaded_file is not None:
    try:
        df = load_file(uploaded_file)
        st.success(f"✅ Loaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
elif use_example:
    df = load_example()
    st.info("Using built-in Iris-style example dataset. Upload your own to replace it.")

if df is None:
    st.warning("Please upload a dataset or enable the example dataset from the sidebar.")
    st.stop()

# Try to parse datetime columns
for col in df.columns:
    if df[col].dtype == object:
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            df[col] = parsed
        except Exception:
            pass

st.markdown("### 🧾 Dataset Overview")
info_cols1, info_cols2, info_cols3 = st.columns(3)
with info_cols1:
    st.metric("Rows", df.shape[0])
with info_cols2:
    st.metric("Columns", df.shape[1])
with info_cols3:
    num_cols = df.select_dtypes(include="number").shape[1]
    st.metric("Numeric Columns", num_cols)

tab_preview, tab_summary, tab_filter, tab_viz, tab_corr, tab_missing, tab_download = st.tabs(
    ["👀 Preview", "📉 Summary", "🎛 Filter Data", "📈 Visualizations",
     "🔥 Correlation", "❗ Missing Values", "⬇️ Download"]
)

with tab_preview:
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("Column types:")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

with tab_summary:
    st.subheader("Numeric Summary")
    if num_cols > 0:
        st.dataframe(df.describe().T, use_container_width=True)
    else:
        st.info("No numeric columns to summarize.")

    st.subheader("Categorical Summary (Top 10 per column)")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        for col in cat_cols:
            st.markdown(f"**{col}**")
            vc = df[col].value_counts().head(10)
            st.dataframe(vc, use_container_width=True)
    else:
        st.info("No categorical columns detected.")

with tab_filter:
    st.subheader("Create a Filtered View")

    filter_cols = st.multiselect(
        "Select columns to apply filters on",
        options=df.columns.tolist(),
    )

    filtered_df = df.copy()
    for col in filter_cols:
        col_data = filtered_df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            selected_range = st.slider(
                f"{col} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            filtered_df = filtered_df[
                (filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])
            ]
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            min_date = col_data.min()
            max_date = col_data.max()
            date_range = st.date_input(
                f"{col} range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df[col] >= pd.to_datetime(start_date)) &
                    (filtered_df[col] <= pd.to_datetime(end_date))
                ]
        else:
            unique_vals = sorted(col_data.dropna().unique().tolist())
            selected_vals = st.multiselect(
                f"Values to keep in {col}",
                options=unique_vals,
                default=unique_vals,
            )
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    st.markdown(f"**Filtered rows:** {len(filtered_df)}")
    st.dataframe(filtered_df, use_container_width=True, height=350)

with tab_viz:
    st.subheader("Visualize Your Data")

    viz_type = st.radio(
        "Choose visualization type",
        ["Univariate", "Bivariate"],
        horizontal=True,
    )

    if viz_type == "Univariate":
        col = st.selectbox("Select a column", options=df.columns.tolist())
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                fig_box = px.box(df, y=col, title=f"Boxplot of {col}")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            vc = series.value_counts().reset_index()
            vc.columns = [col, "count"]
            fig_bar = px.bar(vc, x=col, y="count", title=f"Count of {col}", text_auto=True)
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        col_x = st.selectbox("X-axis", options=df.columns.tolist())
        col_y = st.selectbox("Y-axis", options=df.columns.tolist(), index=min(1, len(df.columns)-1))
        chart_kind = st.selectbox(
            "Chart type",
            ["Scatter", "Line", "Bar", "Box"],
        )

        if chart_kind == "Scatter":
            fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}")
        elif chart_kind == "Line":
            fig = px.line(df, x=col_x, y=col_y, title=f"{col_y} over {col_x}")
        elif chart_kind == "Bar":
            fig = px.bar(df, x=col_x, y=col_y, title=f"{col_y} by {col_x}")
        else:
            fig = px.box(df, x=col_x, y=col_y, title=f"{col_y} distribution by {col_x}")
        st.plotly_chart(fig, use_container_width=True)

with tab_corr:
    st.subheader("Correlation Heatmap (Numeric Columns)")
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.dataframe(corr, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for correlation.")

with tab_missing:
    st.subheader("Missing Values Overview")
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values,
    })
    st.dataframe(missing_df, use_container_width=True)

    non_zero = missing_df[missing_df["missing_count"] > 0]
    if not non_zero.empty:
        fig_missing = px.bar(
            non_zero,
            x="column",
            y="missing_count",
            title="Missing Values per Column",
            text_auto=True,
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values detected! 🎉")

with tab_download:
    st.subheader("Download Processed Data")

    csv_all = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download full dataset as CSV",
        data=csv_all,
        file_name="dataset_full.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Download Filtered Data (quick filter here)")

    col_to_filter = st.selectbox("Select a column to filter (optional)", options=[None] + df.columns.tolist())

    if col_to_filter:
        col_series = df[col_to_filter]
        if pd.api.types.is_numeric_dtype(col_series):
            min_val = float(col_series.min())
            max_val = float(col_series.max())
            selected_range = st.slider(
                f"{col_to_filter} range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            df_dl = df[(col_series >= selected_range[0]) & (col_series <= selected_range[1])]
        else:
            unique_vals = sorted(col_series.dropna().unique().tolist())
            selected_vals = st.multiselect(
                f"Values to keep in {col_to_filter}",
                options=unique_vals,
                default=unique_vals,
            )
            df_dl = df[df[col_to_filter].isin(selected_vals)]
    else:
        df_dl = df

    st.write(f"Filtered rows for download: {len(df_dl)}")
    csv_filtered = df_dl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered dataset as CSV",
        data=csv_filtered,
        file_name="dataset_filtered.csv",
        mime="text/csv",
    )
