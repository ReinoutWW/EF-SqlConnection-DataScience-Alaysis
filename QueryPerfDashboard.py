import streamlit as st
import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.stats import zscore

import plotly.express as px
import plotly.graph_objects as go
import altair as alt

#########################
# 1. App Configuration  #
#########################
st.set_page_config(page_title="Query Performance Dashboard", layout="wide")

st.title("Query Performance Dashboard")

#############################
# 2. Data Loading Function  #
#############################
uploaded_file = st.sidebar.file_uploader("Upload a JSON or JSONL metrics file", type=["json", "jsonl"])
df = None

if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    data_rows = []
    lines = file_content.strip().split("\n")

    # Attempt to parse the file as JSONL or JSON
    try:
        # First try line-by-line (JSONL)
        for line in lines:
            if line.strip():
                data_rows.append(json.loads(line))
    except json.JSONDecodeError:
        # If that fails, try a single JSON or JSON array
        try:
            data_obj = json.loads(file_content)
            if isinstance(data_obj, list):
                data_rows = data_obj
            else:
                data_rows = [data_obj]
        except json.JSONDecodeError:
            st.error("File is not valid JSON/JSONL.")
            st.stop()
    
    # Convert to DataFrame
    df = pd.DataFrame(data_rows)
    st.write("Data successfully loaded. Shape:", df.shape)

###################################
# 3. Only show if data is present #
###################################
if df is not None and not df.empty:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    # -------------------------------
    # 3a. Sidebar Controls
    # -------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset.")
        st.stop()

    # Choose columns for PCA or Clustering
    with st.sidebar:
        st.markdown("## Data & PCA/Clustering Settings")
        selected_pca_cols = st.multiselect(
            "Select numeric columns for PCA/Clustering",
            numeric_cols,
            default=numeric_cols[:2]  # pick first two by default if available
        )
        
        # Clustering algorithm choice
        cluster_algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ["KMeans", "DBSCAN", "AgglomerativeClustering"]
        )
        
        # KMeans/Agglomerative parameters
        n_clusters = st.slider("Number of Clusters (KMeans/Agglomerative)", 2, 10, 3)
        # DBSCAN parameters
        eps_value = st.slider("DBSCAN eps value", 0.1, 10.0, 0.5, 0.1)
        min_samples_value = st.slider("DBSCAN min_samples", 1, 20, 5)

        # Filter data if needed
        st.markdown("### Filter Data")
        filter_col = st.selectbox("Select a column to filter by (optional)", [None] + numeric_cols)
        if filter_col:
            filter_min, filter_max = float(df[filter_col].min()), float(df[filter_col].max())
            chosen_range = st.slider(f"Select range for {filter_col}", 
                                     min_value=filter_min, 
                                     max_value=filter_max, 
                                     value=(filter_min, filter_max))
        else:
            chosen_range = None

    # Possibly filter the data
    df_filtered = df.copy()
    if filter_col and chosen_range:
        df_filtered = df_filtered[
            (df_filtered[filter_col] >= chosen_range[0]) & 
            (df_filtered[filter_col] <= chosen_range[1])
        ]

    if len(selected_pca_cols) < 1:
        st.warning("Please select at least one numeric column for PCA and clustering.")
        st.stop()

    # Scale the data
    X = df_filtered[selected_pca_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    #################
    # 4. PCA Computation
    #################
    num_features = len(selected_pca_cols)

    # --- 2D PCA ---
    # Use min(2, num_features) components for 2D
    pca_2d_components = min(2, num_features)
    pca_2d = PCA(n_components=pca_2d_components)
    coords_2d = pca_2d.fit_transform(X_scaled)

    pca_df = df_filtered.copy()
    pca_df["PCA_1_2D"] = coords_2d[:, 0]
    if pca_2d_components == 2:
        pca_df["PCA_2_2D"] = coords_2d[:, 1]
    else:
        # If there's only 1 feature, the second dimension does not exist
        pca_df["PCA_2_2D"] = 0  # or keep it at 0 to avoid errors

    # --- 3D PCA (only if we have at least 3 features) ---
    can_do_3d = num_features >= 3
    if can_do_3d:
        pca_3d = PCA(n_components=3)
        coords_3d = pca_3d.fit_transform(X_scaled)
        pca_df["PCA_1_3D"] = coords_3d[:, 0]
        pca_df["PCA_2_3D"] = coords_3d[:, 1]
        pca_df["PCA_3_3D"] = coords_3d[:, 2]
    else:
        # Fallback columns for 3D just to avoid reference issues
        pca_df["PCA_1_3D"] = pca_df["PCA_1_2D"]
        pca_df["PCA_2_3D"] = pca_df["PCA_2_2D"]
        pca_df["PCA_3_3D"] = 0

    #################
    # 5. Clustering
    #################
    if cluster_algorithm == "KMeans":
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = cluster_model.fit_predict(X_scaled)
    elif cluster_algorithm == "DBSCAN":
        cluster_model = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        clusters = cluster_model.fit_predict(X_scaled)
    else:  # AgglomerativeClustering
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = cluster_model.fit_predict(X_scaled)

    pca_df["Cluster"] = clusters

    #########################
    # 6. PCA 2D Scatter Plot
    #########################
    st.subheader("PCA 2D Scatter Plot")
    tag_column = "Tag"  # Make sure this matches the column in your dataset

    # If the second PCA dimension is artificially 0, the chart is less interesting, but won't break
    hover_cols = [tag_column] if tag_column in pca_df.columns else None
    fig_2d = px.scatter(
        pca_df,
        x="PCA_1_2D",
        y="PCA_2_2D",
        color="Cluster",
        hover_data=hover_cols,
        title="PCA 2D Scatter Plot"
    )
    st.plotly_chart(fig_2d, use_container_width=True)

    #########################
    # 7. PCA 3D Scatter Plot
    #########################
    if can_do_3d:
        st.subheader("PCA 3D Scatter Plot")
        fig_3d = px.scatter_3d(
            pca_df,
            x="PCA_1_3D",
            y="PCA_2_3D",
            z="PCA_3_3D",
            color="Cluster",
            hover_data=hover_cols,
            title="PCA 3D Scatter Plot"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Not enough features selected for 3D PCA. Displaying 2D PCA only.")

    ########################################
    # 8. Interactive Box Plot (Altair)
    ########################################
    st.subheader("Interactive Box Plot by Tag")
    # Choose a metric column for boxplot
    metric_col = st.selectbox("Choose a metric for box plot", numeric_cols)

    if tag_column in df_filtered.columns:
        box_chart = alt.Chart(df_filtered).mark_boxplot().encode(
            x=alt.X(tag_column, sort=None),
            y=metric_col,
            tooltip=[tag_column, metric_col]
        ).interactive()

        st.altair_chart(box_chart, use_container_width=True)
    else:
        st.warning(f"No '{tag_column}' column found to display box plot by Tag.")

    ######################################################
    # 9. Additional Graphs (At least 3 more visuals)
    ######################################################

    st.subheader("Additional Query Performance Visuals")

    # 9a. Distribution/Histogram of a numeric metric
    st.markdown("#### 1) Histogram of Selected Metric")
    fig_hist = px.histogram(df_filtered, x=metric_col, nbins=30, title=f"Histogram of {metric_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # 9b. Scatter Plot of Two Numeric Columns
    if len(numeric_cols) > 1:
        st.markdown("#### 2) Scatter Plot of Two Numeric Columns")
        scatter_x = st.selectbox("Select X-axis for scatter plot", numeric_cols, key="scatter_x")
        scatter_y = st.selectbox("Select Y-axis for scatter plot", numeric_cols, key="scatter_y")
        
        fig_scatter = px.scatter(
            df_filtered, 
            x=scatter_x, 
            y=scatter_y,
            hover_data=[tag_column] if tag_column in df_filtered.columns else None,
            title=f"Scatter Plot: {scatter_x} vs {scatter_y}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 9c. Bar Plot of Average Metric by Tag
    st.markdown("#### 3) Bar Plot of Average Metric by Tag")
    if tag_column in df_filtered.columns:
        df_bar = df_filtered.groupby(tag_column)[metric_col].mean().reset_index()
        df_bar.columns = [tag_column, f"avg_{metric_col}"]

        fig_bar = px.bar(
            df_bar,
            x=tag_column,
            y=f"avg_{metric_col}",
            title=f"Average {metric_col} by {tag_column}",
            hover_data=[f"avg_{metric_col}"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning(f"No '{tag_column}' column found to display bar chart by Tag.")

    ########################################
    # 10. Outlier Detection
    ########################################
    st.subheader("Outlier Detection for Top 10 Problematic Queries")

    # We'll base 'problematic' on the z-score of the selected metric
    if metric_col in df_filtered.columns:
        df_outliers = df_filtered.copy()
        df_outliers["zscore"] = zscore(df_outliers[metric_col].astype(float))
        df_outliers.sort_values("zscore", ascending=False, inplace=True)
        top_10 = df_outliers.head(10)

        st.write("Here are the top 10 outliers (highest z-scores):")
        if tag_column in df_outliers.columns:
            st.write(top_10[[tag_column, metric_col, "zscore"]])
        else:
            st.write(top_10[[metric_col, "zscore"]])
    else:
        st.warning(f"Cannot perform outlier detection on {metric_col}.")
else:
    st.info("Please upload a valid JSON/JSONL file to proceed.")
