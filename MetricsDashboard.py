import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Plotly imports
import plotly.express as px
from streamlit_plotly_events import plotly_events

def main():
    st.title("Query Metrics Dashboard – Interactive PCA (2D & 3D) with K-Means Clustering")

    st.sidebar.header("Upload your metrics file")
    uploaded_file = st.sidebar.file_uploader("Upload a JSON or JSONL metrics file", type=["json", "jsonl"])
    
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
                return
        
        # Convert to DataFrame
        df = pd.DataFrame(data_rows)
        st.write("Data successfully loaded. Shape:", df.shape)
        
        # ------------------ Numeric Columns ------------------
        # Add newly introduced numeric columns: AnalysisTimeMs, RowCount, ColumnCount
        numeric_columns = [
            "AnalysisTimeMs", "RowCount", "ColumnCount",  # newly added
            "BuffersReceived", "BuffersSent", "BytesReceived", "BytesSent",
            "ServerRoundtrips", "ConnectionTime", "ExecutionTime",
            "NetworkServerTime"
        ]
        
        # Keep only columns that actually exist in the uploaded data
        numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        # Convert numeric columns to numeric (coerce errors to NaN) and fill them with 0
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # ------------------ Descriptive Stats ------------------
        st.subheader("Descriptive Statistics")
        if numeric_columns:
            st.dataframe(df[numeric_columns].describe())
        else:
            st.info("No numeric columns found in the dataset.")
            return
        
        # ------------------ Box Plot ------------------
        st.subheader("Box Plots of Metrics")
        if numeric_columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df[numeric_columns], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns to plot.")
        
        # ------------------ K-Means & PCA ------------------
        # We need at least 2 numeric columns for 2D PCA
        if len(numeric_columns) >= 2:
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_columns])
            
            # K-Means clustering
            n_clusters = st.sidebar.slider("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            
            # Add labels to df
            df["cluster_label"] = labels.astype(str)
            
            # Store row index for reference on clicks
            df["row_index"] = df.index

            # ------------------ 2D PCA ------------------
            st.subheader("2D PCA Visualization (Interactive)")
            pca_2d = PCA(n_components=2)
            pca_2d_result = pca_2d.fit_transform(scaled_data)
            df["pca_2d_1"] = pca_2d_result[:, 0]
            df["pca_2d_2"] = pca_2d_result[:, 1]
            st.write("Explained variance ratio (2D):", pca_2d.explained_variance_ratio_)

            fig_2d = px.scatter(
                df,
                x="pca_2d_1",
                y="pca_2d_2",
                color="cluster_label",
                color_discrete_sequence=px.colors.qualitative.Set1,
                hover_data=["row_index", "Query"] if "Query" in df.columns else ["row_index"],
                title="PCA (2D) – K-Means Clusters"
            )
            # Attach row_index for clicking
            fig_2d.update_traces(customdata=df["row_index"])
            
            selected_points_2d = plotly_events(
                fig_2d,
                click_event=True,
                hover_event=False,
                select_event=False,
                override_height=600,
                override_width="100%"
            )
            
            # Show details if a point is clicked in 2D
            if selected_points_2d:
                clicked_row_idx = selected_points_2d[0]["customdata"]
                st.write(f"**Clicked row index (2D)**: {clicked_row_idx}")
                if "Query" in df.columns:
                    st.write(f"**Query**: {df.loc[clicked_row_idx, 'Query']}")
                else:
                    st.write("No 'Query' column in data.")
            
            # ------------------ 3D PCA ------------------
            if len(numeric_columns) >= 3:
                st.subheader("3D PCA Visualization (Interactive)")
                pca_3d = PCA(n_components=3)
                pca_3d_result = pca_3d.fit_transform(scaled_data)
                df["pca_3d_1"] = pca_3d_result[:, 0]
                df["pca_3d_2"] = pca_3d_result[:, 1]
                df["pca_3d_3"] = pca_3d_result[:, 2]
                
                st.write("Explained variance ratio (3D):", pca_3d.explained_variance_ratio_)

                # Build customdata for better hover/click usage
                if "Query" in df.columns:
                    df["_hover_col"] = df["Query"]  # store Query temporarily
                    customdata = df[["row_index", "Tag", "cluster_label", "_hover_col"]].values
                    hovertemplate = (
                        "<b>Cluster:</b> %{customdata[2]}<br>"
                        "<b>PCA1:</b> %{x:.2f}<br>"
                        "<b>PCA2:</b> %{y:.2f}<br>"
                        "<b>PCA3:</b> %{z:.2f}<br>"
                        "<b>Row Index:</b> %{customdata[0]}<br>"
                        "<b>Tag:</b> %{customdata[1]}<br>"
                        "<extra></extra>"
                    )
                else:
                    customdata = df[["row_index", "Tag", "cluster_label"]].values
                    hovertemplate = (
                        "<b>Cluster:</b> %{customdata[2]}<br>"
                        "<b>PCA1:</b> %{x:.2f}<br>"
                        "<b>PCA2:</b> %{y:.2f}<br>"
                        "<b>PCA3:</b> %{z:.2f}<br>"
                        "<b>Row Index:</b> %{customdata[0]}<br>"
                        "<b>Tag:</b> %{customdata[1]}<br>"
                        "<extra></extra>"
                    )

                fig_3d = px.scatter_3d(
                    df,
                    x="pca_3d_1",
                    y="pca_3d_2",
                    z="pca_3d_3",
                    color="cluster_label",
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    hover_data=["row_index", "Tag", "cluster_label"],  # minimal default hover
                    title="PCA (3D) – K-Means Clusters"
                )
                
                fig_3d.update_traces(
                    customdata=customdata,
                    hovertemplate=hovertemplate
                )

                selected_points_3d = plotly_events(
                    fig_3d,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=600,
                    override_width="100%"
                )
                
                # Show details if a point is clicked in 3D
                if selected_points_3d:
                    point_data = selected_points_3d[0]["customdata"]
                    clicked_row_idx_3d = point_data[0]  # row_index
                    st.write(f"**Clicked row index (3D)**: {clicked_row_idx_3d}")
                    if "Query" in df.columns:
                        st.write(f"**Query**: {df.loc[clicked_row_idx_3d, 'Query']}")
                    else:
                        st.write("No 'Query' column in data.")
            else:
                st.warning("Need at least 3 numeric columns for 3D PCA visualization.")
        else:
            st.warning("Need at least 2 numeric columns for PCA and K-Means clustering.")
        
        # ------------------ Threshold Filtering ------------------
        st.subheader("Filter Large/Slow Queries")
        # Let user pick which columns to apply thresholds on
        threshold_cols = st.multiselect("Columns to apply thresholds on", numeric_columns, default=["BytesSent", "ConnectionTime"])
        
        threshold_values = {}
        for col in threshold_cols:
            max_val = float(df[col].max()) if len(df[col]) else 1.0
            default_val = float(np.percentile(df[col], 90))  # example default: 90th percentile
            threshold_values[col] = st.number_input(
                f"Threshold for {col}",
                min_value=0.0,
                max_value=max_val * 10,
                value=default_val,
                step=1.0
            )
        
        if threshold_cols:
            mask = pd.Series(False, index=df.index)
            for col in threshold_cols:
                mask |= (df[col] > threshold_values[col])
            
            large_queries = df[mask]
            st.write(f"Found {len(large_queries)} queries exceeding thresholds.")
            if not large_queries.empty:
                show_cols = ["Timestamp", "Tag", "Query"] + threshold_cols
                show_cols = [c for c in show_cols if c in large_queries.columns]
                st.dataframe(large_queries[show_cols].sort_values(by=threshold_cols, ascending=False))
        else:
            st.info("No threshold columns selected.")
    
    else:
        st.info("Please upload a metrics file to proceed.")

if __name__ == "__main__":
    main()
