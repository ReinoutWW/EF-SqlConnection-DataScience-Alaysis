
# Query Performance Dashboard
A Streamlit-based interactive application designed to help data scientists, engineers, and analysts visualize and understand query performance metrics. This dashboard provides clustering insights, PCA visualizations, outlier detection, and several other interactive plots to help identify problematic or inefficient queries.

## Dashboard

![image](https://github.com/user-attachments/assets/830e49ae-9064-4bd3-81fa-aa7877c9ab2b)
*Query Data Analysis Dashboard, scenario 1*

---

![image](https://github.com/user-attachments/assets/5e5d4cc6-6a77-492b-9ae1-e0daf2d64d43)
*Query Data Analysis Dashboard, scenario 2 *

---

![image](https://github.com/user-attachments/assets/b4b37086-a9a6-402b-b3e6-602e66a899f3)
*Query Data Analysis Dashboard, scenario 3 *

## Features

1. **Interactive Data Loading**
    - Supports JSON or JSONL files for uploading metrics.
    - Automatically detects JSON vs. JSONL to parse.
    - Displays initial dataset shape and previews data.

2. **PCA Visualizations**

    - 2D PCA Scatter Plot and (when applicable) 3D PCA Scatter Plot.
    - Dynamically adjusts the PCA components based on selected numeric columns.
    - Hover over points to see Tag info (which query is being displayed).

3. **Clustering Algorithms**

    - KMeans, DBSCAN, or AgglomerativeClustering to cluster your queries.
    - Choose the desired number of clusters or tweak DBSCAN parameters.
    - Helps quickly group queries with similar performance characteristics.

4. **Outlier Detection**

    - Quickly identify top 10 “problematic” queries based on a z-score of a selected metric.
    - Outliers are queries significantly higher (z-score) than the average.

5. **Interactive Box Plot**

    - Box plot grouped by Tag, showing outliers and distribution of a selected numeric metric.
    - Hover tooltips highlight individual Tag details.

6. **Additional Graphs**

    - Histogram of a selected metric.
    - Scatter Plot of two numeric columns to visualize relationships.
    - Bar Plot of average metric grouped by Tag.

7. **Dynamic Filtering**
    - Filter your data on any numeric column range to focus on specific segments.

# Why Clustering for Query Performance?

Clustering algorithms help group queries with similar performance characteristics. Here’s how that can be insightful:

- KMeans: Partitions queries into k clusters to identify which group of queries share similar latencies, resource usage, or execution times.
- DBSCAN: Finds densely grouped queries and can mark outliers that behave distinctly from the majority. Useful for discovering truly anomalous queries.
- AgglomerativeClustering: Hierarchical approach that helps to see how queries might cluster in varying degrees of similarity, which can reveal nested groups of performance profiles.

By identifying clusters, you can see which queries require similar tuning or share similar patterns of inefficiency or high execution times. This enables targeted optimizations, such as indexing strategies, rewriting queries, or caching frequently accessed data.

# Getting Started

1. Clone this repository:
   ```
   git clone https://github.com/ReinoutWW/EF-SqlConnection-DataScience-Alaysis.git
   cd EF-SqlConnection-DataScience-Alaysis
   ```
2. Install the dependencies (preferably in a virtual environment):
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run QueryPerfDashboard.py
   ```
4. Open the application in your browser at the URL displayed in the terminal, typically `http://localhost:8501.`
5. Upload a `JSON` or `JSONL` metrics file. Select numeric columns for PCA and clustering, choose your clustering algorithm, and explore the various plots. There's a `example-dataset.jsonl` included in the repository to play around with dummy data.

# Example Usage

1. **Data Upload**: Drag and drop your `metrics.json` or `metrics.jsonl` file.
2. **Select Numeric Columns**: Pick columns like `latency`, `execution_time`, or any other performance-related metrics.
3. **Clustering**: Choose between `KMeans`, `DBSCAN`, or `AgglomerativeClustering` and tweak the parameters (like n_clusters, eps, etc.).
4. **Explore Visuals**:
    - PCA 2D/3D: Quickly see if certain clusters have distinct performance.
    - Box Plot: Pinpoint outliers or wide distributions for each query Tag.
    - Top 10 Outliers: Automatically detect the queries that deviate the most.
5. **Refine**: Adjust filters to zoom into certain query ranges (e.g., high latency only).

# Required dataset

| **Field**            | **Example Value**                                       | **Explanation**                                                                                                                               |
|----------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **Timestamp**        | `2025-02-16T18:15:36.919912`                            | The date and time at which the query metrics were recorded.                                                                                   |
| **Tag**              | `ExampleClass.LargeQuery_9`                             | A human-readable or programmatic label used to identify this particular query or code path.                                                   |
| **Query**            | `SELECT col1, col2, col3, ... FROM ExampleTable ...`    | The actual SQL query text that was executed.                                                                                                  |
| **AnalysisTimeMs**   | `1339.7654`                                             | Time (in milliseconds) taken for analysis before execution (e.g., parsing, planning).                                                         |
| **RowCount**         | `99596`                                                 | Number of rows returned or processed by the query.                                                                                            |
| **ColumnCount**      | `20`                                                    | Number of columns involved (selected or processed) in the query.                                                                              |
| **BuffersReceived**  | `7`                                                     | Count of buffer messages received from the server (specific to some database drivers).                                                        |
| **BuffersSent**      | `14`                                                    | Count of buffer messages sent to the server.                                                                                                  |
| **BytesReceived**    | `211434`                                                | Total bytes received from the server during query execution.                                                                                  |
| **BytesSent**        | `6899`                                                  | Total bytes sent to the server during query execution.                                                                                        |
| **CursorOpens**      | `0`                                                     | Number of cursor opens (if the database driver supports cursor-based queries).                                                                |
| **IduCount**         | `0`                                                     | Number of insert/delete/update (IDU) operations executed within this context.                                                                 |
| **IduRows**          | `0`                                                     | Number of rows affected by IDU operations.                                                                                                    |
| **PreparedExecs**    | `0`                                                     | Count of prepared query executions (if the driver/database supports prepared statements).                                                     |
| **Prepares**         | `0`                                                     | Number of times a query was prepared.                                                                                                         |
| **SelectCount**      | `0`                                                     | Number of SELECT statements executed in this context.                                                                                         |
| **SelectRows**       | `0`                                                     | Number of rows returned by SELECT statements (if tracked separately).                                                                        |
| **ServerRoundtrips** | `10`                                                    | Number of client-server round trips for the query execution.                                                                                  |
| **SumResultSets**    | `0`                                                     | Summation of all result sets returned (if queries can produce multiple result sets).                                                         |
| **Transactions**     | `0`                                                     | Number of transactions opened or committed in this context.                                                                                   |
| **UnpreparedExecs**  | `9`                                                     | Count of non-prepared (direct) query executions.                                                                                              |
| **ConnectionTime**   | `919`                                                   | Duration (in milliseconds) spent establishing the connection or reusing a pooled connection.                                                 |
| **ExecutionTime**    | `0`                                                     | Time (in milliseconds) for the query execution phase (excluding preparation, analysis, etc.).                                                |
| **NetworkServerTime**| `124`                                                   | Time (in milliseconds) spent on network and server-side overhead (as tracked by the client or driver).                                       |
*Supported data for the dashboard*

# Contributing

1. **Fork** the repo.
2. **Create a new branch for your feature or fix.**
3. **Submit a Pull Request** describing your changes, and link any relevant issues.

We welcome contributions in the form of bug reports, code improvements, new visualization features, or performance enhancements.
