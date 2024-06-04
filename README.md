# Analyzing Bitcoin Transactions with Pandas and Spark

## Project Description
This project aims to analyze Bitcoin transactions using both Pandas and Spark. The goal is to perform comprehensive data analysis and compare the performance of Pandas and Spark in processing large datasets. Additionally, the project explores the creation of classification models, clustering techniques for Bitcoin addresses, and network analysis using PageRank.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Contributors](#contributors)

## Installation
To run this project, you need the following libraries:
- Pandas
- Scikit-learn
- PySpark
- Matplotlib
- Seaborn
- cuDF
- cuGraph

You can install the required libraries using:
```bash
pip install pandas scikit-learn pyspark matplotlib seaborn cudf cugraph
```

## Usage
Clone the repository:
```bash
git clone https://github.com/yourusername/bitcoin-transactions-analysis.git
cd bitcoin-transactions-analysis
```
Run the Jupyter notebook:
```bash
jupyter notebook Groupwork_ada_5.ipynb
```

## Methodology

### Part 0: Comparing Spark and Pandas Performance
This section computes the number of transactions each address has been involved in, using both Spark and Pandas. The performance of both methods is measured by the time taken for the computation. Multiple runs are conducted, and the final values are recorded for comparison.

**Discussion:**
- **PySpark:** Ideal for large databases due to distributed computing and in-memory processing, enabling fast, parallel processing of big data across multiple machines. Spark is faster than Pandas for larger datasets.
- **Pandas:** Better for smaller datasets due to efficient in-memory data structures and simpler syntax. Spark's parallel processing incurs overhead when dealing with smaller datasets.

### Part 1: Basic Statistics Computation
This section computes basic statistics on the dataset. It recommends using larger datasets if strange results are encountered with smaller ones during development.

### Part 2: Index Creation for Web Queries
Assumes the development of a web application that provides statistics for any Bitcoin address. At least three statistics are maintained:
1. Address account balance (mandatory)
2. Top-3 commercial partners
3. Average transaction value

Indices are created to support these queries, with each index mapping an address to the computed statistics. Sample code is provided to build and print five lines of each index.

**Discussions:**
1. **Balance:**
   - Calculates the net balance for each address by summing transaction values in the 'Input' and 'Output' columns and finding the difference.
   - Results sorted to identify addresses with the highest net balance.
2. **Address Age:**
   - Determines the age of each address by calculating the time between the first and latest transaction.
3. **Average Transaction Values:**
   - Computes the average transaction value for each address and further breaks it down by day, week, month, and quarter.
4. **Number of Inputs/Outputs:**
   - Counts the occurrences of each address in the 'Input' and 'Output' columns and combines these counts for analysis.

### Part 3: Classification Models
Develops classification models to assign labels to addresses using various features derived from transaction data:
- **Feature Engineering:** Generates features like the number of transactions, average transaction value, and average number of partners per day.
- **Model Selection:** Implements k-Nearest Neighbors (k-NN) and Random Forest classifiers.
- **Model Evaluation:** Uses accuracy, precision, recall, and F1 score for evaluation.

**Discussion:**
- The initial k-NN model showed moderate performance, prompting a switch to Random Forest, which significantly improved accuracy and reliability.
- The Random Forest model achieved 72.31% accuracy, with a precision of 71.27%, recall of 72.31%, and F1 score of 71.51%.

### Part 4: Clustering Addresses
Proposes clustering Bitcoin addresses in the absence of labels using KMeans:
- **Elbow Method:** Determines the optimal number of clusters.
- **Outlier Detection and Removal:** Improves clustering results by removing outliers.
- **PCA Visualization:** Uses Principal Component Analysis to visualize clusters in reduced dimensions.

**Discussion:**
- Initial clustering suggested clear separations influenced by outliers.
- Post-outlier removal, the silhouette score decreased but still indicated strong clustering, affirming the method's effectiveness.

### Part 5: PageRank Analysis Using cuDF and cuGraph
Uses cuDF and cuGraph to analyze Bitcoin address influence via PageRank:
- **Graph Construction:** Creates a directed graph with nodes representing addresses and edges representing transactions weighted by value.
- **PageRank Calculation:** Quantifies the influence of each address and identifies the top ten most influential addresses.

**Discussion:**
- PageRank scores varied significantly, indicating a typical real-world network distribution with a few highly influential nodes.
- High-scoring nodes likely represent large exchange wallets, mining pools, or other significant entities.

## Results
- **Top-10 Largest Transactions:** Identified and displayed.
- **Evolution of Transactions Over Time:** Plotted the trend of Bitcoin transactions.
- **Currency Transfer Evolution:** Plotted the evolution of currency transfer in BTC and USD.
- **Classification Model Performance:** Random Forest outperformed k-NN, demonstrating the effectiveness of feature engineering and model selection.
- **Clustering Results:** KMeans clustering revealed meaningful groupings, validated by a strong silhouette score.
- **PageRank Analysis:** Identified influential addresses, suggesting centralization points in the Bitcoin network.

## Conclusion
This project demonstrates the use of data analysis and machine learning tools to derive insights from Bitcoin transactions. It highlights the strengths and weaknesses of Pandas and Spark in processing large datasets and shows the potential of classification, clustering, and network analysis techniques in understanding cryptocurrency transaction behaviors.
