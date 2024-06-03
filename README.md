# Analyzing Bitcoin Transactions with Pandas and Spark

## Project Description
This project aims to analyze Bitcoin transactions using both Pandas and Spark. The goal is to perform comprehensive data analysis and compare the performance of Pandas and Spark in processing large datasets. Additionally, the project explores the creation of classification models and clustering techniques for Bitcoin addresses.

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

You can install the required libraries using:
```bash
pip install pandas scikit-learn pyspark matplotlib seaborn
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bitcoin-transactions-analysis.git
   cd bitcoin-transactions-analysis
   ```
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook Groupwork_ada_5.ipynb
   ```

## Methodology
### Data Collection
The dataset comprises Bitcoin transactions stored in CSV files. Each transaction records information such as transaction ID, block height, input and output addresses, transaction sum, and time.

### Data Processing
Transactions are processed to extract meaningful insights:
- **Data Loading**: Data is loaded using Pandas and Spark.
- **Data Cleaning**: Data is cleaned and transformed to ensure it is ready for analysis.
- **Exploratory Data Analysis (EDA)**: EDA is performed to understand the data distribution and uncover initial patterns.
- **Performance Comparison**: The performance of Pandas and Spark is compared in processing the datasets.

### Analysis
- **Top-10 Largest Transactions**: Identified the largest transactions by value.
- **Evolution of Transactions**: Analyzed the trend of Bitcoin transactions over time.
- **Performance Comparison**: Compared the efficiency of Pandas and Spark in handling large datasets.

### Machine Learning Models
#### Exercise 3: Classification Models
We explored the creation of classification models to assign labels to Bitcoin addresses using statistics associated with each address. The following steps were involved:
1. **Feature Engineering**: Generated features such as the number of transactions, average transaction value, and the average number of partners per day.
2. **Model Selection**: Implemented different classification models including:
   - **k-Nearest Neighbors (k-NN)**
   - **Random Forest Classifier**
3. **Model Evaluation**: Evaluated the models using metrics like accuracy, precision, recall, and F1 score.

The relevance of features:
- **Number of Transactions**: Indicates the activity level of the address.
- **Average Transaction Value**: Helps in distinguishing between high-value and low-value addresses.
- **Average Number of Partners**: Provides insights into how frequently an address interacts with other addresses.

#### Exercise 4: Clustering Addresses
In the absence of labels, we proposed a clustering solution to group Bitcoin addresses based on their transaction behaviors:
1. **Feature Engineering**: Used similar statistics as in classification to create feature vectors for addresses.
2. **Clustering Algorithm**: Applied the **k-Means clustering** algorithm to group addresses into distinct clusters.
3. **Evaluation**: Assessed the clustering quality using metrics like silhouette score and examined the characteristics of the resulting clusters.

### Tools and Libraries Used
- **Pandas**: Used for data manipulation and analysis.
- **Scikit-learn**: Used for machine learning tasks.
- **Spark**: Used for big data processing and analysis.
- **Matplotlib & Seaborn**: Used for data visualization.

## Results
### Top-10 Largest Transactions
Displayed the top 10 largest transactions in terms of bitcoin currency.

### Evolution of Transactions Over Time
Displayed a graph showing the evolution of the number of transactions over time.

### Evolution of Currency Transferred Over Time
Displayed a graph showing the evolution of the amount of currency transferred (in BTC and USD) over time.

### Classification Model Performance
- **k-Nearest Neighbors (k-NN)**: Achieved a precision of 0.71 and an accuracy of 0.72.
- **Random Forest Classifier**: Provided robust classification with high accuracy and reliability.

### Clustering Results
The k-Means clustering effectively grouped addresses into distinct clusters, revealing patterns in transaction behaviors and highlighting different types of users in the Bitcoin network.

## Conclusion
This project demonstrates the use of data analysis and machine learning tools to derive insights from Bitcoin transactions. It highlights the strengths and weaknesses of Pandas and Spark in processing large datasets and shows the potential of classification and clustering techniques in understanding cryptocurrency transaction behaviors.
