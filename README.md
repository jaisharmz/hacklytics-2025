# Jack Street
## About
Jack Street redefines financial data visualization with AI-driven temporal embeddings. By clustering stocks into insightful price baskets, we reveal macroeconomic trends and hidden market patterns, empowering investors with data-driven insights.

## Capabilities
### Natural Language Processing on Graphs
- Visualize financial news using NLP-driven graph representations.
### Pretraining a Foundation Model
- Pretrained a foundation model on historical financial data to generate temporal embeddings.
- PCA to visualizations of embeddings demonstrate that latent representations preserve key financial attributes such as:
    - Average percentage increase
    - Total percentage increase
    - Correlation with specific stocks
### Clustering Stocks Using Temporal Embeddings
- Apply K-Means clustering to latent representations, creating interpretable and meaningful clusters based on stock tickers.
- Form price baskets that represent different macroeconomic sectors, enabling better market tracking.
### Identifying Statistical Arbitrage Opportunities
- Detect stocks with similar historical trends but recent deviations, revealing potential statistical arbitrage opportunities.
- Example: If Coke and Pepsi typically move in sync but Coke’s stock drops unexpectedly, an investor might anticipate a price correction, creating a buying opportunity.
### Finding Similar Time Intervals
- Given a selected time period for a stock, identify other stocks with similar market behavior during that interval.

## User Interface & Experience
### Graph-Based Navigation
- A React-based interface presents scientific clustering of temporal embeddings in a hierarchical graph, resembling a Linux directory system.
- Users start with broad macroeconomic price baskets and can drill down into sub-clusters for more specific market segments.
### Interactive Visualizations & Statistics
- Cluster-Level Insights: View interactive graph visualizations and meta-statistics for each price basket, including average return rates and volatility.
- Stock-Level Insights: Access detailed stock data and compare time intervals to find similar behavioral patterns.
Identify statistical arbitrage opportunities by analyzing stocks with shared histories but diverging recent trends.