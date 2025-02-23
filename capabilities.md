# Jack Street
## About
Jack Street transforms financial data visualization with AI-driven temporal embeddings, clustering stocks into insightful price baskets that reveal macroeconomic trends and hidden market patterns.

## Capabilities
### Natural Language Processing Visualized on Graphs
- Graph using NLP to visualize financial news
### Pretraining a Foundation Model
- Foundation model for temporal embedding of historical financial data
- Visualize embeddings using PCA; color PCA by average percentage increase, total percentage increase, correlation with a particular stock; this shows that embeddings preserve important features about the data
### Clustering Stocks Using Temporal Embeddings
- KMeans clustering of latent representation data to divide data into clusters and subclusters; clusters and subclusters are interpretable based on stock tickers that are in those clusters
    - Create Price baskets of smaller sectors of the macroeconomy that can be tracked
### Statistical Arbitrage Opportunities
- Given a particular stock, identify statistical arbitrage opportunities based on data from past couple of months; find other stocks with similar histories but different data in the last couple of months
    - Example: If I know that coke and pepsi are almost always in line with each other, and coke stock fell today, i can make a reasonable assumption that if we buy coke that we will get some kind of profit.
### Similar Time Intervals
- Given a selected time interval for a stock, identify other stocks with similar behavior during that time period

## User Interface and User Experience
### Graph Representation
- The React interface embeds the scientific / data driven clustering of temporal embeddings in a graph-like structure similar to a Linux directory system
    - Users start with several broad clusters of the macroeconomy price basket
    - Then, users can click into sub clusters to get more specific about what type of price basket they want to track
### Visualizations and Statistics
- At tree nodes that are not individual stock tickers, users can view interactive graph visualizations as well as meta statistics about that cluster's price basket such as the average rate of return and volatility
- At leaf nodes that represent individual stock tickers, users can visualize stock data, select time intervals to find stocks with similar behavior during those intervals, and identify statistical arbitrage opportunities with stocks that have similar histories but diverging data in recent months