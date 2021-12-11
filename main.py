import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(labels, model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=labels, **kwargs)


# Problem1: which company are similar(correlation), for diversifying portfolio?
# Problem2: which features are suitable for stock price prediction?
# Problem3: which

# read excel file, sheet="Stacked"
raw_stacked = pd.read_excel("SixHKStockData.xls", "Stacked")
# print(raw_stacked.values.shape)
# print(raw_stacked.head(3))
# print(raw_stacked.isnull().all())

stock_ids = raw_stacked['stock_id'].unique()
# print(stocks)

# Normalization: Min-Max scaling for each column
indicators = ['open', 'close', 'high', 'low', 'volume']
data = raw_stacked.copy()
for feature_name in raw_stacked.columns:
    if feature_name in indicators:
        max_value = raw_stacked[feature_name].max()
        min_value = raw_stacked[feature_name].min()
        data[feature_name] = (raw_stacked[feature_name] - min_value) / (max_value - min_value)
# print(data.head(3))

# Split data based on stock_id
splits = dict()
for stock_id in stock_ids:
    stock = data[indicators][data["stock_id"] == stock_id]
    splits[stock_id] = stock.values

# Stock 857 has missing data
for stock_id in splits.keys():
    print("Number of days of stock {}: {}".format(stock_id, splits[stock_id].shape[0]))

# Simply drop days with missing values
for stock_id in splits.keys():
    if stock_id != 857:
        splits[stock_id] = splits[stock_id][68:, :]
for stock_id in splits.keys():
    print("After drop, number of days of stock {}: {}".format(stock_id, splits[stock_id].shape[0]))

# Reshape for clustering
for stock_id in splits.keys():
    splits[stock_id] = splits[stock_id].ravel()

training = np.array(list(splits.values()))
print(training.shape)

# clustering
clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
clustering.fit(training)

# plot
plt.title("Hierarchical Clustering")
plot_dendrogram(stock_ids, clustering, truncate_mode="level", p=3)
plt.xlabel("Stocks")
plt.show()
plt.savefig('Hierarchical Clustering.png')