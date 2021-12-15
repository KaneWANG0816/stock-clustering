import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Problem2: Clustering analysis on rate of return (RoR) with different durations
# Clustering based on the most essential indicator, rate of return = (V2 - V1)/V1,
# where V1= initial value and V2 = Current value
# Also, the close price is chosen for calculating the RoR

# Moreover, in this problem, we calculate RoRs of 7 days, 30 days, 180 days, 360 days and 1080 days
durations = [7, 30, 180, 360, 1080]
stock_ids = ["1", "11", "293", "857", "13", "23"]


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


rawByColumn = pd.read_excel("SixHKStockData.xls", "With Graph")
# print(rawByColumn.info())

# Get close values only
data_closePrice = pd.DataFrame()
data_closePrice["1"] = rawByColumn["close-001"].values
data_closePrice["11"] = rawByColumn["Close-011"].values
data_closePrice["293"] = rawByColumn["Close-293"].values
data_closePrice["857"] = rawByColumn["Close-857"].values
data_closePrice["13"] = rawByColumn["Close-13"].values
data_closePrice["23"] = rawByColumn["Close-23"].values
# print(data_closePrice.info())

# Drop rows with missing data
data_closePrice = data_closePrice.drop(data_closePrice.index[range(68)])
# print(data_closePrice.info())

# Calculate RoRs for each stock in different durations
data = dict()
for duration in durations:
    RoRs = pd.DataFrame()
    for stock_id in stock_ids:
        current = data_closePrice[stock_id]
        initial = current.shift(duration)
        RoRs.loc[:, stock_id] = ((current - initial) / initial).values
    # Drop NaN rows
    RoRs = RoRs.dropna()
    data[duration] = RoRs

# Print RoRs tables of each duration
for duration in data.keys():
    # print("RoRs of {} days:".format(duration))
    # print(data[duration].info())
    # print("\n")

    # Transform for clustering
    data[duration] = data[duration].values.T
    # print(data[duration].shape)

# Plot RoRs  of each duration
for duration in data.keys():
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.title("RoRs of {} days".format(duration))
    plt.xlabel("date")
    for i in range(6):
        plt.plot(range(len(data[duration][i])), data[duration][i], label=stock_ids[i], linewidth=1)
    leg = plt.legend(loc='upper right')
    # plt.savefig("RoRs_of_{}_days.png".format(duration))
    plt.show()


# Print RoRs indicators of each duration
for duration in data.keys():
    print("For RoRs of {} days in percentage:".format(duration))
    stat = pd.DataFrame({'indicator': ["Average", "Max", "Min"]})
    for i in range(6):
        avg = np.round(np.average(data[duration][i]) * 100, 2)
        max = np.round(np.max(data[duration][i]) * 100, 2)
        min = np.round(np.min(data[duration][i]) * 100, 2)
        stat["stock {}".format(stock_ids[i])] = np.array([avg, max, min])
    print(stat)

# Clustering
for duration in data.keys():
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clustering.fit(data[duration])
    plt.title("Hierarchical Clustering for {} days".format(duration))
    plot_dendrogram(stock_ids, clustering, truncate_mode="level", p=3)
    plt.xlabel("Stocks")
    # plt.savefig("Clustering_of_{}_days.png".format(duration))
    plt.show()
