import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
# Problem1: which features are suitable for stock price prediction?
# Problem2: which company are similar(correlation), for diversifying portfolio?
# Problem3: which

# read excel file, sheet="Stacked"
raw_stacked = pd.read_excel("SixHKStockData.xls", "Stacked")
print(raw_stacked.head(3))
# print(raw_stacked.isnull().all())


# Define a normalizer
normalizer = Normalizer()
# Create Kmeans model
kmeans = KMeans(n_clusters = 10,max_iter = 1000)
# Make a pipeline chaining normalizer and kmeans
pipeline = make_pipeline(normalizer,kmeans)
# Fit pipeline to daily stock movements
pipeline.fit(movements)
labels = pipeline.predict(movements)

