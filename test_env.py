import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit
import joblib

print("=" * 45)
print("  All imports successful!")
print(f"  pandas      : {pd.__version__}")
print(f"  numpy       : {np.__version__}")
import sklearn
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  streamlit   : {streamlit.__version__}")
print("=" * 45)