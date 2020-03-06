# Machine Learning on Yelp User Review Data

The task is to read the json file, and put clustering algorithm to check different clusters based on the user compliments of different businesses on yelp

# Prerequisites

# Installing

```Libraries
import pandas as pd
import numpy as np
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
%matplotlib inline
```
# Code

Template code is provided in the yelp_clustering.ipynb notebook file. The datset used is user.json from the yelp dataset. 

# Run

In a terminal or command window, navigate to the top-level project directory Machine-Learning-on-Yelp-Data/ (that contains this README) and run one of the following commands:

```
ipython notebook yelp_clustering.ipynb
```

Or
```
jupyter notebook yelp_clustering.ipynb
```

# Data

The dataset is provided by <a href="https://www.yelp.com/dataset/challenge">Yelp</a>, and have different types of dataset in the repository. For our problem, we have used the user.json data from the dataset. The dataset has initially 22 columns, and have over 2 million data points, for the simplicity, I took only the 1 million data from the user.json file.



