# MMA-Classification
ECS784U Project

***Installations and libraries required for UFC Decision Tree file***
```
!pip install xlrd
!conda install -c conda-forge pydotplus -y
!conda install -c conda-forge python-graphviz -y
from sklearn.externals.six import StringIO #imports all python packages for plotting a decision tree
import pydotplus #Provides a Python interface to Graphviz Dot language
import matplotlib.image as mpimg #Provides an image module to support image loading rescaling and display operations.
from sklearn import tree #imports software for plotting decision trees
from sklearn.tree import DecisionTreeClassifier #imports Decision Tree Classifier model
from sklearn import preprocessing #imports preprocessing library
import numpy as np #imports numpy libary to allow access to large multi, dimensional arrays and matrices along with a series of high-level mathematical functions
from sklearn import metrics #imports metrics library
import pandas as pd #imports pandas libaray for working with structured data sets
from sklearn.model_selection import train_test_split #import train test split method
import matplotlib.pyplot as plt #import matplotlib library to provide visualisation of data
import seaborn as sns #import seaborn library
from sklearn.metrics import classification_report, confusion_matrix #Confusion matrix is imported
import itertools
```

***Installations and libraries required for UFC Logistic Regression file***
```
import pandas as pd #Provides high performance easy to use data structures and operations for manipulating numerical tables.
import pylab as pl #Imports portions of matplotlib and numpy to provide a MATLAB like experience.
import numpy as np #Math Library that allows one to work with N-Dimensional arrays in Python and enables one to do computation efficiently and effectively
import scipy.optimize as opt #Provides several commonly used optimization algorithms.
from sklearn import preprocessing #Contains several common utility functions and tranformer classes.
import matplotlib.pyplot as plt #Provides a MATLAB like plotting function
import seaborn as sns #imports seaborn library
from sklearn.linear_model import LogisticRegression #Import Logistic Regression and fit to the X and y training data
from sklearn import metrics #imports metric libraries
from sklearn.model_selection import train_test_split #import train test split evaluation metric
from sklearn import preprocessing # preprocesses the data and converts the categorical variable into dummy/indicator variables
from sklearn.metrics import log_loss #Try log loss for evaluationg the output is the probability of customer churn is yes
from sklearn.metrics import classification_report, confusion_matrix #Confusion matrix is imported
```

