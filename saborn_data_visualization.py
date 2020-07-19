import numpy as np 
import seaborn as sns

data = np.random.randn(1000)
sns.distplot(data, kde=True, rug=True)
print(data)