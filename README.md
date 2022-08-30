# Admirable methods 🔄🌲✨📈 

Algorithms from [ADMIRABLE Research Group](https://admirable-ubu.es/) from Burgos' University 

## How to
### Rotate a dataset
```python
from rotation import Rotation
import numpy as np

data = np.random.randint(0,100,(10000,30))

rot = Rotation(group_size=3, group_weight=.5)

data_rotated = rot.fit_transform(data)
```
### Rotation Forest

```python
from rotation import Rotation, RotationForest
from sklearn.utils import check_random_state
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

random_state = check_random_state(42)

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.33, random_state=random_state)
 
rf = RotationForestClassifier(base_estimator=DecisionTreeClassifier(random_state=random_state), 
                              n_estimators=10, min_group_size=3, max_group_size=3, 
                              rotation=Rotation(group_weight=.5), random_state=random_state)
rf.fit(X_train, y_train)
rf.score(X_test, y_test) # 0.98
```
