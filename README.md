# Rotated methods 🔄🌲✨📈 

Rotated algorithms  based on Rotation Forest by [Rodriguez et. al.](https://doi.org/10.1109/TPAMI.2006.211)

## How to

```python
from rotation import Rotation
import numpy as np

data = np.random.randint(0,100,(10000,30))

rot = Rotation(group_size=3, group_weight=.5)

data_rotated = rot.fit_transform(data)
```
