import numpy as np
from sklearn.decomposition import PCA
import sklearn.utils
from sklearn.base import TransformerMixin, ClusterMixin, ClassifierMixin

class Rotation(TransformerMixin):
    
    def __init__(self, group_size=3, group_weight=.5, random_state=None):
        self.group_size=group_size
        self.rand=sklearn.utils.check_random_state(random_state)
        self.group_weight=group_weight
        self.groups = []
        self.pcas = []
        
    def fit(self, X):
        rows, cols = X.shape
        cl = list(range(cols))
        rl = list(range(rows))
        # Primero se randomizan las columnas
        self.rand.shuffle(cl)
        # Se generan las columnas para los grupos
        idx = 0
        while idx < len(cl):
            gr = []
            for i in range(self.group_size):
                if i+idx >= len(cl):
                    gr.append(self.rand.choice(cl))
                else:
                    gr.append(cl[i+idx])
            
            self.groups.append(gr)
            idx += self.group_size
        # Se han generado los grupos de las columnas
        # Con esto se crean subconjuntos de los datos
        groups_X = []
        for g in self.groups:
            groups_X.append(X[:,g])
        # De cada grupo con sus datos se escogen aleatoriamente el porcentaje deseado
        groups_T = []
        for g in groups_X:
            # Se "barajean" las filas
            self.rand.shuffle(rl)
            # Se escogen las primeras (group_weight*total_columnas)
            sel = int(self.group_weight*rows)
            groups_T.append(g[0:sel,:])
            
        # Una vez se tienen los objetos para entrenar entonces se crean los PCA
        for g in groups_T:
            p = PCA(self.group_size)
            p.fit(g)
            self.pcas.append(p)
            
        # Ya está el modelo entrenado            
        
            
    def transform(self, X):
        assert len(self.pcas) > 0 and len(self.pcas) == len(self.groups), "No se ha generado al transformación"
        # Se crean los subconjuntos de los datos transformados
        tformed = []
        for i in range(len(self.pcas)):
            pca = self.pcas[i]
            group = self.groups[i]
            x_n = X[:,group]
            x_t = pca.transform(x_n)
            tformed.append(x_t)
            
        return np.concatenate(tformed,axis=1)
            
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
class RotatedCluster(ClusterMixin):
    
    def __init__(self, cluster, rotation=None, random_state=None):
        self.estimator = cluster
        self.rotation = rotation
        if self.rotation is None:
            self.rotation = Rotation(random_state=random_state)
        self.rand = random_state
        
    def fit(self,X):
        X = self.rotation.fit_transform(X)
        self.estimator.fit(X,y)
        
    def predict(self,X):
        X = self.rotation.transform(X)
        return self.estimator.predict(X)
    
    def fit_predict(self,X):
        X = self.rotation.fit_transform(X)
        return self.estimator.fit_predict(X)
            
