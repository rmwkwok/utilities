import time
from ast import literal_eval
from copy import deepcopy as dc
from multiprocessing import Process, Queue
from sklearn.model_selection import ParameterGrid

class GS:
    def __init__(self, cls, param_grid, n_jobs = 1):
        self.cls = cls
        self.n_jobs = n_jobs
        self.param_grid = param_grid
        self.best_score_ = 0
        
    def _worker(self, ansQ, parameter, cls, *args, **kwargs):
        cls = cls.set_params(**parameter).fit(*args, **kwargs)
        ansQ.put( (parameter, list(list(cls.best_score_.values())[0].values())[0]) )
        
    def _reader(self, ansQ):
        try:
            parameter, best_score = ansQ.get(True, 2)
            self.rank[str(parameter)] = best_score
            self.best_params_ = min(self.rank, key=lambda x: self.rank[x])
            self.best_score_ = self.rank[self.best_params_]
        except Exception as e: 
            pass
        
    def fit(self, *args, **kwargs):
        self.rank, self.ansQ, self.workers, tasks = {}, Queue(), [], list(ParameterGrid(self.param_grid))
        
        while len(tasks) or len(self.workers):
            time.sleep(2)
            self._reader(self.ansQ)
            self.workers = [p for p in self.workers if p.is_alive()]
            
            if len(tasks) and len(self.workers) < self.n_jobs:
                self.workers.append(Process(target=self._worker, args=(self.ansQ,tasks.pop(0),dc(self.cls))+args, kwargs=kwargs))
                self.workers[-1].start()
                
            print(f'# To be ran {len(tasks)}, # Active Thread {len(self.workers)}, Best {self.best_score_}', end=' '*20+'\r')
            
        self.best_cls_ = self.cls.set_params(**literal_eval(self.best_params_)).fit(*args, **kwargs)
        
        print(f'Best score: {self.best_score_}')
        print(f'Best parameter: {self.best_params_}')
        return self
        
    def predict(self, *args, **kwargs):
        return self.best_cls_.predict(*args, **kwargs)