import numpy as np
from scipy import stats
from multiprocessing import cpu_count, Process, Queue, process, Pool
from header.unit import select_samples
from header.sys import start_processes, join_processes
from header.tree import DecisionTreeClassifier

class RandomForestClassifier():
    def __init__(
        self, n_estimators=100, criterion='gini', max_depth=None,
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=0, bootstrap=True, oob_score=False, n_jobs=None,
        random_state=None, verbose=0, warm_start=False, class_weight=None,
        ccp_alpha=0.0, max_samples=None
        ):

        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples_split, min_samples_leaf = min_samples_split, min_samples_leaf
        self.max_features, self.min_impurity_decrease = max_features, min_impurity_decrease
        self.min_impurity_split, self.bootstrap = min_impurity_split, bootstrap
        self.oob_score, self.n_jobs = oob_score, n_jobs
        self.verbose, self.class_weight = verbose, class_weight
        self.max_samples = max_samples

        # check n_jobs
        if self.n_jobs is None:
            self.__cpu_num = 1
        elif self.n_jobs == -1:
            self.__cpu_num = cpu_count()
        else:
            if cpu_count() < self.n_jobs:
                raise SystemError('The number of cpu in this device is smaller than parameter: n_jobs. Please reset!!')
            self.__cpu_num = self.n_jobs

        # construct n estimators
        self.estimators_ = []
        self.base_estimator = DecisionTreeClassifier(
            criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf, max_features = max_features,
            min_impurity_decrease = min_impurity_decrease, min_impurity_split = min_impurity_split,
            class_weight = class_weight
        )
        # for i in range(self.n_estimators):
        #     self.estimators_.append(
            #     DecisionTreeClassifier(
            #         criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split,
            #         min_samples_leaf = min_samples_leaf, max_features = max_features,
            #         min_impurity_decrease = min_impurity_decrease, min_impurity_split = min_impurity_split,
            #         class_weight = class_weight
            #     )
            # )
        

    def fit(self, x, y):
        # queue = Queue()
        # processes = []

        # Every Decision tree may choose different samples
        select_lst = []
        for i in range(self.n_estimators):
            if self.max_samples is None:
                x_select, y_select = x, y
            else:
                select_idx = select_samples(len(x), self.max_samples)
                x_select, y_select = x[select_idx], y[select_idx]
                select_lst.append((x_select, y_select))
            # processes.append(Process(target=self.estimators_[i].fit, args=(x_select, y_select, queue)))

        # runtime = self.n_estimators / self.__cpu_num
        # if runtime < 1:
        #     start_processes(processes)
        #     join_processes(processes)
        # else:
        #     for i in range(int(runtime)):
        #         start_processes(processes, start = i * self.__cpu_num, end = (i + 1) * self.__cpu_num)
        #         join_processes(processes, start = i * self.__cpu_num, end = (i + 1) * self.__cpu_num)
            
        #     # remain run time
        #     remain = int(self.n_estimators % self.__cpu_num)
        #     start_processes(processes, start = self.n_estimators - remain, end = self.n_estimators)
        #     join_processes(processes, start = self.n_estimators - remain, end = self.n_estimators)

        # start_processes(processes)
        # join_processes(processes)

        # for i in range(self.n_estimators):
        #     self.estimators_[i] = queue.get()
        pool = Pool(self.__cpu_num)
        self.estimators_ = pool.starmap(self.base_estimator.fit, select_lst)


    def predict(self, x):
        # pool
        result = []

        for i in range(self.n_estimators):
            result.append(self.estimators_[i].predict(x))

        result = stats.mode(np.array(result), axis = 0)[0][0]
        return result