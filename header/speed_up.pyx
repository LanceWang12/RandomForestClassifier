import numpy as np
import math
import random
import cython

cpdef float compute_gini(class_arr, class_num, n_classes):
    # class_arr: an array record every number of class
    # class_num: sum(class_arr), but I don't computer in this function because too much complexity
    # n_classes: the number of class
    cdef float Sum = 0, gini = 0
    for i in range(n_classes):
        Sum += ((class_arr[i] / class_num) ** 2)
    gini = 1 - Sum

    return gini

cpdef discrete_or_continual(x):
    # discrete: return 1
    # continual: return 0
    judge = (x % 1) == 0

    return False in judge

cpdef compute_class_num(n_classes, y):
    # cdef int class_num[6]
    class_num = np.zeros(n_classes)
    for i in range(n_classes):
        class_num[i] = np.sum(y == i)
    return np.asarray(class_num)

cpdef compute_class_weight(class_weight, n_classes, y):
    # class_weight: a flag
    # n_classes: How many classes in the dataset
    # y: the label in the dataset
    # set the class weight
    if  class_weight == 'balanced':
        # class_num: the number of every class in this dataset
        class_num = compute_class_num(n_classes, y)
            
        ones = np.ones(len(class_num))
        tmp = (ones / class_num)
        class_weight = tmp / np.sum(tmp)
    elif class_weight is None:
        cw = np.ones(n_classes)
    #else:
    #    raise ValueError('Please set class weight to balanced or None!')
    return class_weight

cpdef judge_impurity_decrease(gini, total_gini, min_impurity_decrease):
    # judge wether the impurity decrease is bigger than min impurity decrease
    return (total_gini - gini) > min_impurity_decrease

cpdef judge_impurity_split(gini, min_impurity_split):
    # judge wether the impurity is bigger than min_impurity_split
    return gini > min_impurity_split

cpdef select_features(n_features_idx, max_features):
    np.random.shuffle(n_features_idx)

    if (type(max_features) == type(0.8) or max_features == 1):
        if (max_features < 0) or (max_features) > 1:
            raise ValueError('max_features must in [0, 1].')
        length = n_features_idx.shape[0] * max_features
    elif (max_features == 'auto') or (max_features == 'sqrt'):
        length = np.sqrt(n_features_idx.shape[0])
    elif max_features == 'log2':
        length = np.log2(n_features_idx.shape[0])
    elif max_features is None:
        length = n_features_idx.shape[0]
    length = int(np.ceil(length))
    n_features_idx = n_features_idx[: length]

    return n_features_idx

cpdef split(
    x, y, min_samples_split, n_classes_, n_features_, max_features,
    class_weight, min_impurity_decrease, min_impurity_split
):
    # The number of data
    cdef int data_num = len(y)

    if data_num < min_samples_split:
        # end of split
        return None, None
    
    # class_num: the number of every class in this dataset
    class_num = compute_class_num(n_classes_, y)

    cdef int best_idx = -1
    cdef float best_thr = -1
    
    # best_gini: the gini factor of the parent node
    cdef float best_gini = compute_gini(class_num, data_num, len(class_num))
    cdef float parent_gini = best_gini

    # randomize the features
    n_features_idx = np.arange(n_features_)
    n_features_idx = select_features(n_features_idx, max_features)

    cdef int c = 0
    cdef float gini_left = 0, gini_right = 0, total_gini = 0

    for idx in n_features_idx:
        # visit every feature
        decision_thresholds, classes = zip(*sorted(zip(x[:, idx], y)))

        d_or_c_flag = discrete_or_continual(x[:, idx])

        num_left = np.asarray([0] * n_classes_)
        num_right = class_num.copy()
        for i in range(1, data_num):
            # visit every data
            c = classes[i - 1]
            num_left[c] += 1
            num_right[c] -= 1

            gini_left = compute_gini(num_left, i, n_classes_)
            gini_right = compute_gini(num_right, data_num - i, n_classes_)

            # combine gini with class weight
            total_gini = (i * gini_left + (data_num - i) * gini_right) / data_num
            total_gini *= class_weight[c]

            if decision_thresholds[i] == decision_thresholds[i - 1]:
                continue
            
            if total_gini < best_gini:
                best_gini = total_gini
                best_idx = idx

                if d_or_c_flag:
                    # is discrete feature
                    best_thr = decision_thresholds[i]
                else:
                    # is continual feature
                    best_thr = (decision_thresholds[i] + decision_thresholds[i - 1]) / 2

    # pre-prunning
    if judge_impurity_decrease(best_gini, parent_gini, min_impurity_decrease):
        if judge_impurity_split(best_gini, min_impurity_split):
            return best_idx, best_thr
        else: 
            return None, None
    else:
        return None, None