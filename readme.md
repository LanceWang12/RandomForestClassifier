# Decision Tree Classifier

<img src="./image/tree.PNG" style="zoom:70%;" />

## 1. The parameters :

- max_depth: 決策樹的最大深度

- min_sample_split: 每次分裂時，若節點內的資料數少於該參數，則不分裂

- min_samples_leaf: 每個 leaf node 的資料數至少要大於該參數，否則剪枝

- max_features: 分裂時遍歷的特徵數，可選參數如下：
  - float (0, 1]: 選擇幾成的特徵
  - 'sqrt': 遍歷 sqrt(features) 的特徵
  - 'log2': 遍歷 log2(features) 的特徵
- min_impurity_decrease: 分裂時下降的不純度要大於該參數
- min_impurity_split: 分裂時不純度要大於該參數
- class_weight: 
  - None: 所有的類別權重都是一樣的
  - Balanced: 各類別數量的反比

## 2. The attributes :

- n_features_: 特徵的總數
- n_classes_: 類別的總數
- __cw: 各類別的權重
- tree_: 決策樹本身

## 3. The methods :

- fit(x, y): 訓練決策樹的函數
- predict(x): 預測的函數

## 4. Implementation:

- Algorithm: 參考 **[Cart Decision Tree](https://www.researchgate.net/publication/216526201_An_Improved_CART_Decision_Tree_for_Datasets_with_Irrelevant_Feature "Cart Decision Tree")**
- Speed up: 利用 Cython 將迴圈及基尼計算式等複雜計算打包成 C code 編譯並加速，但**速度仍差 sklearn 好大一截**

# Random Forest Classifier

<img src="./image/forest.jpg" alt="forest" style="zoom:45%;" />

## 1. The parameters :

- n_estimators: 決策樹的數量
- max_samples: 遍歷的資料數
- n_jobs: 使用的核心數
- 其他參數請參考決策樹部分

## 2. The attributes :

- n_features_: 特徵的總數
- n_classes_: 類別的總數
- __cw: 各類別的權重
- estimators_: 存取所有決策樹物件的 list

## 3. The methods :

- fit(x, y): 訓練決策樹的函數
- predict(x): 預測的函數

## 4. Implementation

- Algorithm: 參考 **[Random Forest](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf "Random Forest")**
- Speed up: 除了利用 Cython，也使用平行處理加速訓練速度，但**速度仍差 sklearn 好大一截**

### **Demo 請參考 main.py, 環境如 env.txt 所示**
