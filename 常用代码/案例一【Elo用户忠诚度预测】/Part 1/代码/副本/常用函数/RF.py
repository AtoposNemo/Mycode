import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from hyperopt import hp, fmin, tpe

#定义网格搜索函数--RF
def param_grid_search(train):
    """
    网格搜索参数调优
    :param train:训练集
    :return:网格搜索训练结果
    """
    # Step 1.创建网格搜索空间
    print('param_grid_search')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    parameter_space = {
        "n_estimators": [81], 
        "min_samples_leaf": [31],
        "min_samples_split": [2],
        "max_depth": [10],
        "max_features": [80]
    }
    
    # Step 2.执行网格搜索过程
    print("Tuning hyper-parameters for mse")
    # 实例化随机森林模型
    clf = RandomForestRegressor(
        criterion="mse",
        n_jobs=15,
        random_state=22)
    # 带入网格搜索
    grid = GridSearchCV(clf, parameter_space, cv=2, scoring="neg_mean_squared_error")
    grid.fit(train[features].values, train['target'].values)
    
    # Step 3.输出网格搜索结果
    print("best_params_:")
    print(grid.best_params_)
    means = grid.cv_results_["mean_test_score"]
    stds = grid.cv_results_["std_test_score"]
    # 此处额外考虑观察交叉验证过程中不同超参数的
    for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    return grid
###调用 #返回最佳参数 grid = param_grid_search(train_RF)   
        #            grid.best_estimator_
        #            grid.best_estimator_.predict(test)


###用评估器实现交叉验证的集成
def train_predict(train, test, best_clf):
    """
    进行训练和预测输出结果
    :param train:训练集
    :param test:测试集
    :param best_clf:最优的分类器模型
    :return:
    """
    
    # Step 1.选择特征
    print('train_predict...')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")

    # Step 2.创建存储器
    # 测试集评分存储器
    prediction_test = 0
    # 交叉验证评分存储器
    cv_score = []
    # 验证集的预测结果
    prediction_train = pd.Series()
    
    # Step 3.交叉验证
    # 实例化交叉验证评估器
    kf = KFold(n_splits=5, random_state=22, shuffle=True)
    # 执行交叉验证过程
    for train_part_index, eval_index in kf.split(train[features], train['target']):
        # 在训练集上训练模型
        best_clf.fit(train[features].loc[train_part_index].values, train['target'].loc[train_part_index].values)
        # 模型训练完成后，输出测试集上预测结果并累加至prediction_test中
        prediction_test += best_clf.predict(test[features].values)
        # 输出验证集上预测结果，eval_pre为临时变量
        eval_pre = best_clf.predict(train[features].loc[eval_index].values)
        # 输出验证集上预测结果评分，评估指标为MSE
        score = np.sqrt(mean_squared_error(train['target'].loc[eval_index].values, eval_pre))
        # 将本轮验证集上的MSE计算结果添加至cv_score列表中
        cv_score.append(score)
        print(score)
        # 将验证集上的预测结果放到prediction_train中
        prediction_train = prediction_train.append(pd.Series(best_clf.predict(train[features].loc[eval_index]),
                                                             index=eval_index))
    
    # 打印每轮验证集得分、5轮验证集的平均得分
    print(cv_score, sum(cv_score) / 5)
    # 验证集上预测结果写入本地文件
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_randomforest.csv", index=False)
    # 测试集上平均得分写入本地文件
    pd.Series(prediction_test / 5).to_csv("preprocess/test_randomforest.csv", index=False)
    # 在测试集上加入target，也就是预测标签
    test['target'] = prediction_test / 5
    # 将测试集id和标签组成新的DataFrame并写入本地文件，该文件就是后续提交结果
    test[['card_id', 'target']].to_csv("result/submission_randomforest.csv", index=False)
    return

##调用 train_predict(train_RF, test_RF, grid.best_estimator_)
#此时，在本地文件中就能看到一个新的结果文件，该预测结果是交叉验证后各模型的预测结果的均值，
# 相当于是一次简单的“集成”，前者是单模型结果，后者是手动集成后的模型结果


##wrapper特征筛选 特征筛选过程，此处先择使用Wrapper方法进行特征筛选，
# 通过带入全部数据训练一个LightGBM模型，然后通过观察特征重要性，选取最重要的300个特征。
# 当然，为了进一步确保挑选过程的有效性，此处我们考虑使用交叉验证的方法来进行多轮验证。
# 实际多轮验证特征重要性的过程也较为清晰，我们只需要记录每一轮特征重要性，并在最后进行简单汇总即可。

def feature_select_wrapper(train, test):
    """
    lgm特征重要性筛选函数
    :param train:训练数据集
    :param test:测试数据集
    :return:特征筛选后的训练集和测试集
    """
    
    # Part 1.划分特征名称，删除ID列和标签列
    print('feature_select_wrapper...')
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')

    # Step 2.配置lgb参数
    # 模型参数
    params_initial = {
        'num_leaves': 31,
        'learning_rate': 0.1,
        'boosting': 'gbdt',
        'min_child_samples': 20,
        'bagging_seed': 2020,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'max_depth': -1,
        'metric': 'rmse',
        'reg_alpha': 0,
        'reg_lambda': 1,
        'objective': 'regression'
    }
    # 控制参数
    # 提前验证迭代效果或停止
    ESR = 30
    # 迭代次数
    NBR = 10000
    # 打印间隔
    VBE = 50
    
    # Part 3.交叉验证过程
    # 实例化评估器
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    # 创建空容器
    fse = pd.Series(0, index=features)
    
    for train_part_index, eval_index in kf.split(train[features], train[label]):
        # 封装训练数据集
        train_part = lgb.Dataset(train[features].loc[train_part_index],
                                 train[label].loc[train_part_index])
        # 封装验证数据集
        eval = lgb.Dataset(train[features].loc[eval_index],
                           train[label].loc[eval_index])
        # 在训练集上进行训练，并同时进行验证
        bst = lgb.train(params_initial, train_part, num_boost_round=NBR,
                        valid_sets=[train_part, eval],
                        valid_names=['train', 'valid'],
                        early_stopping_rounds=ESR, verbose_eval=VBE)
        # 输出特征重要性计算结果，并进行累加
        fse += pd.Series(bst.feature_importance(), features)
    
    # Part 4.选择最重要的300个特征
    feature_select = ['card_id'] + fse.sort_values(ascending=False).index.tolist()[:300]
    print('done')
    return train[feature_select + ['target']], test[feature_select]

##调用 train_LGBM, test_LGBM = feature_select_wrapper(train, test)，接下来，我们即可带入经过筛选的特征进行建模
