from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse

# 创建空DataFrame用于保存NLP特征
train_x = pd.DataFrame()
test_x = pd.DataFrame()

# 实例化CountVectorizer评估器与TfidfVectorizer评估器
cntv = CountVectorizer()
tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)

# 创建空列表用户保存修正后的列名称
vector_feature =[]
for co in ['merchant_id', 'merchant_category_id', 'state_id', 'subsector_id', 'city_id']:
    vector_feature.extend([co+'_new', co+'_hist', co+'_all'])
    
# 提取每一列进行新特征衍生
for feature in vector_feature:
    print(feature)
    cntv.fit(train[feature].append(test[feature]))
    train_x = sparse.hstack((train_x, cntv.transform(train[feature]))).tocsr()
    test_x = sparse.hstack((test_x, cntv.transform(test[feature]))).tocsr()
    
    tfv.fit(train[feature].append(test[feature]))
    train_x = sparse.hstack((train_x, tfv.transform(train[feature]))).tocsr()
    test_x = sparse.hstack((test_x, tfv.transform(test[feature]))).tocsr()
    
# 保存NLP特征衍生结果
sparse.save_npz("preprocess/train_nlp.npz", train_x)
sparse.save_npz("preprocess/test_nlp.npz", test_x)