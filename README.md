# TSVM实现半监督检测 by KDD99
TSVM using kdd99

### prement_data.py 数据读取以及预处理
merge_sparse_feature()合并稀疏特征
在我们做特征工程的时候，可能会碰到一个特征我们假设其特征列的符号值为v，其特征存在多种取值，标签label设为y，特征v如果有很多特征值对应标签y是相同的，那么这些v之间是没有意义的，我们称之为稀疏特征。这个时候我们可以进行合并稀疏特征，因为合并稀疏特征不仅可以降低计算成本，它也最小化了样品错误分类的可能性。

get_dummies(）训练数据转换
利用get_dummies()实现特征编码，对分类型变量protocol_type，service，flag进行编码处理。

get_lable()标签转换
正常数据normal标签转换为-1，异常数据abnormal标签转换为1。


    
### TSVM.py 算法模型
参考链接：https://github.com/horcham/TSVM
