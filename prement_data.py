import pandas as pd
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import numpy as np
import sklearn.svm as svm
from sklearn.externals import joblib
import pickle
from sklearn.model_selection import train_test_split,cross_val_score
from TSVM import TSVM


def get_dummies(X_train):
       """
       train data covert
       :param X_train:
       :return:
       """
       df_dummies_1 = pd.get_dummies(X_train['protocol_type'],prefix='protocol_type')
       df_dummies_2 = pd.get_dummies(X_train['service'],prefix='service')
       df_dummies_3 = pd.get_dummies(X_train['flag'],prefix='flag')
       df_dummies = pd.concat([df_dummies_1,df_dummies_2,df_dummies_3], axis = 1)
       X_train = pd.concat([X_train,df_dummies],axis=1)

       X_train.drop(['protocol_type','service','flag'],axis = 1, inplace = True)
       return X_train


def get_lable(Y_train):
       """
       lable data covert, normal data lable=> -1, abnormal data lable=> 1
       :param Y_train:
       :return:
       """
       for i in range(len(Y_train)):
              if Y_train[i] == 'normal.':
                     Y_train[i] = -1
              else:
                     Y_train[i] = 1
       return Y_train


def merge_sparse_feature(df):
       """
       Merge sparse matrix
       :param df:
       :return:
       """
       df.loc[(df['service'] == 'ntp_u')
           | (df['service'] == 'urh_i')
           | (df['service'] == 'tftp_u')
           | (df['service'] == 'red_i')
       , 'service'] = 'normal_service_group'

       df.loc[(df['service'] == 'pm_dump')
           | (df['service'] == 'http_2784')
           | (df['service'] == 'harvest')
           | (df['service'] == 'aol')
           | (df['service'] == 'http_8001')
       , 'service'] = 'satan_service_group'
       return df


pd.set_option('display.max_columns',None)
lable_data = pd.read_csv("./data/kddcup.data_10_percent", sep=",", header=None,
                      names=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
                             'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell',
                             'su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
                             'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate',
                             'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
                             'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
                             'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','category'])
lable_data = merge_sparse_feature(lable_data)
X_train = lable_data.iloc[:40000,:41]
Y_train = lable_data.iloc[:40000, 41]
Y_train=np.expand_dims(Y_train,1)


Y_train = get_lable(Y_train)
X_train = get_dummies(X_train)

X_scaled = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = X_scaled.transform(X_train)


#Split training and testing dataset
X_no_lable = X_train[10000:, :]
X_train = X_train[:10000, :]
Y_train = Y_train[:10000, :]
X_train_have_lable, X_test, Y_train_have_lable, Y_test = train_test_split(X_train, Y_train, test_size=0.5)


model = TSVM()
model.initial()
print("X train shape:",X_train_have_lable.shape,"X'lable Y shape:",Y_train_have_lable.shape,"X no lable shape:",X_no_lable.shape)
model.train(X_train_have_lable, Y_train_have_lable, X_no_lable)
Y_hat = model.predict(X_test)
Y_hat = np.expand_dims(Y_hat, 1)
Y_hat = list(map(lambda x: int(x), Y_hat))
Y_test = list(map(lambda x: int(x), Y_test))
print("Accuracy for SVM: ", accuracy_score(Y_test, Y_hat))
print(classification_report(Y_test, Y_hat))
