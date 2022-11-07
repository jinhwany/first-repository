#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Module Import 

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# sklearn 라이브러리에 있는 datasets 을 가지고 오는데 datasets 안에 있는 load_digits를 가져온다.
# sklearn 라이브러리에 있는 model_selection 을 가지고 오는데 model_selection 안에 있는 train_test_split를 가져온다.
# sklearn 라이브러리에 있는 metrics을 가지고 오는데 metrics 안에 있는 classification_report를 가져온다.

# In[37]:


digits


# digits 내용 확인해보기

# In[48]:


digits = load_digits()
digits.keys()


# digits 딕셔너리 키 확인

# In[5]:


import pandas as pd
digits_df = pd.DataFrame(data=digits.data, columns=digits.feature_names)
digits_df['labels'] = digits.target
digits_df


# 안에 있는 data는 digits.data을 가져온것이고, columns=digits.feature_names 이 내용은, 각 딕셔너리의 키를 보기 좋게 정리해주는 도움 역할을 한다.

# In[52]:


digits_df.info()


# 같은 데이터속에서도 다양한 해설을 알아 낼 수 있다.

# In[53]:


digits_df.describe()


# 같은 데이터속에서도 다양한 해설을 알아 낼 수 있다.(2)

# In[55]:


# Feature Data 지정하기
digits_data = digits.data
digits.data


# 왜 _랑 .랑 별차이가 없는데 굳이 바꿀까?
# 그냥 이뻐서..?

# In[7]:


# Label Data 지정하기
digits_label = digits.target
digits_label


# In[ ]:


get_ipython().set_next_input('왜 _랑 .랑 별차이가 없는데 굳이 바꿀까');get_ipython().run_line_magic('pinfo', '바꿀까')
그냥 이뻐서..?


# In[8]:


# Target Names 출력해 보기
digits.target_names


# 우리가 확인해야되는 값들에 대한 확인?

# In[9]:


# 데이터 Describe 해 보기
digits.DESCR


# dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR']) 에 있어서 뜨는거다.

# 다른건 다 뜨는데 왜 'digits.frame' 얘는 안뜰까?

# In[35]:


x_train, x_test, y_train, y_test = train_test_split(digits_data,
                                                    digits_label,
                                                    test_size = 0.2,
                                                    random_state = 1)


# 왜 테스트 사이즈는 0.2가 좋은걸까? 더 좋은 환경은 없는건가?

# 왜 x 랑 y 위치를 바꾸면 에러가 뜨는거지? 순서가 왜 중요하지?

# In[36]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# train
decision_tree = DecisionTreeClassifier(random_state=1) #랜덤이 진짜 말그대로 무작위가 아니라, 특정 알고니즘에 의한 랜덤이라 항상 같은 변수의 랜덤이 있다. 그래서 랜덤을 정해놓는다.
decision_tree.fit(x_train, y_train) #x_train = 문제지, #y_train = 정답지

# test
y_pred_dt = decision_tree.predict(x_test)

## acc
acc = accuracy_score(y_test, y_pred_dt)
acc


# 훈련 모델을 불러와서, 문제와 정답을 학습시킨다.

# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# train
random_forest = RandomForestClassifier(random_state=1)
random_forest.fit(x_train, y_train)

# test
y_pred_rf = random_forest.predict(x_test)

# acc
acc = accuracy_score(y_test, y_pred_rf)
acc


# 의사결정나무보다 랜덤포레스트가 더 정확도가 높다.

# In[27]:


from sklearn import svm
from sklearn.metrics import accuracy_score

# train
svm_model = svm.SVC(random_state=1)
svm_model.fit(x_train, y_train)

# test
y_pred_svm = svm_model.predict(x_test)

# acc
acc = accuracy_score(y_test, y_pred_svm)
acc


# 가장 높은 정확도를 보여준다.

# In[28]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# train
sgd_model = SGDClassifier(random_state=1)
sgd_model.fit(x_train, y_train)

# test
y_pred_sgd = sgd_model.predict(x_test)

# acc
acc = accuracy_score(y_test, y_pred_sgd)
acc


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# train
logistic_model = LogisticRegression(max_iter = 3000, random_state=1)
logistic_model.fit(x_train, y_train)

# test
y_pred_log = logistic_model.predict(x_test)

# acc
acc = accuracy_score(y_test, y_pred_log)
acc


# In[30]:


from sklearn.metrics import classification_report

print("Decision Tree")
print(classification_report(y_test, y_pred_dt))
print("------------------------------------------------------")
print()

print("Random Forest")
print(classification_report(y_test, y_pred_rf))
print("------------------------------------------------------")
print()

print("SVM")
print(classification_report(y_test, y_pred_svm))
print("------------------------------------------------------")
print()

print("SGD Classifier")
print(classification_report(y_test, y_pred_sgd))
print("------------------------------------------------------")
print()

print("Logistic Regression")
print(classification_report(y_test, y_pred_log))


# 최종적으로 5가지의 훈련테스트 방법중에 가장 높은 정확도를 보여주는것은 svm_model이었다. 여러 훈련테스트 방법들을 통해 가장 높은 정확도를 선택하는것이, 가장 효율적인 훈련방법이라고 볼 수 있다.

# 해당 코드는 모두 https://github.com/JaeHeee/AIFFEL_Project/blob/master/EXPLORATION/EXPLORATION%202.%20Iris%EC%9D%98%20%EC%84%B8%20%EA%B0%80%EC%A7%80%20%ED%92%88%EC%A2%85%2C%20%EB%B6%84%EB%A5%98%ED%95%B4%EB%B3%BC%20%EC%88%98%20%EC%9E%88%EA%B2%A0%EC%96%B4%EC%9A%94%3F.ipynb 에서 가져왔으며, 입력한 이후 입력된 코딩을 이해하는데 초점을 두었습니다.
