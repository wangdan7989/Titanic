#coding=utf-8
# data analysis and wrangling
import pandas as pd
import numpy as np

def split_train_test():
    train_df = pd.read_csv('../input/titanic/train.csv')
    test_df = pd.read_csv('../input/titanic/test.csv')
    combine = [train_df, test_df]
    #print(train_df.columns.values);
    #print(test_df.columns.values);
    #print(train_df.head());
    #print(train_df.shape)
    #print(test_df.shape)
    #print(train_df.info())
    #print(train_df.describe())

    #pivot = train_df[['Pclass', 'Survived']]
    pivot = train_df.groupby(['Survived'], as_index=False)
    #print(pivot.size())

    #删除票号和船舱号的特征
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df,test_df]

    #创建名字的title特征
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    pd.crosstab(train_df['Title'], train_df['Sex'])

    #用常见的tile代替不常见的title
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace([
                'Lady', 'Countess','Capt', 'Col',
                'Don', 'Dr', 'Major', 'Rev', 'Sir',
                'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    pivot = train_df[['Title', 'Survived']]
    pivot = pivot.groupby(['Title'], as_index=False).mean()
    pivot.sort_values(by='Survived', ascending=False)

    #将title转换成数字类型
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    #删除Name  PassengerId
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    #将sex转换成数字类型（分类特征）
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    #填充缺失数据，age根据pclass（1,2,3）和sex来进行填充（0,1）两两组合看是什么age
    guess_ages = np.zeros((2,3))
    for dataset in combine:
        for i in range(0,2):
            for j in range(0,3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()
                age_guess = guess_df.median()
                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)

    #根据年龄分段进行划分类别
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    pivot = train_df[['AgeBand', 'Survived']]
    pivot = pivot.groupby(['AgeBand'], as_index=False).mean()
    pivot.sort_values(by='AgeBand', ascending=True)

    for dataset in combine:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']

    #删除ageband字段
    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]

    #创建familysize特征
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    pivot = train_df[['FamilySize', 'Survived']]
    pivot = pivot.groupby(['FamilySize'], as_index=False).mean()
    pivot.sort_values(by='Survived', ascending=False)

    #根据familysize特征创建isalone
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

    #因为familysize特征和surval没有相关性，所以用alone，故删除'Parch', 'SibSp', 'FamilySize'
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    #创建age和pclass积的特征
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    #因为Embarked只有两个缺失值，所以用mode填充即可
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    pivot = train_df[['Embarked', 'Survived']]
    pivot = pivot.groupby(['Embarked'], as_index=False).mean()
    pivot.sort_values(by='Survived', ascending=False)

    #将Embarked转换成分类特征
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map(
            {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    #fare只有一个缺失值，故用中位数填充，然后创建分段特征
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    pivot = train_df[['FareBand', 'Survived']]
    pivot = pivot.groupby(['FareBand'], as_index=False).mean()
    pivot.sort_values(by='FareBand', ascending=True)

    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) &
                    (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) &
                    (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    return X_train,Y_train,X_test,train_df,test_df

if __name__ == '__main__':
    split_train_test()