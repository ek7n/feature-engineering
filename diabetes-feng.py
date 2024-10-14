import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df= pd.read_csv("datasets/diabetes.csv")

#feature engineering

df.shape
df.head()
df.info()

df.describe().T

df.columns = [col.upper() for col in df.columns]

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

df.head()
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
df.columns
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df,"OUTCOME" )

df.head()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "AGE")
num_summary(df, "DIABETESPEDIGREEFUNCTION")

for col in num_cols:
    num_summary(df, col)


df.groupby("OUTCOME")[num_cols].mean()

#Outlier examination

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col , check_outlier(df, col))

#replace outliers with certain thresholds
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)





#Handling missing values

df.isnull().sum()

df.describe().T

selected_cols = ["GLUCOSE","INSULIN","BMI","SKINTHICKNESS","BLOODPRESSURE"]

## dataset does not contain missing values but there are some 0s which should be handled as NAs. Insulin, Blood pressure cannot be O ie.


df[selected_cols] = df[selected_cols].apply(lambda x: x.replace(0,np.nan))
#filling NAs based on target variable's classes
df["INSULIN"] = df["INSULIN"].fillna(df.groupby("OUTCOME")["INSULIN"].transform("median"))
df["SKINTHICKNESS"] = df["SKINTHICKNESS"].fillna(df.groupby("OUTCOME")["SKINTHICKNESS"].transform("mean"))

num_summary(df,"SKINTHICKNESS")


# checking the correlation between the variables, target and features specifically before feature extraction
corr_matrix = df.corr()
corr_matrix
outcome_corr = corr_matrix[['OUTCOME']].drop('OUTCOME').sort_values(by="OUTCOME",ascending=False)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

#feature extraction

df["NEW_PEDIGREE"] = pd.qcut(df["DIABETESPEDIGREEFUNCTION"], q=5, labels=['Düşük', 'Orta-düşük', 'Orta', 'Orta-yüksek', 'Yüksek'])

cat_summary(df,"NEW_PEDIGREE")
num_summary(df,"AGE")

df.loc[(df['AGE'] < 35), 'NEW_AGE'] = 'young'
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 50), 'NEW_AGE'] = 'mid'
df.loc[(df['AGE'] >= 51), 'NEW_AGE'] = 'senior'


df.loc[(df['GLUCOSE'] < 100), 'NEW_GLUCOSE'] = 'normal'
df.loc[(df['GLUCOSE'] >= 100) & (df['GLUCOSE'] <= 125), 'NEW_GLUCOSE'] = 'risky'
df.loc[(df['GLUCOSE'] >= 126), 'NEW_GLUCOSE'] = 'high'

df.head()
df.shape

#encoding

#since we add new variables regenerating variable lists
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

#checking rareness of the categories at cat_cols
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

#if there rare classes merge them as "rare"
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

rare_analyser(df, "OUTCOME", cat_cols)

cat_summary(df, "NEW_GLUCOSE")
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder_cols = [col for col in cat_cols if col not in "OUTCOME"]


#encoding ordinal columns with label encoder
for col in label_encoder_cols:
    label_encoder(df, col)

df.head()

#scaling
#scaling numerical columns with robust scaler
scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()
df.head()

df.shape

#model

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)

from sklearn.model_selection import GridSearchCV

# Hiperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Ağaç sayısı
    'max_depth': [10, 20, None],     # Maksimum derinlik
    'min_samples_split': [2, 5, 10], # Bir düğümün bölünmesi için gereken minimum örnek sayısı
    'min_samples_leaf': [1, 2, 4],   # Bir yaprak düğümünde bulunması gereken minimum örnek sayısı
    'max_features': ['sqrt', 'log2', None]  # Her ağacı oluştururken değerlendirilecek maksimum özellik sayısı
}

# Random Forest model
rf = RandomForestClassifier(random_state=46)

# Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Train model again
grid_search.fit(X_train, y_train)

# En iyi parametreler ve sonuçlar
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi doğruluk skoru:", grid_search.best_score_)

#best score:0.8883523710626514