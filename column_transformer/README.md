# ColumnTransformer and Pipeline

## ColumnTransformer

- ColumnTransformer is a class that allows to apply different transformations to different columns or variables in a dataset.
- It helps to streamline the preprocessing pipeline by applying specific transformations to specific subsets of features.
- CoulmnTransformer helps when a dataset contains different types of data such as discrete, continuous, nominal, and ordinal which require to preprocess before applying any machine learning algorithms.

## Why Need?

Consider the below dataset where we need to perform different preprocessing techniques before sending data to machine learning algorithms

![image](https://github.com/smitesht/machine_learning/assets/52151346/53dd4314-649e-4bd7-ae44-613a2f6fb4cb)

- Each preprocessing technique generates one or multiple columns based on the nature of the technique. For example, StandardScaler, and OrdinalEncoder generate single columns while OneHotEncoder generates multiple columns.
- If we apply all the preprocessing methods individually, we have to concate all the columns in a single dataset which is sometimes a tedious task for a large number of columns.

  ![image](https://github.com/smitesht/machine_learning/assets/52151346/16657ce4-b802-41ea-930e-2ac9bbb3240a)

## Without ColumnTransformer

```python
  # Perform OneHotEncoder on Location, FuelType and Transmission
# Location: 11 -> one remove -> 10
# Fuel_Type: 4 -> one remove -> 3
# Transmission: 2-> one remove -> 1
# Total 14 column must be generated through OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
X_train_ohe = ohe.fit_transform(X_train[['Location','Fuel_Type','Transmission']])  

X_test_ohe = ohe.transform(X_test[['Location','Fuel_Type','Transmission']])
X_train_ohe.shape

# Ordinal Columns
# I am considering Year as an ordinal column as price may vary based on Year
# Year: lower (2006) -> higher (2019)
# Owner_Type: lower (Fourth & Above) -> higher (First)
# Seats: 4, 5, 7, 8

ordEnc = OrdinalEncoder(categories=[
    ["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"],
    ["Fourth & Above","Third","Second","First"],
    [4.0, 5.0,7.0,8.0]
    ])

X_train_ordencode = ordEnc.fit_transform(X_train[["Year","Owner_Type","Seats"]])

X_test_ordencode = ordEnc.transform(X_test[["Year","Owner_Type","Seats"]])

# Apply Standardization to Kilometers_Driven, Mileage

stdScalar = StandardScaler()

X_train_scaled = stdScalar.fit_transform(X_train[['Kilometers_Driven','Mileage']])

X_test_scaled = stdScalar.transform(X_test[['Kilometers_Driven','Mileage']])

X_train_all = np.concatenate((X_train_scaled, X_train_ordencode, X_train_ohe), axis=1)
X_test_all = np.concatenate((X_test_scaled, X_test_ordencode, X_test_ohe), axis= 1)

```

## With ColumnTransfer

![image](https://github.com/smitesht/machine_learning/assets/52151346/115fb18c-1bbc-4de5-8e40-132f5fa65dd4)

```python
# Perform OneHotEncoder on Location, FuelType and Transmission
# Location: 11 
# Fuel_Type: 4 
# Transmission: 2
# Total 17 column must be generated through OneHotEncoder

# Ordinal Columns
# I am considering Year as an ordinal column as price may vary based on Year
# Year: lower (2006) -> higher (2019)
# Owner_Type: lower (Fourth & Above) -> higher (First)
# Seats: 4, 5, 7, 8

encoding_trns = ColumnTransformer(
    [
        ("OneHot_Encoder",  OneHotEncoder(sparse_output=False, handle_unknown='ignore'),[0, 3, 4 ]  ),
        
        ("Ordinal_Encoder", OrdinalEncoder(
            categories=[
                ["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"],
                ["Fourth & Above","Third","Second","First"],
                [4.0, 5.0,7.0,8.0]
                    ]), [1,5,7] )
    ],    
    remainder='passthrough'
)

# Apply Standardization to Kilometers_Driven, Mileage
stnd_scaler_trns = ColumnTransformer(
    [("std_scaler", StandardScaler(),[2, 6 ]  )],
    remainder='passthrough'
)

```

![image](https://github.com/smitesht/machine_learning/assets/52151346/ffc40777-53aa-4c40-bba3-ed02c6e0cab5)


## Pipeline

- A pipeline is a sequence of data processing steps that are chained together.
- The purpose of the pipeline is to streamline the workflow by bundling together multiple preprocessing steps and model fitting into a single object. 

![image](https://github.com/smitesht/machine_learning/assets/52151346/742fa8e4-0119-43d0-a0bd-2b668e89c68e)

## ColumnTransformer with Pipeline

```python
# Perform OneHotEncoder on Location, FuelType and Transmission
# Location: 11 
# Fuel_Type: 4 
# Transmission: 2
# Total 17 column must be generated through OneHotEncoder

# Ordinal Columns
# I am considering Year as an ordinal column as price may vary based on Year
# Year: lower (2006) -> higher (2019)
# Owner_Type: lower (Fourth & Above) -> higher (First)
# Seats: 4, 5, 7, 8

encoding_trns = ColumnTransformer(
    [
        ("OneHot_Encoder",  OneHotEncoder(sparse_output=False, handle_unknown='ignore'),[0, 3, 4 ]  ),
        
        ("Ordinal_Encoder", OrdinalEncoder(
            categories=[
                ["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"],
                ["Fourth & Above","Third","Second","First"],
                [4.0, 5.0,7.0,8.0]
                    ]), [1,5,7] )
    ],    
    remainder='passthrough'
)

# Apply Standardization to Kilometers_Driven, Mileage
stnd_scaler_trns = ColumnTransformer(
    [("std_scaler", StandardScaler(),[2, 6 ]  )],
    remainder='passthrough'
)

rf = RandomForestRegressor(n_estimators = 100)

pipe = Pipeline([
    ('encoding',encoding_trns),
    ('standard_scaler',stnd_scaler_trns ) ,
    ('random_forest',rf) 
])

pipe.fit(X_train, y_train)

```
![image](https://github.com/smitesht/machine_learning/assets/52151346/bc1a5965-1f4d-430a-9e34-d2b322122ec8)

## Benefits of Pipeline

- It bundles multiple preprocessing steps and models fitting into a single object which simplifies the code and makes it easy to manage
- Automate the preprocessing and modeling process.
- In production, we do not need to remember the order of preprocessing steps.
- Improve code readability
- Encapsulate all preprocessing steps and model fitting into a single object which makes it easy to reuse the same processing steps and model across 

