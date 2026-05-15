import os
import joblib
import numpy as np
import pandas as pd
import streamlit as strl

strl.set_page_config(
    page_title= 'House Pirce Prediction ',
    page_icon="🏠",
    layout= 'centered'
)

strl.title('Оценка Недвижимости')
strl.write('Оценка стоимости недвижимости на основе ансамбля  CatBoost, XGB, LGBM and Ridge')

MODELS_DIR = 'models'

try:
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    ridge_model = joblib.load(os.path.join(MODELS_DIR, 'ridge_model.joblib'))
    lgbm_model = joblib.load(os.path.join(MODELS_DIR, 'lgbm_model.joblib'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    cat_model = joblib.load(os.path.join(MODELS_DIR, "cat_model.joblib"))

except Exception as e:
    strl.error('Critical Error')
    strl.stop()
strl.sidebar.header('House Characteristics')

overall_qual = strl.sidebar.slider('общее количество материалов и отделки', 1, 10, 6)
overall_cond = strl.sidebar.slider('общее состояние дома', 1, 10, 5)

total_bsmt_sf = strl.sidebar.number_input('площадь подвала кв. футы', min_value=0, value=1000)
bsmt_fin_sf1 = strl.sidebar.number_input('готовая площадь подвала первого типа', min_value=0, value=40)
bsmt_fin_sf2 = strl.sidebar.number_input('готовая площадь подвала второго типа', min_value=0, value=0)
gr_liv_area = strl.sidebar.number_input('жилая площадь над землей', min_value=100, value=1500)
garage_area = strl.sidebar.number_input('площадь гаража', min_value=0, value=480)

year_built = strl.sidebar.number_input('год постройки', min_value=1800, max_value=2026, value=2000)
year_remod = strl.sidebar.number_input('год реновации', min_value=1800, max_value=2026, value=2000)
yr_sold =  strl.sidebar.number_input(' год продажи', min_value=2000, max_value=2026, value=2010)

neighborhood = strl.sidebar.selectbox(
    'район',
    ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge' ,'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer']
)

ms_zoning = strl.sidebar.selectbox('зона застройки', ['RL', 'RM', 'C (all)', 'FV', 'RH'])
kitchen_qual = strl.sidebar.selectbox('качество кухни', ['Ex', 'Gd', 'TA', 'Fa', 'Po'])






if strl.button("расчитать рыночную стоимость"):
    
    base_input = {
        "MSSubClass": 60,
        "LotFrontage": 70.0,
        "LotArea": 10000,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "MasVnrArea": 0.0,
        "ExterQual": "TA",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "Functional": "Typ",
        "Fireplaces": 1,
        "GarageType": "Attchd",
        "GarageYrBlt": 2000.0,
        "GarageFinish": "RFn",
        "GarageCars": 2.0,
        "GarageQual": "TA",
        "GarageCond": "TA",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 0,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 6,
        "SaleType": "WD",
        "SaleCondition": "Normal",
        "BsmtUnfSF": 600.0,
        "Heating": "GasA",
        "BsmtQual": "TA",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinType2": "Unf",
        "1stFlrSF": 1100.0,  
        "2ndFlrSF": 400.0,
        'FireplaceQu': 'None',  
        'LowQualFinSF': 0.0    
    }

    
    user_updates = {
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "TotalBsmtSF": total_bsmt_sf,
        "BsmtFinSF1": bsmt_fin_sf1,
        "BsmtFinSF2": bsmt_fin_sf2,
        "GrLivArea": gr_liv_area,
        "GarageArea": garage_area,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "YrSold": yr_sold,
        "Neighborhood": neighborhood,
        "MSZoning": ms_zoning,
        "KitchenQual": kitchen_qual,
    }
    base_input.update(user_updates)
    input_df = pd.DataFrame([base_input])

    
    input_df["TotalSF"] = (
        input_df["TotalBsmtSF"] + input_df["1stFlrSF"] + input_df["2ndFlrSF"]
    )
    input_df["TotalBath"] = input_df["BedroomAbvGr"] * 0.5 + 1.0
    input_df["HouseAge"] = input_df["YrSold"] - input_df["YearBuilt"]
    input_df["RemodAge"] = input_df["YrSold"] - input_df["YearRemodAdd"]
    input_df["AreaPerRoom"] = input_df["GrLivArea"] / 6.0
    input_df["HasGarage"] = (input_df["GarageArea"] > 0).astype(int)
    input_df["HasBsmt"] = (input_df["TotalBsmtSF"] > 0).astype(int)
    input_df["hasFirePlace"] = 1

    
    cols_to_drop = [
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "TotRmsAbvGrd",
    ]
    col_exst = [col for col in cols_to_drop if col in input_df.columns]
    df_clean = input_df.drop(col_exst, axis=1)

    try:
        
        X_ready = preprocessor.transform(df_clean)
        pred_ridge_log = ridge_model.predict(X_ready)
        pred_lgbm_log = lgbm_model.predict(X_ready)

        final_log = (0.3 * pred_ridge_log) + (0.7 * pred_lgbm_log)

        # ИСПРАВЛЕНО: Извлекаем скалярное число через приведение типов float()
        final_price = float(np.expm1(final_log).flatten()[0])

        strl.success("расчет успешно завершен")
        strl.metric(
            label="рекомендуемая стоимость ", value=f"$ {final_price:,.2f}"
        )

        col1, col2 = strl.columns(2)

        
        total_sf_val = int(input_df["TotalSF"].iloc[0])
        house_age_val = int(input_df["HouseAge"].iloc[0])

        col1.metric(
            "Синтезированная Общая площадь (TotalSF)", f"{total_sf_val:,} кв. фт"
        )
        col2.metric("Возраст здания на момент продажи", f"{house_age_val} лет")

    except Exception as error:
        strl.error(f"ошибка в расчете : {error}")