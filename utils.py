def feature_drop(data):
    return data.copy().loc[:, data.nunique() != 1].drop(columns=["feature756", "feature642"])
    



def get_categorical_columns(df):
    binar = set(df.columns[df.nunique() == 2])
    cat_indexes = df[df.columns[~(df.nunique() == 2)]].nunique().\
    div((df[df.columns[~(df.nunique() == 2)]] != 0).sum().values, axis=0) * 100 <= 0.15

    potentially_categorical = binar.union(set(cat_indexes[cat_indexes == True].index))

    # из потенциально категориальных попробуем вычесть колонки, которые могут быть численными

    potentially_continuous = set(df.columns[(df.min(axis=0) == 0) & \
                                (df.max(axis=0) != 1) & \
                                df.isin([1]).any() & \
                                df.isin([2]).any() & \
                                df.isin([3]).any() & \
                                df.isin([4]).any() & \
                                df.isin([5]).any() & \
                                (df.nunique() < 500)])

    cat_cols = list(potentially_categorical - potentially_continuous)
    return cat_cols


def get_df1():

    file_path = "Data/train_ai_comp_final_dp.parquet"
    pf = ParquetFile(file_path)
    df = pf.to_pandas()
    return df




def remove_highly_correlated_features(X_train, shap_df, threshold=0.9):
    import pandas as pd
    import numpy as np

    df_copy = X_train.copy()

    # Если размер данных больше 100000 строк, делаем выборку
    if len(df_copy) > 100000:
        df_sample = df_copy.sample(n=100000, random_state=1)
    else:
        df_sample = df_copy

    corr_matrix = df_sample.corr().abs()

    # Получаем пары фич с высокой корреляцией
    high_corr_var = np.where(corr_matrix > threshold)
    high_corr_var = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if
                     x != y and x < y]

    # Подготавливаем список фич для удаления
    features_to_remove = []

    for feature_a, feature_b in high_corr_var:
        # Получаем SHAP значения для каждой фичи
        shap_a = shap_df.loc[shap_df['feature'] == feature_a, 'shap_importance'].values[0]
        shap_b = shap_df.loc[shap_df['feature'] == feature_b, 'shap_importance'].values[0]

        # Удаляем фичу с меньшим SHAP значением
        if shap_a < shap_b:
            features_to_remove.append(feature_a)
        else:
            features_to_remove.append(feature_b)

    # Удаляем дубликаты в списке фич для удаления
    features_to_remove = list(set(features_to_remove))

    # Возвращаем обновлённый DataFrame без удалённых фич
    return features_to_remove


def get_shap_feature(X_train, y_train, X_val, classifiers):
    import numpy as np
    import pandas as pd
    import shap

    # Список для хранения результатов по каждой модели
    models_shap_values = []

    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        X_sample = shap.utils.sample(X_train, 10000)  # Выборка из 100 наблюдений
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample)

        # Создаем DataFrame с SHAP значениями и фильтруем значимые признаки
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).query("shap_importance > 0")

        # Усредняем оставшиеся SHAP значения для каждой фичи
        feature_importance = feature_importance.groupby('feature', as_index=False).mean()

        models_shap_values.append(feature_importance)

    # Объединяем SHAP значения из всех моделей
    final_shap_df = pd.concat(models_shap_values).groupby('feature', as_index=False).mean()

    return final_shap_df


def download_raw_data_from_drive_and_open_in_pandas(file_id="1cS6pE2ZD127iSVEiLRDzGd_b65J7GTd9",
                                                    file_path="Data/raw_data.parquet"):
    import pandas as pd
    import gdown
    # скачивает файл
    gdown.download(id=file_id, output=file_path)

    return pd.read_parquet(file_path)