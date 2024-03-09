def feature_drop(data):
    
    return data.copy().loc[:, data.nunique() != 1].drop(columns=["feature642", "feature756"], axis = 1) 
