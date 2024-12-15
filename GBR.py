from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
from sklearn.pipeline import make_pipeline

data = pd.read_excel('criteria_train.xlsx', header=0, index_col = 0)  

X = data.iloc[:,1:].apply(lambda x: round(x, 4))
y = data.iloc[:,0].values

model = make_pipeline(
    MinMaxScaler(),
    GradientBoostingRegressor()
)

model.fit(X, y)

crossvalidation = KFold(n_splits =129, shuffle=True)
rmse_score_gbr = cross_val_score(model, X, y, scoring = 'neg_root_mean_squared_error',cv = crossvalidation)

print('CV_results:')
print('Folds: %i, mean_RMSE: %.3f' % (len(rmse_score_gbr),-rmse_score_gbr.mean()))
