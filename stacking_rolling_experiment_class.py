#!/usr/bin/env python
# coding: utf-8

# # Stacking :rolling predict, file output

# check scikit-learn version
# import sklearn
# print(sklearn.__version__)

import numpy as np
from numpy import std
from matplotlib import pyplot
import tqdm
import pandas as pd
import time

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.ensemble import StackingRegressor

# For Feature Selection
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr


# # class
class ROLLING_EXP:
    def __init__(self, models_set, level0_set, level1_set, passthrough, n_tscv, X, Y ):
        self.level0_set = level0_set
        self.level1_set = level1_set
        self.passthrough = passthrough
        self.n_tscv = n_tscv
        self.X = X
        self.Y = Y
        self.final_res = dict()
        self.models = dict()
        self.n_test_ori = 0
        self.merged_df = dict()
        self.merged_df_test = dict()
        self.merged_res = dict()
        self.models_set = models_set
        self.sd = 0

    def get_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.Y,
                                                            test_size=0.2,
                                                            shuffle=False)
        X_train, X_test, y_train, y_test = self.np_cc_feature_select(
            X_train, X_test, y_train, y_test)
        self.n_test_ori = len(y_test)
        self.sd = std(y_train)
        return X_train, X_test, y_train, y_test

    def Remove_MCL_feature(self, X_train, y_train):  #input = dataframe format
        y_train = np.array(y_train, dtype=np.float)
        y_train = y_train.flatten()
        # clustering
        distance = 1 - abs(np.corrcoef(X_train.values, rowvar=False))
        d = squareform(distance, checks=False)
        Z = linkage(d, method='complete')
        f = fcluster(Z, t=0.2, criterion='distance')
        Group_Index = pd.DataFrame({
            'Variable': X_train.columns.to_list(),
            'Group_Index': f
        })
        # Calculate the correlation between FDC (log p-value)
        #     Correlation = np.apply_along_axis(lambda x: -np.log(pearsonr(y_train.values, x)[1]), 0, X_train.values)
        Correlation = np.apply_along_axis(
            lambda x: -np.log(pearsonr(y_train, x)[1]), 0, X_train.values)
        Correlation = pd.DataFrame({
            'Variable': X_train.columns.to_list(),
            'Correlation': Correlation
        })
        # Pick the selected SVIDs
        Features_Info = pd.merge(Group_Index, Correlation, on="Variable")
        selected_features = Features_Info.loc[Features_Info.groupby(
            ["Group_Index"], sort=False).Correlation.idxmax(
            ), 'Variable'].to_list()
        return selected_features

    def np_cc_feature_select(self, X_train, X_test, y_train, y_test):
        '''
        input = np.array to dataframe format for Remove_MCL_feature
        then
        output = np.array for next usage
        '''
        # arr to df
        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        X_test = pd.DataFrame(X_test)
        print('Original:')
        print('X_train_shape:', X_train.shape)
        print('X_test_shape:', X_test.shape)

        # Remove the columns that is constant
        features = X_train.columns[~(X_train.std() < 1e-6)]
        X_train = X_train.loc[:, features]
        X_test = X_test.loc[:, features]
        print('\nAfter remove feature with low variance')
        print('X_train_shape:', X_train.shape)
        print('X_test_shape:', X_test.shape)

        # Deal the Multicollinearity problem
        features = self.Remove_MCL_feature(X_train, y_train)
        X_train = X_train.loc[:, features]
        X_test = X_test.loc[:, features]
        print('\nAfter deal multicollinearity problem')
        print('X_train_shape:', X_train.shape)
        print('X_test_shape:', X_test.shape)

        # df to arr
        X_train, X_test, y_train = X_train.values, X_test.values, y_train.values
        return X_train, X_test, y_train, y_test

    # ---evaluation---
    def my_rsq_f(self, y_true, y_pred):
        print(f'y_true={y_true}; y_pred={y_pred}')
        SS_res = np.sum([(y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i])
                         for i in range(len(y_true))])
        SS_tot = np.sum([
            (y_true[i] - np.mean(y_true)) * (y_true[i] - np.mean(y_true))
            for i in range(len(y_true))
        ])
        return (1 - SS_res / (SS_tot + 0.000000001))

    def itv_acc(self, y_true, y_pred):
        rate = 0.5
        hit = [
            1 for i in range(len(y_pred))
            if (y_pred[i] >= y_true[i] - rate * self.sd)
            & (y_pred[i] <= y_true[i] + rate * self.sd)
        ]
        return sum(hit) / len(y_true)

    def eval_model_test(self, fitted_model, X_test, y_test):
        list_y_pred = [fitted_model.predict([data])[0] for data in X_test]
        #     list_y_pred = [fitted_model.predict(data) for data in X_test]
        df_test = pd.DataFrame(zip(y_test, list_y_pred))
        df_test.columns = ['y_true', 'y_pred']
        df_test['y_true'] = [x[0] for x in df_test['y_true']]
        itv_accu = round(self.itv_acc(df_test['y_true'], df_test['y_pred']), 3)
        r2 = round(self.my_rsq_f(df_test['y_true'], df_test['y_pred']), 3)
        res = {'itv_accu': itv_accu, 'r2': r2}
        print('res = ', res)
        return df_test, res

    def plot_pred_trend(self, df_res, title):
        pyplot.figure(figsize=(20, 8))
        pyplot.plot(df_res['y_true'], label="y_true")
        pyplot.plot(df_res['y_pred'], label="y_pred")
        pyplot.title(title, size=15)
        pyplot.legend()
        pyplot.show()
        pyplot.close()

    def print_stage_res(self, df):
        print('***total***')
        print('itv_acc =', round(self.itv_acc(df['y_true'], df['y_pred']), 3))
        print('r2 =', round(self.my_rsq_f(df['y_true'], df['y_pred']), 3))

        print('*** 0~999')
        print('itv_acc =',
              round(self.itv_acc(df['y_true'][:999], df['y_pred'][:999]), 3))
        print('r2 =',
              round(self.my_rsq_f(df['y_true'][:999], df['y_pred'][:999]), 3))

        print('*** 1000~1999')
        print(
            'itv_acc =',
            round(
                self.itv_acc(df['y_true'][1000:1999].reset_index(drop=True),
                             df['y_pred'][1000:1999].reset_index(drop=True)),
                3))
        print(
            'r2 =',
            round(
                self.my_rsq_f(df['y_true'][1000:1999].reset_index(drop=True),
                              df['y_pred'][1000:1999].reset_index(drop=True)),
                3))

        print('*** 2000~')
        print(
            'itv_acc =',
            round(
                self.itv_acc(df['y_true'][2000:].reset_index(drop=True),
                             df['y_pred'][2000:].reset_index(drop=True)), 3))
        print(
            'r2 =',
            round(
                self.my_rsq_f(df['y_true'][2000:].reset_index(drop=True),
                              df['y_pred'][2000:].reset_index(drop=True)), 3))

    # ---rolling predict and result---
    def merge_dataset_for_rolling(self, X_train, X_test, y_train, y_test):
        all_X = np.concatenate((X_train, X_test), axis=0)
        all_y = np.concatenate((y_train, y_test), axis=0)
        return all_X, all_y

    def rolling_train_pred(self, all_X, all_y):
        # --- train and predict rolling ---
        tscv = TimeSeriesSplit(n_splits=self.n_tscv)

        # rolling
        c = 0
        for train_index, test_index in tscv.split(all_X):
            #(1)滾動
            #     if c == 0:
            #         n = len(train_index)
            #         c += 1
            #(2)累積滾動
            n = len(train_index)
            train_ind_rolling = train_index[-n:]
            print('len of rolling: ', len(train_ind_rolling))
            print("TRAIN:", train_ind_rolling, "TEST:", test_index)
            print(
                f'**train size:{len(train_ind_rolling)}; **test size:{len(test_index)}'
            )
            X_train, X_test = all_X[train_ind_rolling], all_X[test_index]
            y_train, y_test = all_y[train_ind_rolling], all_y[test_index]

            # set model
            self.models = self.set_and_get_models(X_train, y_train)
            dict_df_test = dict()
            dict_res = dict()
            for name in tqdm.tqdm(self.models.keys()):
                print(name)
                model = self.models[name]
                df_test, res = self.eval_model_test(model, X_test, y_test)
                dict_df_test[name] = df_test
                dict_res[name] = res

            self.final_res[test_index[0]] = {
                'test_index': test_index,
                'df_test': dict_df_test,
                'res': dict_res
            }

    def merge_result(self):
        # merge res
        for m in tqdm.tqdm(self.models.keys()):
            for i in self.final_res.keys():
                if i == list(self.final_res.keys())[0]:
                    df = self.final_res[i]['df_test'][m]
                else:
                    df_add = self.final_res[i]['df_test'][m]
                    df = pd.concat([df, df_add], axis=0, ignore_index=True)
            self.merged_df[m] = df
            df_test = df[-self.n_test_ori:].reset_index(drop=True)
            self.merged_df_test[m] = df_test
            # get only test range:
            itv_accu = round(
                self.itv_acc(df_test['y_true'], df_test['y_pred']), 3)
            r2 = round(self.my_rsq_f(df_test['y_true'], df_test['y_pred']), 3)
            self.merged_res[m] = {'itv_accu': itv_accu, 'r2': r2}

    def show_testset_result(self):
        # show res added
        for m in tqdm.tqdm(self.models.keys()):
            print(self.merged_res[m])
            self.plot_pred_trend(self.merged_df_test[m], m)

    def deal_ensemble_outlier_get_res(self):
        df_stacking = self.merged_df_test['stacking']
        df_outlier = df_stacking[(df_stacking['y_pred'] < 3050) |
                                 (df_stacking['y_pred'] > 3400)]
        print(df_outlier)
        df_stacking_rm_out = df_stacking[(df_stacking['y_pred'] > 3050) & (
            df_stacking['y_pred'] < 3400)].reset_index(drop=True)
        self.print_stage_res(df_stacking_rm_out)
        self.plot_pred_trend(df_stacking_rm_out, 'stacking')

    # ---define models---
    def get_stacking(self):
        # define the base models
        level0 = list()
        for name in self.level0_set:
            level0.append((name, self.models_set[name]))

        # define meta learner model
        level1 = self.models_set[self.level1_set]
        # define the stacking ensemble
        model = StackingRegressor(estimators=level0,
                                  final_estimator=level1,
                                  passthrough=self.passthrough,
                                  cv=5)
        return model

    def set_and_get_models(self, X_train, y_train):
        models_new = dict()
        stacking_model = self.get_stacking()
        stacking_model.fit(X_train, y_train)
        base_models = stacking_model.estimators_
        i = 0
        for name in self.level0_set:
            models_new[name] = base_models[i]
            i = i + 1
        models_new['stacking'] = stacking_model
        return models_new

    def dict_df_to_list(self, dict_df):
        dict_npy = dict()
        for k in dict_df.keys():
            dict_npy[k] = dict_df[k].to_numpy().tolist()
        return dict_npy

    def json_output(self):
        log = {
            'n_cv_rolling': self.n_tscv,
            'li_level0_set': self.level0_set,
            'level_1_model': self.level1_set,
            'passthrough': self.passthrough
        }
        t = time.localtime()
        output = dict()
        output['log'] = log
        output['merged_all_true_pred'] = self.dict_df_to_list(self.merged_df)
        output['merged_test_true_pred'] = self.dict_df_to_list(self.merged_df_test)
        output['merged_res'] = self.merged_res
        with open(f'output_{str(t.tm_mday)+str(t.tm_hour)+str(t.tm_min)}.json',
                  'w') as f:
            json.dump(output, f)
            

def main(exp_obj):
    X_train, X_test, y_train, y_test = exp_obj.get_dataset()
    all_X, all_y = exp_obj.merge_dataset_for_rolling(X_train, X_test, y_train, y_test)
    exp_obj.rolling_train_pred(all_X, all_y)
    exp_obj.merge_result()
    exp_obj.show_testset_result()
    exp_obj.deal_ensemble_outlier_get_res()
    exp_obj.json_output()

if __name__ == '__main__':
    # FDC dataset
    X = np.load('FDC_X.npy')
    Y = np.load('FDC_Y.npy')


    # exp setting
    level0_set = [
        ['br'],
        ['rf', 'gb', 'knn', 'br'],
        ['rf', 'gb', 'knn', 'br', 'svm'],
        ['rf', 'gb', 'knn', 'br', 'svm', 'mlp']
    ]
    # remember to update the tunned model to models_set
    models_set = {
        'rf': RandomForestRegressor(),
        'gb': GradientBoostingRegressor(),
        'knn': KNeighborsRegressor(),
        'br': BayesianRidge(),
        'svm': SVR(),
        'mlp': MLPRegressor(),
        'lr': LinearRegression(),
        'rcv': RidgeCV()
    }

    level1_set = ['lr', 'rcv']
    passthrough = [False, True]
    n_tscv = [5, 10]

    # run experiments
    for level1 in tqdm.tqdm(level1_set):
        for level0 in tqdm.tqdm(level0_set):
            exp1 = ROLLING_EXP(models_set, level0, level1, passthrough[0],
                               n_tscv[0], X, Y)
            main(exp1)
            exp2 = ROLLING_EXP(models_set, level0, level1, passthrough[1],
                               n_tscv[0], X, Y)
            main(exp2)
            exp3 = ROLLING_EXP(models_set, level0, level1, passthrough[0],
                               n_tscv[1], X, Y)
            main(exp3)
            exp4 = ROLLING_EXP(models_set, level0, level1, passthrough[1],
                               n_tscv[1], X, Y)
            main(exp4)

