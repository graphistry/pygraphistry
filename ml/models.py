import matplotlib.pyplot as plt
import numpy as np

# import optuna
import pandas as pd
import seaborn as sns
import sklearn
import umap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor

# from xgboost import XGBClassifier, plot_importance

if True:
    from sklearn.experimental import enable_hist_gradient_boosting

    # now you can import the HGBR from ensemble, which is a good predictor for data with heterogeneous columns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from ml import get_botnet_dataframe, get_netflow_dataframe, get_host_dataframe
from ml.utils import fit_pipeline, process_dirty_dataframes, calculate_column_similarity


def get_data(model="botnet", nrows=10000):
    y = None
    X = None
    if model == "botnet":
        x_feats = [
            "Dur",
            "Proto",
            "SrcAddr",
            "Sport",
            "DstAddr",
            "Dir",
            "Dport",
            "State",
            "sTos",
            "dTos",
            "TotPkts",
            "TotBytes",
            "SrcBytes",
        ]  # removed 'StartTime',
        df = get_botnet_dataframe(nrows=nrows)
        y = df.Label
        X = df[x_feats]
        print(f"{model} is a labeled dataset!")
    if model == "netflow":
        x_feats = [
            "epochtime",
            "duration",
            "srcDevice",
            "dstDevice",
            "protocol",
            "srcPort",
            "dstPort",
            "srcPackets",
            "dstPackets",
            "srcBytes",
            "dstBytes",
        ]  # same as cols
        df = get_netflow_dataframe(nrows=nrows)
        X = df
        print(f"{model} is an unsupervised dataset")
    if model == "host":
        x_feats = [
            "epochtime",
            "eventID",
            "logHost",
            "userName",
            "domainName",
            "logonID",
            "processName",
            "processID",
            "parentProcessName",
            "parentProcessID",
        ]
        # same as cols
        df = get_host_dataframe(nrows=nrows)
        X = df
        print(f"{model} is an unsupervised dataset")

    # X = X.fillna(0)
    # y = y.fillna(0)

    calculate_column_similarity(y, 10)

    X_enc, y_enc, sup_vec, sup_label = process_dirty_dataframes(X, y)

    return df, X_enc, y_enc, sup_vec, sup_label


# USING OPTUNA FOR BETTER MODEL SELECTION
# 1. Define an objective function to be maximized.
def objective(trial, model="botnet"):
    df, X, y, sup_vec, sup_label = get_data(
        model
    )  # ToDo move this outside as it takes a long time....

    # 2. Suggest values for the hyperparameters using a trial object.
    classifier_name = trial.suggest_categorical(
        "classifier", ["SVC", "RandomForest", "HistGradient"]
    )
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    elif classifier_name == "HistGradient":
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.HistGradientBoostingRegressor(
            max_depth=rf_max_depth, n_estimators=10
        )
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestRegressor(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(
        classifier_obj, X, y, n_jobs=-1, cv=5
    )
    accuracy = score.mean()
    return accuracy


# 3. Create a study object and optimize the objective function.
if __name__ == "__main__":

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)
    # print(study.best_trial)

    # to see it explicitly
    # pipeline = make_pipeline(
    # SuperVectorizer(auto_cast=True),
    #    HistGradientBoostingRegressor()
    # )

    df, X_enc, y_enc, sup_vec, sup_label = get_data(model="botnet", nrows=100000)
    labels = df.Label

    # test here is validation set after cross val
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, train_size=0.9, shuffle=False
    )

    # fit_pipeline(pipeline, X_train, y_train)

    all_transformers = sup_vec.transformers_
    feature_names = sup_vec.get_feature_names()

    model = RandomForestRegressor()
    score = sklearn.model_selection.cross_val_score(
        model, X_train, y_train, n_jobs=-1, cv=3
    )
    print(f"Cross Val Score: {score.mean()}")

    print(f"RandomForest-Regressor scored {model.score(X_test, y_test)*100:0.2f}%")

    ## now we can umap it
    reducer = umap.UMAP()
    xys = reducer.fit_transform(X_enc)

    target_names = sup_label.get_feature_names()
    effective_targets = [
        target_names[k] for k in y_enc.values.argmax(1)
    ]  # not ideal, but gets us there

    plot_df = pd.concat(
        [
            pd.DataFrame({"class": effective_targets}),
            pd.DataFrame(xys, columns=["x", "y"]),
        ],
        axis=1,
    )
    plot_df.head()

    sns.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    # Create scatterplot of umap'd points
    sns.lmplot(
        x="x",
        y="y",
        data=plot_df,
        fit_reg=False,
        legend=True,
        height=9,
        hue="class",
        scatter_kws={"s": 200, "alpha": 0.5},
    )

    # from sklearn.preprocessing import LabelEncoder
    #
    # lenc = LabelEncoder()
    # cat_targets = lenc.fit_transform(labels)
    # ## need to throw away samples with only a few labels...or train_test_split chokes
    # ## or just find all the botnet traffic
    # target = labels.apply(lambda x: 1 if 'Botnet' in x else 0)
    #
    # plot_df = pd.concat(
    #     [pd.DataFrame({'class': target}), pd.DataFrame(xys, columns=['x', 'y'])],
    #     axis=1
    # )
    # plot_df.head()
    # # let's just train XGB classifer using encoded features
    # # model = MultiOutputRegressor(XGBClassifier(objective='reg:linear',
    # #     colsample_bytree=0.7, subsample=0.7, max_depth=23, n_estimators=150, n_jobs=28, eval_metric=["error", "auc"],
    # #     feature_names=feature_names
    # # ))
    #
    # model2 = XGBClassifier(colsample_bytree=0.7, subsample=0.7, max_depth=23, n_estimators=150, n_jobs=28, eval_metric=["error", "auc"],
    #     feature_names=['bot', 'not_bot'],
    # )
    #
    # X_train, X_test, y_train, y_test = train_test_split(X_enc, target, train_size=0.8, shuffle=False)
    #
    # model2.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    #
    # model2.get_booster().feature_names = lenc.classes_
    #
    # fig, ax = plt.subplots(figsize=(8, 12))
    # plot_importance(model2, ax=ax, max_num_features=10)
