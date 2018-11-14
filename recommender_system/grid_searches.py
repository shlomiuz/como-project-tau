from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn_wrapper import MC


pairs = """members - item pairs"""
ratings = """ratings"""
n_u = """number of members"""
n_m = """number of items"""


# Grid Search with Batch GD
params_batch = {"lmbda": (45, 50, 55),
                "n_factors": (15, 18, 21)}

mc_batch = MC(n_u=n_u, n_m=n_m, gamma=6e-5, n_epochs=400, solver="batch_gd")
grid_batch = GridSearchCV(mc_batch, param_grid=params_batch, cv=4)
grid_batch.fit(pairs, ratings)

best_batch = grid_batch.best_estimator_
results_batch = pd.DataFrame(grid_batch.cv_results_)
results_batch[["mean_test_score", "std_test_score", "params"]]\
    .sort_values(by=["mean_test_score"], ascending=True).head()


# Grid Search with Stochastic GD
params_stochastic = {"lmbda": (0.25, 0.5, 0.75),
                     "n_factors": (18, 20, 22)}

mc_stochastic = MC(n_u=n_u, n_m=n_m, gamma=0.01, n_epochs=50, solver="sgd")
grid_stochastic = GridSearchCV(mc_stochastic, param_grid=params_stochastic, cv=4)
grid_stochastic.fit(pairs, ratings)

best_stochastic = grid_stochastic.best_estimator_
results_stochastic = pd.DataFrame(grid_stochastic.cv_results_)
results_stochastic[["mean_test_score", "std_test_score", "params"]]\
    .sort_values(by=["mean_test_score"], ascending=True).head()
