import numpy as np
from sklearn.svm import SVC, LinearSVC
import cvxopt as cvx
from joblib import Parallel, delayed
import pandas as pd
import functools
from tqdm import tqdm
from itertools import product
from datetime import datetime
import sys


def replicate(n_times, par = True):
    global parallel
    def inner_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            f = functools.partial(func, *args, **kwargs)
            if par:
                return parallel(delayed(f)() for _ in range(n_times))
            else:
                return (f() for _ in range(n_times))

        return wrapper

    return inner_func


@replicate(n_sim, par=par)
def generate_sol_sklearn(n_sample, dimension, distribution="Gaussian", classifier=None):
    classifier = classifier or LinearSVC(**kwargs)
    observed = random_generator(distribution)(n_sample, dimension) / np.sqrt(dimension)
    labels = np.repeat([-1, 1], int((n_sample + 1) / 2))[:n_sample]  # drop last one if necessary
    classifier.fit(observed, labels)
    flag = np.allclose(classifier.decision_function(observed), labels, atol=1e-3)
    return flag


@replicate(n_sim, par=par)
def generate_sol_cvx(n_sample, dimension, distribution = "Gaussian"):
    observed = random_generator(distribution)(n_sample, dimension) / np.sqrt(dimension)
    labels = np.repeat([-1, 1], int((n_sample + 1) / 2))[:n_sample, None]  # drop last one if necessary
    inputs = observed * labels
    q = cvx.matrix( np.vstack([-1.0] * n_sample) )
    P = cvx.matrix( np.dot(inputs, inputs.T) )
    qp = cvx.solvers.qp(P, q, solver=None, options={'show_progress': False})
    flag = np.all( np.asarray(qp['x']) > 0 )
    return flag

def random_generator(distribution, state = None):
    rng = np.random.RandomState(state)
    suite = {"Uniform": lambda *shape: rng.uniform(low=-1, high=1, size=shape),
             "Gaussian": rng.randn,
             "GaussianBiased": lambda *shape: rng.randn(*shape) + 1.0,
             "Bernoulli": lambda *shape: rng.binomial(n=1, p=0.5, size=shape),
             "Laplacian": lambda *shape: rng.laplace(size=shape),
             "Radamacher": lambda *shape: 2 * (rng.binomial(n=1, p=0.5, size=shape) - 0.5)
             }
    return suite[distribution]


if __name__ == "__main__":
    
    # GLobal Variables
    kwargs = {'fit_intercept': False, 'dual': True, 'C': 1e8, 'tol': 1e-6, 'max_iter': 5000}
    method = "CVX"
    n_sim = 400
    N = range(40, 100, 2)
    D = range(100, 1000, 10)
    clf = LinearSVC(**kwargs)
    num_cores = 1 if sys.argv[-1] is '' else int(sys.argv[-1])
    par = num_cores > 1
    parallel = Parallel(n_jobs=num_cores, backend="loky")   
    suite = ["Uniform", "Gaussian", "GaussianBiased", "Bernoulli", "Laplacian", "Radamacher"]

    df = pd.DataFrame(data=product(N, D, suite), columns=["NSample", "Dimension", "Distribution"])
    df["prob"] = 0.0

    for index, row in tqdm(df.iterrows(), desc="Monte Carlo Iteration"):
        df.loc[index, "prob"] = sum( generate_sol_cvx(row["NSample"],
                                                          row["Dimension"],
                                                          distribution=row["Distribution"])
                                    ) / float(n_sim)

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    # Save results
    df.to_csv(f"SVMProbs-{method}-{n_sim}-{date}.csv", index=False)
