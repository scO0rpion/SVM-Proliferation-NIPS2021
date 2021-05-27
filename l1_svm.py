import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import functools
from tqdm import tqdm
from itertools import product
from datetime import datetime
import sys
import cvxopt as cvx

# GLobal Variables
n_sim = 400
N = range(10, 70, 1)
D = range(80, 1000, 5)
parallel = Parallel(n_jobs=4, backend="loky")


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


@replicate(n_sim, par=False)
def generate_sol(n_sample, dimension, distribution="Gaussian"):
    observed = random_generator(distribution)(n_sample, dimension) / np.sqrt(dimension)
    c = cvx.matrix(np.vstack([-1.0] * n_sample))  # minus is to account for maximization
    h = cvx.matrix(np.vstack([1.0] * 2 * dimension))
    B = cvx.matrix(np.hstack((observed, -observed)).T)
    lp = cvx.solvers.lp(c, B, h, solver="cvxopt_glpk", options={'show_progress': False})
    if lp['status'] != 'optimal':
        print("Optimizer did not converge \n")

    flag = np.all(np.asarray(lp['x']) > 0)
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

@replicate(10)
def test():
    return random_generator("Gaussian")

if __name__ == "__main__":

    suite = ["Uniform", "Gaussian", "GaussianBiased", "Bernoulli", "Laplacian", "Radamacher"]
    suite = ["Gaussian"]
    df = pd.DataFrame(data=product(N, D, suite), columns=["NSample", "Dimension", "Distribution"])
    df["prob"] = 0.0

    for index, row in tqdm(df.iterrows(), desc="Monte Carlo Iteration"):
        # Surprisingly parallelizing the for loop takes longer to run the job.
        df.loc[index, "prob"] = sum(
            generate_sol(row["NSample"], row["Dimension"], distribution= row["Distribution"])
        )/ float(n_sim)

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    # Save results
    df.to_csv(f"L1SVMProbs-{n_sim}-{date}.csv", index=False)
