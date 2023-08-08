from sko.DE import DE
import pandas as pd
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
from scipy.optimize import minimize


def optimal_portfolio(returns):
    # Markovitz
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


class DE_portfolio:
    '''
    min f(x1, ..., x100) = - annualized_IR(x1, ..., x100)
    s.t.
        0.25 <= abs(x1) + ... + abs(x100) <= 1
        -1 <= x1, ..., x100 <= 1
    '''
    def __init__(self, constraint_eq, constraint_ueq):
        self.constraint_eq = constraint_eq
        self.constraint_ueq = constraint_ueq

    @staticmethod
    def obj_func(p):
        ''' IR越大越好
        '''
        daily_returns = pd.read_csv('real_returns_0501.csv')[-20:].reset_index(drop=True).T
        # daily_returns = pd.read_csv('real_returns.csv')[:10].T
        daily_returns['ID'] = daily_returns.index.tolist()
        decisions = pd.DataFrame({'decision': p, 'ID':daily_returns.index.tolist()})    
        portfolio = daily_returns.merge(decisions, on='ID')
        portfolio = portfolio[portfolio.decision != 0]
        weighted_portfolio_return = pd.DataFrame((portfolio.iloc[:, :daily_returns.shape[1]-1].to_numpy().T * portfolio.decision.to_numpy()).T)
        # print(weighted_portfolio_return)
        portfolio_daily_return = weighted_portfolio_return.apply(lambda x: np.log(1 + x)).sum(axis=0)
        # print(weighted_portfolio_return.sum(axis=0))

        ret_T = sum(portfolio_daily_return)
        D = portfolio_daily_return.shape[0]
        var = np.sum([(x - ret_T / D)**2 for x in portfolio_daily_return.tolist()]) / (D-1)
        sdp = np.sqrt(var)
        annualized_IR = (D+1)/D * 12 * ret_T / np.sqrt(252) / sdp
        return -annualized_IR

    def run(self):
        de = DE(func=DE_portfolio.obj_func, n_dim=100, size_pop=2, max_iter=20, lb=[-1]*100, ub=[1]*100,
    constraint_eq=self.constraint_eq, constraint_ueq=self.constraint_ueq)

        
        best_x, best_y = de.run()
        print('best_y:', best_y)

        best20_generation = np.array(de.generation_best_Y).argsort()[:20]

        return best20_generation, de.generation_best_X


def black_litterman(returns):
    n_asset = returns.shape[1]
    cov = np.cov(returns.T)
    volatility = np.sqrt(np.diag(cov))
    risk_aversion_coefficient = 2.5

    # Define the investor views on the asset returns (neutral position)
    view_uncertainty = np.eye(n_asset) * 1e10
    views = np.zeros(n_asset)

    inv_cov = np.linalg.inv(cov)
    # Calculate the implied equilibrium returns
    implied_return = risk_aversion_coefficient * (cov @ np.ones((n_asset, 1)))

    # Calculate the Black-Litterman expected returns
    bl_expected_return = np.linalg.inv(view_uncertainty + inv_cov) @ \
                     (view_uncertainty @ implied_return + (inv_cov @ views)[:, np.newaxis])
    optimal_weights = risk_aversion_coefficient * cov @ bl_expected_return
    # Normalize the weights to sum up to 1
    optimal_weights /= np.sum(optimal_weights)
    return optimal_weights

def risk_parity(return_vec, max_iter=10):
    num_asset = return_vec.shape[1]
    cov = np.cov(return_vec.T)
    volatility = np.sqrt(np.diag(cov))
    inv_cov = np.linalg.inv(cov)

    target_risk_contribution = 1 / num_asset
    weights = np.ones(num_asset) / num_asset
    tolerance = 1e-6

    for i in range(max_iter):
        risk_contribution = np.multiply(weights, volatility) @ inv_cov
        diff_risk_contribution = risk_contribution - target_risk_contribution
        gradient = 2 * inv_cov @ diff_risk_contribution
        step_size = 1 / np.sqrt(np.dot(np.dot(gradient, cov), gradient))
        weights -= step_size * gradient
        if np.max(np.abs(diff_risk_contribution)) < tolerance:
            break

    return weights


def min_variance(asset_returns):
    n_assets = asset_returns.shape[1]
    covariance_matrix = np.cov(asset_returns.T)
    weights = np.linalg.inv(covariance_matrix) @ np.ones((n_assets, 1))
    weights /= np.sum(weights)
    return weights

def monte_carlo(asset_returns, n_scenarios=10):
    n_assets = asset_returns.shape[1]
    cov = np.cov(asset_returns.T)
    # Define the minimum acceptable return for the portfolio
    min_acceptable_return = 0.05

    # Define the maximum acceptable risk (standard deviation) for the portfolio
    max_acceptable_risk = 0.10

    # Calculate the optimal portfolio weights using the mean-variance optimization
    weights = np.linalg.inv(cov) @ np.ones((n_assets, 1))
    weights /= np.sum(weights)

    # Run the Monte Carlo simulation
    portfolio_returns = np.zeros(n_scenarios)
    portfolio_risks = np.zeros(n_scenarios)

    for i in range(n_scenarios):
        # Generate random returns for the portfolio based on the covariance matrix
        random_returns = np.random.multivariate_normal(np.zeros(n_assets), cov)

        # Calculate the return and risk of the portfolio for the current scenario
        portfolio_return = weights.T @ random_returns
        portfolio_risk = np.sqrt(weights.T @ cov @ weights)

        # Store the return and risk values for each scenario
        portfolio_returns[i] = portfolio_return
        portfolio_risks[i] = portfolio_risk

    # Identify the optimal portfolio using the simulated outcomes
    optimal_portfolio_index = np.argmax(portfolio_returns)
    optimal_portfolio_return = portfolio_returns[optimal_portfolio_index]
    optimal_portfolio_risk = portfolio_risks[optimal_portfolio_index]
    optimal_portfolio_weights = weights

    # # Refine the optimal portfolio by adjusting the weights based on the simulated outcomes
    # while (optimal_portfolio_return < min_acceptable_return) or (optimal_portfolio_risk > max_acceptable_risk):
    #     # Generate random returns for the portfolio based on the covariance matrix
    #     random_returns = np.random.multivariate_normal(np.zeros(n_assets), cov)

    #     # Calculate the return and risk of the portfolio for the current scenario
    #     portfolio_return = optimal_portfolio_weights.T @ random_returns
    #     portfolio_risk = np.sqrt(optimal_portfolio_weights.T @ cov @ optimal_portfolio_weights)

    #     # Adjust the weights based on the simulated outcomes
    #     if portfolio_return < min_acceptable_return:
    #         optimal_portfolio_weights *= 0.9
    #     elif portfolio_risk > max_acceptable_risk:
    #         optimal_portfolio_weights *= 1.1

    #     # Normalize the weights to sum up to 1
    #     optimal_portfolio_weights /= np.sum(optimal_portfolio_weights)

    #     # Re-calculate the return and risk of the portfolio with the new weights
    #     optimal_portfolio_return = optimal_portfolio_weights.T @ np.mean(asset_returns, axis=1)
    #     optimal_portfolio_risk = np.sqrt(optimal_portfolio_weights.T @ cov @ optimal_portfolio_weights)
    return optimal_portfolio_weights