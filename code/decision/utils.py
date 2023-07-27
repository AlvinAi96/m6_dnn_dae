from sko.DE import DE
import pandas as pd
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers

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

        return best20_generation

    
