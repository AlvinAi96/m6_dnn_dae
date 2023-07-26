from utils import optimal_portfolio, DE_portfolio

class Test:
    def __init__(self):
        self.freq = '1min'
        return

    def preprocess_data(self):
        return data


    def run(self):
        weights, returns, risks = optimal_portfolio(return_vec.to_numpy())

        de = DE_portfolio(constraint_eq=[], 
                            constraint_ueq=[
                                            # 小于等于0
                                            lambda x: sum([abs(i) for i in x]) - 1,
                                            lambda x: 0.25 - sum([abs(i) for i in x])
                                        ])

        weights = de.run()


