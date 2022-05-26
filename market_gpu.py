import numpy as np
import cupy as cp
from util import *


class Market:
    """Market class for agents to trade with different strategies.
    ...
    Attributes
    ----------
    strats_per_agent : int
        number of different strategies per each agent
    len_agents : int
        number of agents in the market
    len_signals : int
        number of signals in the market state (will be automatic in the future)
    action_ratio : float
        weight of BUY and SELL in strategy choice
    dividend_mode : str
        - ou_process: Ornstein-Uhlenbeck Process
        - binary: Alternating 0, 200 dividend

    Methods
    -------
    simulate_k_step(k=1)
        Simulate the market with trades, feedbacking and mutating strategies.
    """

    def __init__(
        self,
        strats_per_agent=50,
        len_agents=100,
        len_signals=121,
        action_ratio=0.025,
        dividend_mode='ou_process',
    ):
        self.strats = cp.random.choice(
            [1, 0, -1],
            size=(len_agents,
                  strats_per_agent, len_signals),
            p=[action_ratio, 1-2 *
               action_ratio, action_ratio]
        )
        self.actions = cp.random.choice(
            [1, -1],
            size=(len_agents, strats_per_agent)
        )
        self.len_agents = len_agents
        self.strat_strengths = .1*cp.ones_like(self.actions)
        self.market_state = cp.array([0]*len_signals)
        self.agents_stock = cp.array([50.]*len_agents)
        self.agents_cash = cp.array([10.**3]*len_agents)
        self.dividend = 10/252.
        if dividend_mode == 'ou_process':
            self.ou_proc = ou_process(mu=.1, r=.2)
        elif dividend_mode == 'binary':
            self.ou_proc = boolean_gen()
        self.r, self.eta = 0.02/252, 0.001
        self.c, self.dividend = 0.0001, 1/252.
        self.price = 50.

        self.acting_agent_indeces = None
        self.acting_agent_acts = None
        self.last_price = self.price

        self.k2 = 30  # how many timestep info we need max
        self.d = 5  # take 5 timesteps more with each next signal
        self.k = int(self.k2/self.d)
        self.price_history = cp.array([self.price]*self.k2)
        self.volume_history = cp.array([0] * self.k2)
        self.dividend_history = cp.array([self.dividend]*self.k2)
        self.buy_history = [-1]*self.k
        self.sell_history = [-1]*self.k
        self.market_state_history = []
        self.sell_strs = []
        self.buy_strs = []
        
        self.acting_agent_strengths = None
        self.acting_agent_wealth = None

        # self.strat_history = []
        self.ignore_agents = cp.array([])
        self.sell_str_sum_history, self.buy_str_sum_history = [], []

    def _get_activated_strats(self):
        """Returns [[i1,j1],[i2,j2],...]
        where i is agent index and j is strategy index"""
        activations = cp.multiply(self.strats, 2*self.market_state-1)
        activated_agent_strats = (activations.min(
            axis=2) != -1).get()  # get as numpy
        del activations
        return cp.array(np.where(activated_agent_strats)).T

    def _get_agent_actions(self):
        """Returns acting_agent_indeces, acting_agent_acts"""
        b = self._get_activated_strats()
        t_strs = cp.array(self.strat_strengths[b.T[0], b.T[1]])
        t_acts = cp.array(self.actions[b.T[0], b.T[1]])
        t_cash = cp.array(self.agents_cash[b.T[0]])
        t_stoc = cp.array(self.agents_stock[b.T[0]])

        """Not enough stock or cash, discard this timestep"""
        keep_indices = cp.where(
            1-((t_acts > 0) & (t_cash < self.price) | (t_acts < 0) & (t_stoc < 1.)))
        t_strs = t_strs[keep_indices]
        t_acts = t_acts[keep_indices]
        t_cash = t_cash[keep_indices]
        t_stoc = t_stoc[keep_indices]
        b = b[keep_indices]
        l = b.shape[0]

        cdf = cp.array([b.T[0], b.T[1], t_strs]).T  # cp.arange(l),
        random_weights = (cp.random.randn(l)*cdf[:, 2])
        random_weights = (random_weights-random_weights.min() +
                          0.0001)/random_weights.max()
        cp.random.shuffle(cdf)
        cdf = cdf[(cdf[:, 0]+random_weights).argsort()][:, :2]
        self.buy_strs.append(t_strs[t_acts > 0].sum().get())
        self.sell_strs.append(t_strs[t_acts < 0].sum().get())
        del keep_indices, t_strs, t_acts, t_cash, t_stoc
        it = cdf[cp.r_[cp.array([True]), cp.diff(
            cdf[:, 0])] == 1].T.astype(int)
        # Following line needs to include agent indeces in the
        # future, if not acting is a choice also
        self.acting_agent_strengths = self.strat_strengths[it[0], it[1]]
        self.acting_agent_wealth = self.agents_cash[it[0]] + self.price * self.agents_stock[it[0]]
        self.acting_agent_strat_tuple = it[0], it[1]
        return it[0], self.actions[it[0], it[1]]

    def fundamental_value(self):
        """p[t] == dividend/risk_free_rate"""
        return self.dividend/(252.*self.r)

    def _transact(self):
        """Collect interest, dividend. Exchange stock/cash. Set new price."""
        acts = self.acting_agent_acts
        agents = self.acting_agent_indeces
        B, O = cp.sum(acts > 0).get(), cp.sum(acts < 0).get()
        V = min(B, O)

        buy_agent_indeces = agents[acts > 0]
        sell_agent_indeces = agents[acts < 0]
        self.buy_history.append(agents[acts > 0].get())
        self.sell_history.append(agents[acts < 0].get())
        self.price_history = cp.append(self.price_history, self.price)
        self.volume_history = cp.append(self.volume_history, V)
        self.agents_cash *= 1+self.r
        self.agents_cash += self.agents_stock*self.dividend

        if V > 0:
            self.agents_stock[sell_agent_indeces] -= V/O
            self.agents_cash[sell_agent_indeces] += V/O*self.price
            self.agents_stock[buy_agent_indeces] += V/B
            self.agents_cash[buy_agent_indeces] -= V/B*self.price
            self.price *= 1 + max(self.eta*(B-O), -0.9)

    def _refresh_params(self):
        """Note that in the future the following is not deterministic."""
        self.acting_agent_indeces, self.acting_agent_acts = \
            self._get_agent_actions()
        self.last_price = self.price
        self.dividend = next(self.ou_proc)/252.

    def _update_market_state(self):
        """Vectorised aggregated function and fundamental signals."""
        a = np.vstack(
            [(np.array(self.price_history.get())[-self.k2:])]*(self.k2-1)).T
        a2 = np.vstack(
            [(np.array(self.volume_history.get())[-self.k2:])]*(self.k2-1)).T

        b = np.ma.array(a, mask=create_n_mat(self.k2))[
            :, ::self.d]  # price_history
        c = np.ma.array(a2, mask=create_n_mat(self.k2))[
            :, ::self.d]  # vol_history

        means = np.ma.mean(b, axis=0).data
        stdevs = np.ma.std(b, axis=0).data
        volmeans = np.ma.mean(c, axis=0).data

        mean_signals = np.greater(
            np.vstack([means]*self.k),
            (np.linspace(0.5, 1.5, self.k)*self.price)[:, None]
        ).flatten('F')
        stdev_signals = np.greater(
            np.vstack([stdevs]*self.k),
            (np.linspace(0.001, 1, self.k)*self.price)[:, None]
        ).flatten('F')
        meanvol_signals = np.greater(
            np.vstack([volmeans]*self.k),
            (np.linspace(1, 50, self.k))[:, None]
        ).flatten('F')
        fvals = self.price > np.round(np.linspace(
            0.25, 4.25, 8), 1)*self.fundamental_value()
        rel_means = (means[:-1]/means[1:]) >= 1.
        self.market_state_history.append(self.market_state.get())
        self.market_state = cp.array(
            np.r_[mean_signals, stdev_signals, meanvol_signals, fvals, rel_means])

    def _feedback_strats(self):
        """Each strat is updated by the potential profit it turns * constant self.c."""
        dp = self.price - self.last_price
        b = self._get_activated_strats()
        self.strat_strengths[b.T[0], b.T[1]] += self.c * \
            dp * self.actions[b.T[0], b.T[1]]

    def _ignore_agents(self, a, b, ignore_agents):
        """Ignore agents in a and also exclude same indeces in b"""
        ignore_inds = cp.isin(a, ignore_agents)
        return a[~ignore_inds], b[~ignore_inds]

    def _mutate_strats(self):
        """ If weakest strat has negative strength, invert action and strength.
        Otherwise change one index.
        Also set self.ignore_agents to manually mutate them."""
        agent_indeces = cp.arange(self.len_agents)
        strat_indeces = self.strat_strengths.argmin(axis=1)
        b = cp.where(self.strats[agent_indeces, strat_indeces] != 0)

        cdf = cp.array([b[0], b[1]]).T  # ,columns=['strat_ind','strat_cond'])
        cp.random.shuffle(cdf)
        cdf = cdf[(cdf[:, 0]).argsort()]
        si, sb = cdf.T

        change_strats = cp.abs(
            self.strats[agent_indeces, strat_indeces][si]).sum(axis=1)
        inv_bools = change_strats < cp.random.choice(
            np.arange(10, 20), size=change_strats.shape[0])

        a, b = si[inv_bools], sb[inv_bools]
        a2, b2 = si[~inv_bools], sb[~inv_bools]

        ag, st = np.where(self.strat_strengths < -0.25)
        # invert superweak strat
        self.actions[self._ignore_agents(ag, st, self.ignore_agents)] *= -1
        self.strat_strengths[self._ignore_agents(
            ag, st, self.ignore_agents)] *= -1

        self.strats[self._ignore_agents(
            agent_indeces[a], strat_indeces[b], self.ignore_agents)] *= -1  # mutate
        self.strats[self._ignore_agents(
            agent_indeces[a2], strat_indeces[b2], self.ignore_agents)] *= 0  # generalise

    def simulate_k_step(self, k=1, refresh_params_callback=None):
        """Simulate k timesteps and f is callback function if needed. Useful while developing mostly."""
        for i in range(k):
            """Note that order TBD"""
            # self.strat_history.append(np.array(self.strats).copy())
            self.buy_str_sum_history.append(np.sum((self.actions > 0)*self.strat_strengths))
            self.sell_str_sum_history.append(np.sum((self.actions < 0)*self.strat_strengths))
            self._refresh_params()  # sets last price
            if refresh_params_callback != None: refresh_params_callback(self)
            self._transact()  # sets new price
            self._feedback_strats()
            if i % 10 == 0:
                self._mutate_strats()
            self._update_market_state()
            if np.all(np.diff(self.price_history[-100:]) == 0.) and \
                    len(self.price_history) > 100:
                print("Ending")
                break
