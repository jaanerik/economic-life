import numpy as np
import pandas as pd

"""Let condition be 1 for matching True, -1 for matching False
and 0 be indifferent (* in the paper). Action=1 is buy."""

# Helper funs
def create_n_mat(n=3,step=1):
    """Create matrix that has False, False, True, ... , True as first row 
    (for mean and std per each timestep)
    
    Return: n x (n-1) matrix for 2 , ... , n aggregated values.
    
    TODO: step broken, for longer histories it makes sense to look at the avg of last k, then k + step, etc.
    Problem probably arises from the fact that the matrix shape isn't n x n, but n x m with some n != m"""
    b = np.where(np.ones((step*n,step*n)))
    return (b[0] > 1+b[1]).reshape((step*n,step*n))[::,:(step*n)-1:step]

lmap = lambda f,l: list(map(f,l))

class Market:
    def __init__(self, strats_per_agent = 50, len_agents = 100, len_signals = 98, action_ratio = 0.1):
        """Market class that handles all transactions"""
#         create_condition = lambda: np.random.choice([1,0,-1], size=len_signals,p=[0.1,0.8,0.1])
        create_condition = lambda: np.random.choice([1,0,-1], size=len_signals,p=[action_ratio,1-2*action_ratio,action_ratio])
        create_action = lambda: np.random.choice([1,-1])
        
        self.strats = np.array([
            [create_condition() for _ in range(strats_per_agent)] 
            for _ in range(len_agents)
        ])
        self.actions = np.array([
            [create_action() for _ in range(strats_per_agent)] 
            for _ in range(len_agents)
        ])
        self.len_agents = len_agents
        self.strat_strengths = .1*np.ones_like(self.actions)
        self.market_state = np.array([0]*len_signals)
        self.agents_stock = np.array([50.]*len_agents)
        self.agents_cash = np.array([10.**3]*len_agents)
        self.dividend = 20/252.
        self.r, self.eta, self.c, self.dividend = 0.02/252, 0.01, 0.001, 1/252.
        self.price = self.fundamental_value()
        
        self.latest_acting_agent_indeces = None
        self.latest_acting_agent_acts = None
        self.last_price = self.price
        
        self.k = 7 #how many timestep info we need max
        self.price_history, self.volume_history, self.dividend_history = \
            [self.price]*self.k, [0]*self.k, [self.dividend]*self.k
        self.buy_history, self.sell_history = [None]*self.k, [None]*self.k
        self.market_state_history = []
        
        # self.strat_history = []
        self.sell_str_sum_history, self.buy_str_sum_history = [], []

    
    def _get_activated_strats(self):
        """Returns [[i1,j1],[i2,j2],...] where i is agent index and j is strategy index"""
        activations = 2*np.multiply(self.strats,self.market_state-0.5)
        activated_agent_strats = np.array(lmap(lambda b: lmap(lambda a: -1 not in a, b), activations))
        return np.transpose(np.where(activated_agent_strats))

#   def _get_activated_strats(self):
#       """Returns [[i1,j1],[i2,j2],...] where i is agent index and j is strategy index"""
#       activations = cp.multiply(self.strats,2*self.market_state-1)
#       activated_agent_strats = (activations.min(axis=2) != -1).get() #get as numpy
#       del activations
#       return np.transpose(np.where(activated_agent_strats))
    
    def fundamental_value(self):
        """p[t] == dividend/risk_free_rate"""
        return self.dividend/(self.r)
    
    def _get_agent_actions(self):
        """Returns [[i1,j1],[i2,j2],...] where i is agent index and j is action.
        
        Get all activated strats, discard undoable strats, choose action rand with strat weight."""
        #TODO: Change to random activated strat, not first. Weighted by strength random.
        b = self._get_activated_strats()
        
        t_strs = self.strat_strengths[b.T[0],b.T[1]]
        t_acts = self.actions[b.T[0],b.T[1]]
        t_cash = self.agents_cash[b.T[0]]
        t_stoc = self.agents_stock[b.T[0]]

        """Not enough stock or cash, discard this timestep"""
        keep_indices = np.where(1-((t_acts > 0)&(t_cash<self.price)|(t_acts<0)&(t_stoc<1.)))
        
        t_strs = t_strs[keep_indices]
        t_acts = t_acts[keep_indices]
        t_cash = t_cash[keep_indices]
        t_stoc = t_stoc[keep_indices]
        b = b[keep_indices]
        
        df = pd.DataFrame(np.array([b.T[0],b.T[1],t_strs]).T,columns=['agent','strat','str'])
        #df = df[df.str > 0]
        if df.shape[0] == 0:
            return np.array([]), np.array([])
        df = df.reset_index()
        df.str = df.str+np.abs(df.str.min())+0.01 #any strat has some chance of getting selected now
        norms = dict(df.groupby(by='agent').str.sum())
        df = df.groupby(by='agent').sample(weights=((df.str>0) * df.str)/df.agent.map(lambda a: norms[a]))
        ndf = df[['agent','strat']].to_numpy(dtype=int).T
        
        agents,strats = ndf[0], ndf[1]
        #print(np.unique(self.actions[agents, strats],return_counts=True))
        return agents, self.actions[agents, strats]
    
    def _transact(self):
        """Collect interest, dividend. Exchange stock/cash. Set new price.
        
        Be sure _set_action_0_if_negative_wealth_or_stock is run before.
        """
        acts = self.latest_acting_agent_acts
        agents = self.latest_acting_agent_indeces
        B, O = np.sum(acts > 0), np.sum(acts < 0)
        V = min(B,O)
        
        buy_agent_indeces = agents[acts > 0]
        sell_agent_indeces = agents[acts < 0]
        self.buy_history.append(agents[acts > 0])
        self.sell_history.append(agents[acts < 0])
        self.price_history.append(self.price)
        self.volume_history.append(V)
        
        self.agents_cash *= 1+self.r
        self.agents_cash += self.agents_stock*self.dividend
        
        if V > 0:
            self.agents_stock[sell_agent_indeces] -= V/O
            self.agents_cash[sell_agent_indeces] += V/O*self.price
            self.agents_stock[buy_agent_indeces] += V/B
            self.agents_cash[buy_agent_indeces] -= V/B*self.price
            self.price *= 1 + max(self.eta*(B-O),-0.9)
        
    def _refresh_params(self):
        """Note that in the future the following is not deterministic."""
        self.latest_acting_agent_indeces, self.latest_acting_agent_acts = self._get_agent_actions()
        
        self.last_price = self.price
        self.dividend = self.dividend
        
    def _update_market_state(self):
        """Vectorised aggregated function and fundamental signals."""
        a = np.vstack([(np.array(self.price_history)[-self.k:])]*(self.k-1)).T
        a2 = np.vstack([(np.array(self.volume_history)[-self.k:])]*(self.k-1)).T

        b = np.ma.array(a,mask=create_n_mat(self.k)) #price_history
        c = np.ma.array(a2,mask=create_n_mat(self.k)) #vol_history

        means = np.ma.mean(b,axis=0).data
        stdevs = np.ma.std(b,axis=0).data
        volmeans = np.ma.mean(c,axis=0).data

        mean_signals = np.greater(
            np.vstack([means]*self.k), 
            (np.linspace(0.5,1.5,self.k)*self.price)[:,None]
        ).flatten('F')
        stdev_signals = np.greater(
            np.vstack([stdevs]*self.k), 
            (np.linspace(0.001,1,self.k)*self.price)[:,None]
        ).flatten('F')
        meanvol_signals = np.greater(
            np.vstack([volmeans]*self.k), 
            (np.linspace(1,50,self.k))[:,None]
        ).flatten('F')
        fvals =self.price>np.round(np.linspace(0.25,4.25,8),1)*self.fundamental_value()

        np.r_[mean_signals,stdev_signals,meanvol_signals,fvals]
        
    def _feedback_strats(self):
        """Each strat is updated by the potential profit it turns * constant self.c."""
        dp = self.price - self.last_price
        b =self._get_activated_strats()
        self.strat_strengths[b.T[0], b.T[1]] += self.c * dp * self.actions[b.T[0], b.T[1]]
        
    def _mutate_strats(self):
        """Currently in its simplest form: 
        If weakest strat has negative strength, invert action and strength.
        
        Otherwise change one index."""
        agent_indeces = np.arange(self.len_agents)
        strat_indeces = self.strat_strengths.argmin(axis=1)
        b = np.where(self.strats[agent_indeces,strat_indeces] != 0)

        df = pd.DataFrame(np.array([b[0],b[1]]).T,columns=['strat_ind','strat_cond'])
        df = df.groupby(by='strat_ind').sample()
        si,sb = df.to_numpy().T
        
        change_strats = np.abs(self.strats[agent_indeces,strat_indeces][si]).sum(axis=1)
        inv_bools = change_strats<np.random.choice(np.arange(10,20),size=len(change_strats))

        a,b = si[inv_bools],sb[inv_bools]
        a2,b2 = si[~inv_bools],sb[~inv_bools]
        self.strats[agent_indeces[a],strat_indeces[b]] *= -1 #mutate
        self.strats[agent_indeces[a2],strat_indeces[b2]] *= 0 #generalise
        
    def simulate_k_step(self, k = 1):
        """Definitely cannot be vectorised"""
        for _ in range(k):
            """Note that order TBD"""
            # self.strat_history.append(np.array(self.strats).copy())
            a = np.sum((self.actions<0)*self.strat_strengths)
            b = np.sum((self.actions>0)*self.strat_strengths)
            self.buy_str_sum_history.append(b)
            self.sell_str_sum_history.append(a)
            self._refresh_params() #sets last price
            self._transact() #sets new price
            self._feedback_strats()
            self._mutate_strats()
            self._update_market_state()
            if np.all(np.diff(self.price_history[-100:]) == 0.) and len(self.price_history) > 100:
                print("Ending")
                break
