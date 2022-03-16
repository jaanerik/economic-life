import numpy as np
import cupy as cp
from util import ou_process
"""Let condition be 1 for matching True, -1 for matching False
and 0 be indifferent (* in the paper). Action=1 is buy."""

# Helper funs
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
lmap = lambda f,l: list(map(f,l))
def create_n_mat(n=3,step=1):
    """Create matrix that has False, False, True, ... , True as first row 
    (for mean and std per each timestep)
    
    Return: n x (n-1) matrix for 2 , ... , n aggregated values.
    
    TODO: step broken, for longer histories it makes sense to look at the avg of last k, then k + step, etc.
    Problem probably arises from the fact that the matrix shape isn't n x n, but n x m with some n != m"""
    b = np.where(np.ones((step*n,step*n)))
    return (b[0] > 1+b[1]).reshape((step*n,step*n))[::,:(step*n)-1:step]

def plot_market(m, from_index = 0, to_index = None, skipstep = 1, price_alpha = 1., richest_alpha = 1.):
    if to_index == None:
        to_index = len(m.buy_history)
    price_history = m.price_history.get()
    df = pd.DataFrame(
        np.array([m.agents_cash.get(),m.agents_stock.get()]).T,
        columns= ['cash','stock']
    )
    df['wealth'] = df.cash + m.price * df.stock
    richest = df.wealth.argmax()
    prange = np.arange(len(price_history)-m.k2-m.k2+m.k)
    when_buy = (prange[lmap(lambda l: richest in l, m.buy_history[m.k2:])])
    when_buy = when_buy[(when_buy >= from_index) & (when_buy <= to_index)]
    when_sell = (prange[lmap(lambda l: richest in l, m.sell_history[m.k2:])])
    when_sell = when_sell[(when_sell >= from_index) & (when_sell <= to_index)]

    #m.k = 7

    fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(12,13))
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"

    # a = axes[0]
    axes[0].plot(
        np.arange(to_index-m.k)[from_index:to_index:skipstep],
        price_history[m.k+from_index:to_index:skipstep],
        label = 'price',
        alpha = price_alpha
    ) #-', '--', '-.', ':', ''
    axes[0].scatter(
        (when_buy-m.k)[3:],
        (np.array(price_history)[when_buy])[3:]#[m.k+from_index:to_index+m.k])
        ,s=30,c='green',marker="^",label = 'richest buy', alpha = richest_alpha)
    axes[0].scatter(
        (when_sell-m.k)[3:],
        (np.array(price_history)[when_sell])[3:],#[m.k+from_index:to_index+m.k])[when_sell],
        s=40,c='red',marker="v",label = 'richest sell', alpha = richest_alpha)
    X = prange[from_index+m.k: to_index: skipstep]
    axes[1].plot(X,np.array(lmap(len, m.buy_history[m.k+from_index:to_index:skipstep])),label = 'buyers')
    axes[1].plot(X,np.array(lmap(len, m.sell_history[m.k+from_index:to_index:skipstep])),label = 'sellers',alpha=0.5)
    # axes[1].plot(m.volume_history[m.k:],'--',label = 'vol')
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='best')
    plt.show()

cdir = 'test_model'
def save_model(m, cdir = cdir): #str((datetime.now())).replace(' ','_')
#     save_series = lambda data, name: pd.Series(data).to_csv(cdir+'/'+name+'.csv',index=False)
#     save_df = lambda data, name: pd.DataFrame(data).to_csv(cdir+'/'+name+'.csv',index=False)
    np.save(cdir+'/price_history',m.price_history.get())
    np.save(cdir+'/volume_history',m.volume_history.get())
    np.save(cdir+'/buy_history', np.array(m.buy_history,dtype=object))
    np.save(cdir+'/sell_history', np.array(m.sell_history,dtype=object))
    np.save(cdir+'/strats',m.strats.get())
    np.save(cdir+'/actions',m.actions.get())
    np.save(cdir+'/agents_cash',m.agents_cash)
    np.save(cdir+'/agents_stock',m.agents_stock)
    print(f"Saved model to dir: {cdir}")

def read_model(cdir = None):
    m = Market()
    m.price_history = cp.array(np.load(cdir+'/price_history.npy'))
    m.volume_history = cp.array(np.load(cdir+'/volume_history.npy'))
    m.buy_history = list(np.load(cdir+'/buy_history.npy',allow_pickle=True))
    m.sell_history = list(np.load(cdir+'/sell_history.npy',allow_pickle=True))
    m.strats = cp.array(np.load(cdir+'/strats.npy'))
    m.actions = cp.array(np.load(cdir+'/actions.npy'))
    m.agents_cash = cp.array(np.load(cdir+'/agents_cash.npy'))
    m.agents_stock = cp.array(np.load(cdir+'/agents_stock.npy'))
    return m

# assert m.price_history.shape == m2.price_history.shape
# assert m.volume_history.shape == m2.volume_history.shape
# assert len(m.buy_history) == len(m2.buy_history)
# assert len(m.sell_history) == len(m2.sell_history)
# assert m.strats.shape == m2.strats.shape
# assert m.actions.shape == m2.actions.shape
# assert m.agents_cash.shape == m2.agents_cash.shape
# assert m.agents_stock.shape == m2.agents_stock.shape

# assert type(m.price_history) == type(m2.price_history)
# assert type(m.volume_history) == type(m2.volume_history)
# assert type(m.buy_history) == type(m2.buy_history)
# assert type(m.sell_history) == type(m2.sell_history)
# assert type(m.strats) == type(m2.strats)
# assert type(m.actions) == type(m2.actions)
# assert type(m.agents_cash) == type(m2.agents_cash)
# assert type(m.agents_stock) == type(m2.agents_stock)

class Market:
    def __init__(
        self, 
        strats_per_agent = 50, 
        len_agents = 100, 
        len_signals = 116, 
        action_ratio = 0.025
    ):
        """Market class that handles all transactions"""
        self.strats = cp.random.choice([1,0,-1],
            size=(len_agents, strats_per_agent, len_signals),
            p=[action_ratio,1-2*action_ratio,action_ratio]
        )
        self.actions = cp.random.choice([1,-1],
            size=(len_agents, strats_per_agent)
        )
        self.len_agents = len_agents
        self.strat_strengths = .1*cp.ones_like(self.actions)
        self.market_state = cp.array([0]*len_signals)
        self.agents_stock = cp.array([50.]*len_agents)
        self.agents_cash = cp.array([10.**3]*len_agents)
        self.dividend = 10/252.
        self.ou_proc = ou_process(mu=.1, r=.2)
        self.r, self.eta, self.c, self.dividend = 0.02/252, 0.005, 0.0000001, 1/252.
        self.price = 50.
        
        self.latest_acting_agent_indeces = None
        self.latest_acting_agent_acts = None
        self.last_price = self.price
        
        self.k2 = 30 #how many timestep info we need max
        self.d = 5 #take 5 timesteps more with each next signal
        self.k = int(self.k2/self.d)
        self.price_history, self.volume_history, self.dividend_history = \
            cp.array([self.price]*self.k2), cp.array([0]*self.k2), cp.array([self.dividend]*self.k2)
        self.buy_history = [-1]*self.k
        self.sell_history = [-1]*self.k
        self.market_state_history = []
        self.sell_strs = []
        self.buy_strs = []
        
        # self.strat_history = []
        self.sell_str_sum_history, self.buy_str_sum_history = [], []

    
    def _get_activated_strats(self):
        """Returns [[i1,j1],[i2,j2],...] where i is agent index and j is strategy index"""
        activations = cp.multiply(self.strats,2*self.market_state-1)
        activated_agent_strats = (activations.min(axis=2) != -1).get() #get as numpy
        del activations
        return cp.array(np.where(activated_agent_strats)).T
    
    def _get_agent_actions(self):
        b = self._get_activated_strats()
        t_strs = cp.array(self.strat_strengths[b.T[0],b.T[1]])
        t_acts = cp.array(self.actions[b.T[0],b.T[1]])
        t_cash = cp.array(self.agents_cash[b.T[0]])
        t_stoc = cp.array(self.agents_stock[b.T[0]])

        """Not enough stock or cash, discard this timestep"""
        keep_indices = cp.where(1-((t_acts > 0)&(t_cash<self.price)|(t_acts<0)&(t_stoc<1.)))
        t_strs = t_strs[keep_indices]
        t_acts = t_acts[keep_indices]
        t_cash = t_cash[keep_indices]
        t_stoc = t_stoc[keep_indices]
        b = b[keep_indices]
        l = b.shape[0]

        cdf = cp.array([b.T[0],b.T[1],t_strs]).T #cp.arange(l),
        random_weights = (cp.random.randn(l)*cdf[:,2]) #NB: Maybe use [0,1] uniform here? This is important
        random_weights = (random_weights-random_weights.min()+0.0001)/random_weights.max()
        cp.random.shuffle(cdf)
        cdf = cdf[(cdf[:,0]+random_weights).argsort()][:,:2]
        self.buy_strs.append(t_strs[t_acts > 0].sum().get())
        self.sell_strs.append(t_strs[t_acts < 0].sum().get())
        del keep_indices,t_strs,t_acts,t_cash,t_stoc
        it = cdf[cp.r_[cp.array([True]), cp.diff(cdf[:,0])] == 1].T.astype(int) 
        #it = agents, strats
        return it[0], self.actions[it[0],it[1]]
    
    def fundamental_value(self):
        """p[t] == dividend/risk_free_rate"""
        return self.dividend/(252.*self.r)
    
    
    ########
    def _transact(self):
        """Collect interest, dividend. Exchange stock/cash. Set new price.
        
        Be sure _set_action_0_if_negative_wealth_or_stock is run before.
        """
        acts = self.latest_acting_agent_acts
        agents = self.latest_acting_agent_indeces
        B, O = cp.sum(acts > 0).get(), cp.sum(acts < 0).get()
        V = min(B,O)
        
        buy_agent_indeces = agents[acts > 0]
        sell_agent_indeces = agents[acts < 0]
        self.buy_history.append(agents[acts > 0].get())
        self.sell_history.append(agents[acts < 0].get())
        self.price_history = cp.append(self.price_history,self.price)
        self.volume_history = cp.append(self.volume_history,V)
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
        self.dividend = next(self.ou_proc)/252.
        
    def _update_market_state(self):
        """Vectorised aggregated function and fundamental signals."""
        a = np.vstack([(np.array(self.price_history.get())[-self.k2:])]*(self.k2-1)).T
        a2 = np.vstack([(np.array(self.volume_history.get())[-self.k2:])]*(self.k2-1)).T

        b = np.ma.array(a,mask=create_n_mat(self.k2))[:,::self.d] #price_history
        c = np.ma.array(a2,mask=create_n_mat(self.k2))[:,::self.d] #vol_history

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

#         self.market_state_history.append(self.market_state.get())
        self.market_state = cp.array(np.r_[mean_signals,stdev_signals,meanvol_signals,fvals])
        
    def _feedback_strats(self):
        """Each strat is updated by the potential profit it turns * constant self.c."""
        dp = self.price - self.last_price
        b = self._get_activated_strats()
        self.strat_strengths[b.T[0], b.T[1]] += self.c * dp * self.actions[b.T[0], b.T[1]]
        
    def _mutate_strats(self):
        """Currently in its simplest form: 
        If weakest strat has negative strength, invert action and strength.
        
        Otherwise change one index."""
        agent_indeces = cp.arange(self.len_agents)
        strat_indeces = self.strat_strengths.argmin(axis=1)
        b = cp.where(self.strats[agent_indeces,strat_indeces] != 0)

        cdf = cp.array([b[0],b[1]]).T#,columns=['strat_ind','strat_cond'])
        cp.random.shuffle(cdf)
        cdf = cdf[(cdf[:,0]).argsort()]
        si,sb = cdf.T
        
        change_strats = cp.abs(self.strats[agent_indeces,strat_indeces][si]).sum(axis=1)
        inv_bools = change_strats<cp.random.choice(np.arange(10,20),size=change_strats.shape[0])

        a,b = si[inv_bools],sb[inv_bools]
        a2,b2 = si[~inv_bools],sb[~inv_bools]

        ag,st = np.where(self.strat_strengths < -0.25)
        self.actions[ag,st] *= -1 #invert superweak strat
        self.strat_strengths[ag,st] *= -1

        reverse_bools = self.strat_strengths < -0.4
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
