from strategy import *
from uuid import uuid4 as UUID
import numpy as np
from copy import deepcopy as copy
from util import ou_process

lmap = lambda f,l: list(map(f,l))
nmap = lambda f,l: np.array(list(map(f,l)))
lfilter = lambda f,l: list(filter(f,l))
get_bought_sold_ts = lambda action_df: (total_bought(get_actions(action_df)),total_sold(get_actions(action_df)),get_ts(action_df))
total_bought = lambda actions: np.sum(lmap(lambda d: d['amount'], filter(lambda d: d['type'] == BUY, actions)))
total_sold = lambda actions: np.sum(lmap(lambda d: d['amount'], filter(lambda d: d['type'] == SELL, actions)))
get_ts = lambda action_df: action_df[0]['timestamp']
get_actions = lambda agents: lfilter(lambda o: o != None, lmap(lambda a: a.get('action'), agents))
get_strat_strength = lambda s: s.strength
get_strats = lambda agents: np.squeeze(np.reshape([agent['strats'] for agent in agents], (1,-1)))

SELL, BUY = -1, 1

def clear(agent: dict, verbose: bool = False) -> dict:
        """Transacts each agent's action and sets new action to None

        Order is important:
            1) cash earns r
            2) stock price change changes wealth
            3) stock is bought/sold (liquidity changes inherently)

        Aka Clearing House function
        """
        agent['wealth'] += dividend * agent['stock']/stock_total

        if agent.get('action') == None or B == 0 or O == 0:
            return agent

        stock_diff = (agent['action']['type']==BUY) * V/B * agent['action']['amount'] -\
            (agent['action']['type']==SELL) * V/O * agent['action']['amount']
        agent['timestamp'] = ts
        agent['wealth'] += calculate_agent_cash(agent) * r #cash earned risk free rate
        agent['wealth'] += agent['stock'] * (price-price_history[-1]) # how much stock worth changed.
        agent['stock'] += stock_diff #buy/sell happens at new price
        if agent['stock'] < 0. or agent['wealth'] < 0.:
            print(ts, agent['wealth'], agent['stock'])
            assert False
        if verbose:
            print("Agent aquired dividend wealth: %s" % (dividend * agent['stock']/stock_total))
            print("Stock diff is %s" % stock_diff)
        agent.pop('action')
        return agent

def create_agent():
    name = str(UUID())
    return {
        'name': name,
        'strats': [Strategy(agent=name) for _ in range(60)],
        'wealth': 10.**4,
        'stock': 100.,
        'action': {'type': np.random.choice([BUY,SELL]), 'amount': 1}
    }

class Market:
    """Handles all variables normally. Not having a class made 
    it a nightmare to keep global variables. Also this is a bad
    class, because it handles both market logic and strat feedback.
    """
    
    def __init__(self,no_agents=150, mutate_prob=0.02):        
        self.r = 0.02/252
        self.dividend = 1./252.
        self.price = 10.
        self.c = 0.001
        self.s_min,self.s_max=-1,1
        self.eta = 0.001
        self.t,self.x = ou_process(mu = self.dividend, dt=10**(-6),r=3)
        self.dividend_with_timestamp = iter(zip(self.x,self.t))
        self.signals = []
        self.mutate_prob = mutate_prob

        self.price_history = [self.price,self.price]
        self.volume_history = [1,1]
        self.dividend_history = [self.dividend,self.dividend]
        self.action_history = []
        self.market_state_history = []
        self.ts_history = [] #tmp dealing with some bug
        self.mutate_history = []

        self.no_agents = no_agents
        self.agents = [create_agent() for _ in range(self.no_agents)]
        self.stock_total = self.no_agents*10.
        self.market_state = []
        self.B,self.O,self.V = 0.,0.,0.
        
        self.create_signals()
        self.update_market_state()

    def update_market_state(self):
        self.market_state = np.array([signal_dict['signal']() for signal_dict in self.signals])
        
    def feedback_all(self):
        for agent in self.agents:
            for strat in agent['strats']:
                self.feedback(strat)
                
    def clear(self, agent: dict, verbose: bool = False) -> dict:
        """Transacts each agent's action and sets new action to None

        Order is important:
            1) cash earns r
            2) stock price change changes wealth
            3) stock is bought/sold (liquidity changes inherently)

        Aka Clearing House function
        """
        def calculate_agent_cash(agent: dict) -> float:
                """Return the liquidity of an agent. How much stock can be bought."""
                return agent['wealth'] - agent['stock'] * self.price

        agent['wealth'] += self.dividend * agent['stock']/self.stock_total

        if agent.get('action') == None or self.B == 0 or self.O == 0:
            return agent

        stock_diff = (agent['action']['type']==BUY) * self.V/self.B * agent['action']['amount'] -\
            (agent['action']['type']==SELL) * self.V/self.O * agent['action']['amount']
        agent['timestamp'] = self.ts
        agent['wealth'] += calculate_agent_cash(agent) * self.r #cash earned risk free rate
        agent['wealth'] += agent['stock'] * (self.price-self.price_history[-1]) # how much stock worth changed.
        agent['stock'] += stock_diff #buy/sell happens at new price
        if agent['stock'] < 0. or agent['wealth'] < 0.:
            print(self.ts, agent['wealth'], agent['stock'])
            assert False
        if verbose:
            print("Agent aquired dividend wealth: %s" % (self.dividend * agent['stock']/self.stock_total))
            print("Stock diff is %s" % stock_diff)
        agent.pop('action')
        return agent
    
    def transact(self, agents: list, verbose: bool = False) -> list:
        """Clear all transactions.

        Return: deepcopied agents with cleared transactions and removed action from dict

        If there are more buy bids then fraction of all gets through.
        This is the rationing scheme mentioned in the paper.

        B - bid total
        O - offer total
        V - volume (minimum of O, B)
        """

#         global price, stock_total, dividend, B, O, V, ts
        self.dividend, self.ts = next(self.dividend_with_timestamp)
        self.dividend = max(0,self.dividend)
        self.ts_history.append((self.dividend, self.ts))
        agents = list(map(copy, agents))

        iter_t = lambda action_type: filter(lambda d: d.get('action', {}).get('type') == action_type, agents)
        buyers = list(iter_t(1))
        sellers = list(iter_t(-1))

        self.B = np.sum(list(
            map(lambda d: d['action']['amount'],
            buyers)
        ))
        self.O = np.sum(list(
            map(lambda d: d['action']['amount'],
            sellers)
        ))
        self.V = min(self.B,self.O)
        if (self.V > 0. and verbose):
            print("transaction happened: B=%s,O=%s"%(B,O))

        #oldprice = self.price
        #TODO: Is this where signals ought to be called?
        self.price *= 1 + self.eta*(self.B-self.O)
        self.price_history.append(self.price)
        self.volume_history.append(self.V)
        self.dividend_history.append(self.dividend)

        #print('After transacting (B=%s ;O=%s), %.3f -> %.3f ' % (self.B,self.O,oldprice,self.price) )

        return list(map(self.clear, agents))
    
    def feedback(self, strat):
        if not strat.is_activated(self.market_state):
            k = np.random.choice([0,1,2,3,4],p=[1-self.mutate_prob,self.mutate_prob/4,self.mutate_prob/4,self.mutate_prob/4,self.mutate_prob/4])
            strat.mutate_existing(k=k)
            if k > 0: self.mutate_history.append({'time':self.ts,'strat':strat.name,'agent':strat.agent})
            #if k > 0: print(strat.name)
        if strat.is_activated(self.market_state):
            strat.strength = (1-self.c)*strat.strength + self.c*strat.action*(self.price_history[-1]-(1+self.r)*self.price_history[-2]+self.dividend_history[-1])
            strat.strength = max(min(self.s_max, strat.strength),self.s_min)
            if strat.strength < -0.3:
                #print('inverted', strat.name)
                strat.strength = 0.3
                strat.action *= -1
                
                
    def create_and_add_signal(self, description: str, formula) -> dict:
            """Add dict market signal object with description and signal"""
            self.signals.append({'description': description, 'signal': formula})
            
    def fundamental_value(self):
        """p[t] == dividend/risk_free_rate"""
        return self.dividend/(self.r)
        
    def create_signals(self):
        """Fundamental price signal"""
        s = 'Price is over %s times fundamental value'
        for ratio in np.round(np.linspace(0.25,4.25,8),1):
            self.create_and_add_signal(
                description = s % ratio,
                formula = lambda ratio=ratio: self.price > ratio * self.fundamental_value()
            )

        """Avg relative signal compared to last k days"""
        s = 'Price is over %s times last %s timestep avg'
        for k in np.linspace(1,17,5).astype(np.int64):
            for ratio in np.round(np.linspace(0.5,1.5,5),1):
                self.create_and_add_signal(
                    description = s % (ratio, k),
                    formula = lambda ratio=ratio,k=k: self.price > ratio * np.mean(self.price_history[-k:])
                )

        """Std volatility signal"""
        s = 'Stdev is more than %s over last %s timesteps'
        for k in np.linspace(5,17,3).astype(np.int64):
            for stdev_norm in np.round(np.linspace(0.001,100,2),1):
                self.create_and_add_signal(
                    description = s % (stdev_norm, k),
                    formula = lambda stdev_norm=stdev_norm,k=k: stdev_norm < np.std(self.price_history[-k:])
                )

        """Volume signal"""
        s = 'Volume is more than %s over last %s timestep avg'
        for k in np.linspace(1,5,4).astype(np.int64):
            for vol_norm in np.round(np.linspace(0.01,5,5),1):
                self.create_and_add_signal(
                    description = s % (vol_norm, k),
                    formula = lambda vol_norm=vol_norm,k=k: vol_norm < np.mean(self.volume_history[-k:])
                )
        
    def simulate_k_steps(self, k: int = 1):
        for i in range(k):
            self.market_state_history.append(self.market_state)
            self.agents = self.transact(self.agents)
            self.update_market_state()
            self.feedback_all()
            self.set_next_actions(self.agents)
            self.action_history.append(copy(self.agents))
            if int(100*(i/k)) % 20 == 0 and k > 20 and i % 10 == 0:
                print("%.2f percent done." % int(100*(i/k)) )  
            
    def set_next_actions(self, agents: list) -> None:
        """Set random actions within boundaries (no short selling, negative wealth)"""
        def set_next_action(agent: dict) -> None:
            def calculate_agent_cash(agent: dict) -> float:
                """Return the liquidity of an agent. How much stock can be bought."""
                return agent['wealth'] - agent['stock'] * self.price
            def get_action_within_boundaries(agent, action_type, amount_wish):
                """Avoids selling short and having negative wealth."""
                return min(amount_wish, agent['stock']) \
                      if action_type == SELL \
                      else max(0,min(self.price * amount_wish, calculate_agent_cash(agent)))/self.price
            def select_random_activated_strategy(agent, market_state):
                def create_activated_strategy(market_state):
                    indices = np.random.choice(range(len(self.signals)), replace=False, size=np.random.randint(low=1,high=len(self.signals)))
                    condition = market_state[indices]
                    return Strategy(indices, condition)

                activated_strats = list(filter(
                    lambda strat: strat.is_activated(market_state) and strat.strength > 0.,
                    agent['strats']
                ))
                if len(activated_strats) == 0:
#                   new_strat = create_activated_strategy(market_state)
#                   mutate_strats = np.array((agent['strats']))[
#                     np.argsort(np.abs(list(map(lambda a: a.strength,agent['strats']))))[:10]
#                   ]
#                   for strat in mutate_strats:
#                       strat.mutate_existing(k=5)
                    return None #new_strat
                get_strat_strength = lambda s: s.strength
                norm_const = np.sum(lmap(get_strat_strength, activated_strats))
                activated_strat = np.random.choice(activated_strats, p = lmap(get_strat_strength, activated_strats)/norm_const)
                return activated_strat
            selected_strat = select_random_activated_strategy(agent, self.market_state)
            if selected_strat == None:
                return None
            action_type = selected_strat.action
            amount_wish = 2.
            real_amount = get_action_within_boundaries(agent, action_type, amount_wish)
            agent['action'] = {'type': action_type, 'amount': real_amount}
        list(filter(lambda agent: agent != None, map(set_next_action, agents)))
