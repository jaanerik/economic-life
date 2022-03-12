import numpy as np
from copy import deepcopy as copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from uuid import uuid4 as UUID
import json

def ou_process(
    sigma = 1., mu = 10., tau = .05,
    dt = .001, T = 1.0, r = 2
):
    """Returns time series, Ornstein-Uhlenbeck process series

    sigma = 1.  # Standard deviation.
    mu = 10.  # Mean.
    tau = .05  # Time constant.
    dt = .001  # Time step.
    T = 1.  # Total time.
    r = 2 (20 percent?) # Drift"""
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.

    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)

    x = np.r_[np.array([mu]),np.zeros(n-1)]

    for i in range(n - 1):
        #x[i + 1] = x[i] + dt * (-(x[i] - mu ) / tau) + \
        x[i + 1] = x[i] + dt * (-(x[i] - (mu+(1+i*r/n)) ) / tau) + \
            sigma_bis * sqrtdt * np.random.randn()
    return t, x


def _from_matrix(m: np.array):
    return Strategy(condition_market_indices=m[:,0],condition=m[:,1])

def feedback(strat, market_state, context):
    print('feedbackin')
    dividend_history = context['dividend_history']
    volume_history = context['volume_history']
    price_history = context['price_history']
    
    if not strat.is_activated(market_state):
        print('Mutated existing')
        strat.mutate_existing(k=5)
    if strat.is_activated(market_state):
        strat.strength = (1-c)*strat.strength + c*strat.action*(price_history[-1]-(1+r)*price_history[-2]+dividend_history[-1])
        strat.strength = max(min(s_max, strat.strength), s_min)
        if strat.strength < -0.5:
            strat.strength = 0.5
            strat.action *= -1


"""Wealth cannot be less than worth in stock.
Paper has defined wealth := stock * price + cash"""

"""Helper functions"""
def calculate_agent_cash(agent: dict) -> float:
    """Return the liquidity of an agent. How much stock can be bought."""
    return agent['wealth'] - agent['stock'] * price

def get_action_within_boundaries(agent, action_type, amount_wish):
    """Avoids selling short and having negative wealth."""
    return min(amount_wish, agent['stock']) \
            if action_type == SELL \
            else max(0,min(price * amount_wish, calculate_agent_cash(agent)))/price

def create_activated_strategy(market_state):
    indices = np.random.choice(range(len(signals)), replace=False, size=np.random.randint(low=1,high=len(signals)))
    condition = market_state[indices]
    return Strategy(indices, condition)

def select_random_activated_strategy(agent, market_state):
    activated_strats = list(filter(
        lambda strat: strat.is_activated(market_state) and strat.strength > 0.,
        agent['strats']
    ))
    if len(activated_strats) == 0:
        new_strat = create_activated_strategy(market_state)
#         mutate_strats = np.array(agent['strats'])([np.argsort(np.abs(list(map(lambda a: a.strength,agent['strats']))))][:5])
        mutate_strats = np.array((agent['strats']))[
            np.argsort(np.abs(list(map(lambda a: a.strength,agent['strats']))))[:10]
        ]
        for strat in mutate_strats:
            strat.mutate_existing(k=5)
        return None#new_strat
    activated_strat = np.random.choice(activated_strats)
    #print("Some strats active at %s with action %s " %(ts,activated_strat.action))
    return activated_strat

def set_next_actions(agents: list) -> None:
    """Set random actions within boundaries (no short selling, negative wealth)"""
    def set_next_action(agent: dict) -> None:
        selected_strat = select_random_activated_strategy(agent, market_state)
        if selected_strat == None:
            return None
        action_type = selected_strat.action
        amount_wish = 1
        real_amount = get_action_within_boundaries(agent, action_type, amount_wish)
        agent['action'] = {'type': action_type, 'amount': real_amount}

    list(filter(lambda agent: agent != None, map(set_next_action, agents)))

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

"""Main Functions"""
def transact(agents: list, dividend_with_timestamp, verbose: bool = False) -> list:
    """Clear all transactions.

    Return: deepcopied agents with cleared transactions and removed action from dict

    If there are more buy bids then fraction of all gets through.
    This is the rationing scheme mentioned in the paper.

    B - bid total
    O - offer total
    V - volume (minimum of O, B)
    """

    global price, stock_total, dividend, B, O, V, ts
    dividend, ts = next(dividend_with_timestamp)
    if verbose:
        print("Total dividend payout is %s" % dividend)

    agents = list(map(copy, agents))

    iter_t = lambda action_type: filter(lambda d: d.get('action', {}).get('type') == action_type, agents)
    buyers = list(iter_t(1))
    sellers = list(iter_t(-1))

    B = np.sum(list(
        map(lambda d: d['action']['amount'],
        buyers)
    ))
    O = np.sum(list(
        map(lambda d: d['action']['amount'],
        sellers)
    ))
    V = min(B,O)
    if (V > 0. and verbose):
        print("transaction happened: B=%s,O=%s"%(B,O))

    price_history.append(price)
    volume_history.append(V)
    dividend_history.append(dividend)
    price *= 1 + eta*(B-O)

    return list(map(clear, agents))

def feedback_all(agents, market_state, context):
    for agent in agents:
        for strat in agent['strats']:
            feedback(strat, market_state, context)


def create_signals():
    signals = []
    global price
    def create_and_add_signal(description: str, formula) -> dict:
        """Add dict market signal object with description and signal"""
        signals.append({'description': description, 'signal': formula})

    def fundamental_value():
        """p[t] == dividend/risk_free_rate"""
        return dividend/r

    """Fundamental price signal"""
    s = 'Price is over %s times fundamental value'
    for ratio in np.round(np.linspace(0.25,4.25,9),1):
        create_and_add_signal(
            description = s % ratio,
            formula = lambda: price > ratio * fundamental_value()
        )

    """Avg relative signal compared to last k days"""
    s = 'Price is over %s times last %s timestep avg'
    for k in np.linspace(1,17,5).astype(np.int64):
        for ratio in np.round(np.linspace(0.5,1.5,11),1):
            create_and_add_signal(
                description = s % (ratio, k),
                formula = lambda: price > ratio * np.mean(price_history[-k:])
            )

    """Std volatility signal"""
    s = 'Stdev is more than %s over last %s timesteps'
    for k in np.linspace(5,17,3).astype(np.int64):
        for stdev_norm in np.round(np.linspace(0.5,100,5),1):
            create_and_add_signal(
                description = s % (ratio, k),
                formula = lambda: stdev_norm < np.std(price_history[-k:])
            )

    """Volume signal"""
    s = 'Volume is more than %s over last %s timestep avg'
    for k in np.linspace(1,5,3).astype(np.int64):
        for vol_norm in np.round(np.linspace(0.1,5,5),1):
            create_and_add_signal(
                description = s % (ratio, k),
                formula = lambda: stdev_norm < np.mean(volume_history[-k:])
            )
    return signals

c=0.001
s_min,s_max =-1,1
eta = 0.00001
r = .02*10/252 # risk-free rate
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
t,x = ou_process(dt=10**(-6), r=3)
x /= 10
dividend_with_timestamp = iter(zip(x,t))


