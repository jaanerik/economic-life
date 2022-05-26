import numpy as np
from copy import deepcopy as copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lmap(f, l): return list(map(f, l))

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
    sigma_bis = sigma * np.sqrt(2. / tau)
    sqrtdt = np.sqrt(dt)
    x = np.r_[np.array([mu,0])]
    i = 0

    #for i in range(n - 1):
    while True:
        #x[i + 1] = x[i] + dt * (-(x[i] - mu ) / tau) + \
        x[i + 1] = x[i] + dt * (-(x[i] - (mu+(1+i*r/n)) ) / tau)
        x = np.r_[x, [x[-1] + dt * (-(x[-1] - (mu+(1+i*r/n)) ) / tau) +\
                sigma_bis + sqrtdt + np.random.randn() ]]
        i += 1
        yield x[-1]

def boolean_gen():
    """Returns alternatively -1 and 1"""
    i = 0
    while True:
        i += 1
        yield 20000.*((-1)**i>0)

def create_n_mat(n=3, step=1):
    """Create matrix that has False, False, True, ... , True as first row 
    (for mean and std per each timestep)

    Return: n x (n-1) matrix for 2 , ... , n aggregated values.
    """
    b = np.where(np.ones((step*n, step*n)))
    return (b[0] > 1+b[1]).reshape((step*n, step*n))[::, :(step*n)-1:step]


def plot_market(
        m, from_index=0, to_index=None, 
        skipstep=1, price_alpha=1., richest_alpha=1.
    ):
    if to_index == None:
        to_index = len(m.buy_history)
    price_history = m.price_history.get()
    df = pd.DataFrame(
        np.array([m.agents_cash.get(), m.agents_stock.get()]).T,
        columns=['cash', 'stock']
    )
    df['wealth'] = df.cash + m.price * df.stock
    richest = df.wealth.argmax()
    prange = np.arange(m.k2,len(price_history)-m.k2+m.k)
    when_buy = (prange[lmap(lambda l: richest in l, m.buy_history[m.k2:])])
    when_buy = when_buy[(when_buy >= from_index) & (when_buy <= to_index)]
    when_sell = (prange[lmap(lambda l: richest in l, m.sell_history[m.k2:])])
    when_sell = when_sell[(when_sell >= from_index) & (when_sell <= to_index)]

    #m.k = 7

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 13))
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

    # a = axes[0]
    axes[0].plot(
        np.arange(m.k2,to_index+m.k2)[from_index:to_index:skipstep],
        price_history[m.k2+from_index:to_index+m.k2:skipstep],
        label='price',
        alpha=price_alpha
    )  # -', '--', '-.', ':', ''
    axes[0].scatter(
        (when_buy),
        # [m.k+from_index:to_index+m.k])
        (np.array(price_history)[when_buy]), 
        s=30, c='green', marker="^", label='richest buy', alpha=richest_alpha)
    axes[0].scatter(
        (when_sell),
        # [m.k+from_index:to_index+m.k])[when_sell],
        (np.array(price_history)[when_sell]),
        s=40, c='red', marker="v", label='richest sell', alpha=richest_alpha)
    X = prange[from_index+m.k2: to_index+m.k2: skipstep]
    axes[1].plot(X, np.array(
        lmap(len, m.buy_history[m.k2+from_index:to_index+m.k2:skipstep])), 
        label='buyers'
    )
    axes[1].plot(X, np.array(lmap(
        len, 
        m.sell_history[from_index+m.k2: to_index+m.k2:skipstep])),
        label='sellers', alpha=0.5
    )
    # axes[1].plot(m.volume_history[m.k:],'--',label = 'vol')
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='best')
    plt.show()


cdir = 'test_model'


def save_model(m, cdir=cdir):  # str((datetime.now())).replace(' ','_')
    np.save(cdir+'/price_history', m.price_history.get())
    np.save(cdir+'/volume_history', m.volume_history.get())
    np.save(cdir+'/buy_history', np.array(m.buy_history, dtype=object))
    np.save(cdir+'/sell_history', np.array(m.sell_history, dtype=object))
    np.save(cdir+'/strats', m.strats.get())
    np.save(cdir+'/actions', m.actions.get())
    np.save(cdir+'/agents_cash', m.agents_cash)
    np.save(cdir+'/agents_stock', m.agents_stock)
    print(f"Saved model to dir: {cdir}")


def read_model(cdir=cdir):
    m = Market()
    m.price_history = cp.array(np.load(cdir+'/price_history.npy'))
    m.volume_history = cp.array(np.load(cdir+'/volume_history.npy'))
    m.buy_history = list(np.load(cdir+'/buy_history.npy', allow_pickle=True))
    m.sell_history = list(np.load(cdir+'/sell_history.npy', allow_pickle=True))
    m.strats = cp.array(np.load(cdir+'/strats.npy'))
    m.actions = cp.array(np.load(cdir+'/actions.npy'))
    m.agents_cash = cp.array(np.load(cdir+'/agents_cash.npy'))
    m.agents_stock = cp.array(np.load(cdir+'/agents_stock.npy'))
    return m
