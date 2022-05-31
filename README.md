# Artificial Economic Life
## Model of a simple stock market

[![version](https://img.shields.io/badge/version-0.0.1-yellow.svg)](https://semver.org)

[Eesti keelne kirjeldus](https://docs.google.com/document/d/e/2PACX-1vQ7bxCckPXOJBdQC7R79ytD9tJND1fqNh1x041VnPAJdxm-njXjInUWh7CutEVJ918weBhtF4m0bwon/pub)

This is a reproduction of a 1994 paper [Artificial economic life](https://deepblue.lib.umich.edu/handle/2027.42/31402) by Palmer, et al. It tries to implement ideas described in the paper, but is quite quick thanks to being implemented on a GPU.

Feel free to look at the more interesting [scenarios](situations/) or to look at the [Jupyter notebook](Presentation.ipynb).

<img
  src="https://i.imgur.com/iIKBljg.png"
  alt="Sample"
  title="Output of a simulation"
  style="display: inline-block; margin: 0 auto; max-width: 200px"
  width="700" height="700"/>
  
<img
  src="https://i.imgur.com/NaW5P8b.png"
  alt="Sample2"
  title="Analysis of strategies used"
  style="display: inline-block; margin: 0 auto; max-width: 200px"
  scale="1"/>

## Motivation

There recently has been a surge of [evolutionary biology simulations](https://www.youtube.com/watch?v=N3tRFayqVtk&t=1147s) and [simulations of different traits in evolution](https://www.youtube.com/watch?v=YNMkADpvO4w). Also I have felt interest in the works of Stephen Wolfram and his idea of [computational irreducibility](https://en.wikipedia.org/wiki/Computational_irreducibility). This led me to the paper by Palmer, Arthur, et al, which ran a stock market simulation back in 1994. The thesis of the paper is that the price of a stock emerges as the result of complex network of traders maximising their profit and this cannot be solved analytically or by finding points of equilibria. So running simulations, even easy ones, can offer new insight into this field and others.

## Theory

The simulation starts by creating e.g. 1000 agents with 100 strategies. Each agent chooses a strategy that they will use to trade one possible stock. Strategy is chosen by looking at the state of the market, which is a simple boolean array of yes/no answers to different questions - for example "is the current trading volume higher than the average of the past 5 timestep trading volume?". Each strategy also has an corresponding action, so if the market state fits the strategy, then it might buy the stock if agent has the cash to do so. If the decision was to buy stock, but the price drops, then it gives a negative feedback to the strategy's strength, which very roughly corresponds to the reliability of the strategy. Each strategy is currently (2022-05-30) chosen randomly, but will soon be updated. By letting agents compete, some interesting behaviour emerges.

The main point of interest to me is this - by letting agents access the signal of market state and by competing with each other, agents learn what those signals mean without having any access to the explanation beforehand. Or they have to create their own meaning. So this simulation is like a playground to test different ideas related to agent-based machine learning as well.

## Next steps

In addition to the finishing touches to the code to fully reproduce the paper, I have following ideas with this project.

- Learning to predict simple toy signal. This would involve figuring out how agents can access signal that it needs to predict and how it would make sure that it knows how to solve it - how to prune agents who are not contributing.
- More complex reasoning. Agents have either individual MLP or some shared hidden neurons as "concepts" to "communicate" between each other.
- Memory. How to save the information in a way that it can be reproduced? One idea to test out would be to close off agents in groups of 10 and making predictions as a [mixture of experts](https://en.wikipedia.org/wiki/Mixture_of_experts).
- Fitting the model with historical data. The simulation would generate different scenarios relative to minor changes in the initial conditions. This might be the most practical of tasks as it would be potentially useful to create proper backtesting in the stock market instead of just overfitting everything to the current data as it is currently done.


## Dependencies
The only non-standard library is currently cupy and is not compatible with CPU at the moment.
