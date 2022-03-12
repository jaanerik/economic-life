import numpy as np
from uuid import uuid4 as UUID
from copy import deepcopy as copy

class Strategy:
    """Strategy class for an agent."""
    def __init__(self,condition_market_indices=None,condition=None,agent=None):
        """condition_market_indices - list of indices to select market_state
        condition - what market_state condition have to be for strategy to be activated

        TODO: might be smarter to indeed use ****10*1*** notation instead of indices
        TODO: rm hardcoded market length 59 - len_market_state"""
        self.name = str(UUID())
        self.agent = agent
        if np.any(condition_market_indices == None) and np.any(condition == None):
            self.condition_market_indices = np.random.choice(range(59),size=np.random.randint(1,59),replace=False)
            self.condition_market_indices = np.array(sorted(self.condition_market_indices))
            self.condition = np.random.choice([True,False],size=len(self.condition_market_indices))
        else:
            self.condition_market_indices = condition_market_indices
            self.condition = condition
        self.strength = 0.#np.random.randn()+0.2 #has to be positive for any to be activated
        self.action = np.random.choice([1,-1]) #1 corresponds to BUY

    def __str__(self):
        return "---\nAction : %s,Market indicators: %s,\nCondition: %s\n---" % (self.action, self.condition_market_indices, list(map(int,self.condition)))

    def generalise(self):
        """Exclude some indicators"""
        I = list(range(len(self.condition_market_indices)))
        exclude = np.random.choice(range(0,I.shape[0]), np.random.choice(range(1,3)))
        new_I = I[~np.isin(I, exclude)]
        self.condition_market_indices = condition_market_indices[new_I]
        self.condition = condition[new_I]

    def mutate_new(self, market_state):
        """Add new strategy component"""
        index_signal = np.c_[self.condition_market_indices, self.condition]
        I = np.array(list(range(len(market_state))))
        new_index = np.random.choice(I[~np.isin(I, self.condition_market_indices)])
        new_signal = np.random.randn() > .5
        new_index_signal = np.vstack((index_signal, np.array([new_index, new_signal])))
        new_index_signal = np.array(sorted(new_index_signal, key = lambda t: t[0])) #self.condition order must be preserved
        self.condition_market_indices, self.condition = new_index_signal[:,0], new_index_signal[:,1]

    def mutate_existing(self, k=1):
        """Inverses existing boolean predicate k times.

        It might happen that it inverses same condition twice if k > 1."""
        k_max = len(self.condition_market_indices)
        rel_index = np.random.choice(range(len(self.condition_market_indices)), size=min(k_max,k), replace=False)
        self.condition[rel_index] = ~self.condition[rel_index]

    def to_matrix(self):
        return np.vstack((self.condition_market_indices,self.condition)).transpose()

    def is_activated(self, market_state):
        return np.all(self.condition == market_state[self.condition_market_indices])

    def crossover(strat1, strat2):
        """Combine two strategies by selecting unifrom p for weights.

        Drops duplicates and thus might become a more general version of both parent rules,
        not sure if this is a bug or a feature yet."""
        p = np.random.uniform()
        mat_1,mat_2 = strat1.to_matrix(), strat2.to_matrix()
        l1,l2 = mat_1.shape[0],mat_2.shape[0]
        k1,k2 = int(l1*p),int(l2*(1-p))
        rel_index1 = np.random.choice(range(l1), size=k1, replace=False)
        rel_index2 = np.random.choice(range(l2), size=k2, replace=False)
        mat_1 = mat_1[rel_index1]
        mat_2 = mat_2[rel_index2]
        mat_2 = mat_2[~np.isin(mat_2[:,0], mat_1[:,0])] #deduplicate condition_indices
        new_mat = np.array(sorted(np.r_[mat_1,mat_2], key = lambda t: t[0]))
        return _from_matrix(new_mat)
