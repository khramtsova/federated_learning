import itertools
import random
from collections import Counter
from functools import wraps


class Distribute:

    def __init__(self,  num_agents, num_classes):
        self.n_class = num_classes
        self.n_agent = num_agents
        self.distr = [[0 for j in range(self.n_class)] for i in range(self.n_agent)]

    def _create_distribution(func):
        @wraps(func)
        def wrap(arg, n_labels_per_agent=None, sub_labels=None):
            # if there are some labels in the argument
            if sub_labels:
                arg._change_labels(sub_labels)
            else:
                arg._set_default_labels()
            func(arg, n_labels_per_agent)
            return arg.distr
        return wrap

    @_create_distribution
    def create_iid(self, _):
        val = 1/self.n_agent
        for j in self.labels:
            for i in range(self.n_agent):
                self.distr[i][j] = val
        #self.distr = [[val for j in self.labels] for i in range(self.n_agent)]
        #return distr

    # Each client have a certain number of labels (2 for original paper of FedAvr)
    # The labels may cross
    @_create_distribution
    def create_non_iid(self, n_labels):
        print(self.labels)

        agents = [random.sample(self.labels, n_labels)for j in range(self.n_agent)]

        count = Counter(i for i in list(itertools.chain.from_iterable(agents)))
        #print(count)

        for i, ag in enumerate(agents):
            for j in ag:
                self.distr[i][j] = 1/count[j]

    # Each client have a certain number of labels (2 for original paper of FedAvr)
    # The labels are unique for each client
    @_create_distribution
    def create_non_iid_exclusive(self, n_labels):
        print("n_labels",n_labels)
        assert self.n_agent * n_labels <= len(self.labels)
        random_id = random.sample(self.labels, self.n_agent * n_labels)

        for i in range(self.n_agent):
            for j in range(n_labels):
                self.distr[i][random_id.pop(0)] = 1

    def _change_labels(self, new_labels):
        self.labels = new_labels

    def _set_default_labels(self):
        self.labels = [i for i in range(self.n_class)]


if __name__ == "__main__":

    distribution = Distribute(2, 10)
    distribution.create_iid(sub_labels=[1, 2, 3, 4, 5])
