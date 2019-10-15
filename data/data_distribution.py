import itertools
import random
from collections import Counter


class Distribute:

    def __init__(self,  num_agents, num_classes):
        self.n_class = num_classes
        self.n_agent = num_agents

    def create_iid(self):
        val = 1/self.n_agent
        distr = [[val for j in range(self.n_class)] for i in range(self.n_agent)]
        return distr

    # Each client have a certain number of labels (2 for original paper of FedAvr)
    # The labels may cross
    def create_non_iid(self, n_labels):

        agents = [random.sample(range(self.n_class), n_labels)for j in range(self.n_agent)]

        count = Counter(i for i in list(itertools.chain.from_iterable(agents)))
        print(count)
        distr = [[0 for j in range(self.n_class)] for i in range(self.n_agent)]

        for i,ag in enumerate(agents):
            for j in ag:
                distr[i][j] = 1/count[j]
        return distr

    # Each client have a certain number of labels (2 for original paper of FedAvr)
    # The labels are unique for each client
    def create_non_iid_exclusive(self, n_labels):

        assert self.n_agent * n_labels <= self.n_class
        random_id = random.sample(range(self.n_class), self.n_agent * n_labels)
        distr = [[0 for j in range(self.n_class)] for i in range(self.n_agent)]

        for i in range(self.n_agent):
            for j in range(n_labels):
                distr[i][random_id.pop(0)] = 1
        return distr


if __name__ == "__main__":

    distribution = Distribute(2, 10)#.create_iid()
    print(distribution.create_non_iid_exclusive(5))
