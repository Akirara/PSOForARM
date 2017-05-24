"""PSO Algorithm

This module contains methods for searching optimum solution using PSO algorithm.

The module structure is the following:

- The ``Particle`` class implements the basic particle structure.

- The ``PSO`` class implements the call interface of the algorithm.
"""

# Authors: Zhao Xuan
# Created at December 8th, 2016

import numpy as np
import random
import math
import datetime
import copy


class Particle(object):
    """Particle Definition.

    Particle builds a basic particle model for PSO algorithm.

    Parameters
    ----------
    dimension : int, optional (default=3)
        Dimensions of a particle.
        Indicates the length of one particle position array.

    random_init : boolean, optional (default=False)
        Whether init the particle position randomly.
        If True, init the position according to lower_bound and upper_bound; if lower_bound or upper_bound is None,
        take (0, 1) as random range.
        If False, init the position all by zero(the origin).

    lower_bound : array, optional (default=None)
        The low boundary of random range.
        Array length should equals to the dimension.

    upper_bound : array, optional (default=None)
        The high boundary of random range.
        Array length should equals to the dimension.

    Attributes
    ----------
    _position : ndarray, shape = [3 * dimension]
        The current position of the particle.

    _local_best : array, shape = [{'pos': , 'fit': }]
        The local best history position and corresponding fitness.

    global_best : array, shape = [{'pos': , 'fit': }]
        The global best history position and corresponding fitness.

    _velocity : ndarray, shape = [3 * dimension]
        The current velocity of the particle.

    _fitness : array
        The current fitness of the particle, calculated from current position.
        Array length depends on the number of optimization objectives.

    References
    ----------
    Beiranvand V, Mobasher-Kashani M, Bakar A A. Multi-objective PSO algorithm for mining numerical association
    rules without a priori discretization. Expert Systems with Applications An International Journal,
    2014, 41(9):4259-4273.
    """
    global_best = []

    def __init__(self, dimension=3, random_init=False, n_interval=10, lower_bound=None, upper_bound=None):
        if random_init:
            self._position = np.array(Particle._generate_random_list(dimension, n_interval, lower_bound, upper_bound))
        else:
            self._position = np.zeros(3 * dimension)
        self._local_best = []
        self._velocity = np.zeros(3 * dimension)
        self._fitness = []
        self.n_interval = 10
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.velocity_bound = self.set_velocity_bound()

    def set_velocity_bound(self):
        velocity_bound = []
        for i in range(len(self.lower_bound)):
            step = (self.upper_bound[i] - self.lower_bound[i]) / self.n_interval
            velocity_bound.append(0.33)
            velocity_bound.append(step)
            velocity_bound.append(step)
        return velocity_bound

    def get_position(self):
        return self._position

    def update_fitness(self, fitness):
        self._fitness = fitness

    def _update_best(self, best_set):
        """Base method to update the best set.

        Parameters
        ----------
        best_set : array, shape = [{'pos': , 'fit': }]
            The set to be updated, self._local_best or Particle.global_best.
        """
        if len(best_set) == 0:
            best_set.append({'pos': self._position, 'fit': self._fitness})
        else:
            delete_index = []
            for i in range(len(best_set)):
                flag = Particle._compare(self._fitness, best_set[i]['fit'])
                if flag == 1:
                    delete_index.append(i)
                elif flag == -1:
                    return
            for i in range(0, len(delete_index))[::-1]:
                del best_set[i]
            best_set.append({'pos': copy.deepcopy(self._position), 'fit': copy.deepcopy(self._fitness)})

    def update_local_best(self):
        self._update_best(self._local_best)

    def update_global_best(self):
        self._update_best(Particle.global_best)

    def update_velocity(self, w, c1, c2):
        """Update the particle velocity.

        Use Roulette Wheel Selection to select lbest and gbest from local_best and global_best,
        where r1 and r2 are two uniform random numbers in (0, 1) that bring the stochastic
        state to the algorithm.

        Parameters
        ----------
        w : float
            Inertia weight.
        c1, c2 : float
            Two constant value, usually referred to as cognitive and social factors.
        """
        r1 = random.random()
        r2 = random.random()
        li = random.randint(0, len(self._local_best) - 1)
        gi = random.randint(0, len(Particle.global_best) - 1)
        self._velocity = w * self._velocity + c1 * r1 * (self._local_best[li]['pos'] - self._position) \
                        + c2 * r2 * (Particle.global_best[gi]['pos'] - self._position)
        for i in range(len(self.velocity_bound)):
            if abs(self._velocity[i]) > self.velocity_bound[i]:
                self._velocity[i] = self.velocity_bound[i] if self._velocity[i] > 0 else -self.velocity_bound[i]

    def update_position(self):
        """Update the current position.

        Revise the position if any dimension overflows .
        """
        self._position += self._velocity
        for i in range(len(self.lower_bound)):
            index = 3 * i
            self._position[index] = max(self._position[index], 0)
            self._position[index] = min(self._position[index], 1)
            self._position[index + 1] = max(self._position[index + 1], self.lower_bound[i])
            self._position[index + 1] = min(self._position[index + 1], self.upper_bound[i])
            self._position[index + 2] = max(self._position[index + 2], self._position[index + 1])
            self._position[index + 2] = min(self._position[index + 2], self.upper_bound[i])

    @staticmethod
    def _generate_random_list(length, n_interval, low, high):
        """Generate a list with random values.

        Random range is defined by [low, high] or (0, 1) if low / high is None.

        Parameters
        ----------
        length : int
            The length of array.
        low : array, shape = [length], optional
            The low boundary list of random range.
        high : array, shape = [length], optional
            The high boundary list of random range.

        Returns
        -------
        random_list : array, shape = [3 * length]
            The generated random list.
        """
        random_list = []
        for i in range(length):
            random_list.append(random.random())
            k = random.randint(0, n_interval - 1)
            step = (high[i] - low[i]) / n_interval
            lower_bound = low[i] + k * step
            upper_bound = lower_bound + step
            random_list.append(lower_bound)
            random_list.append(upper_bound)
        return random_list

    @staticmethod
    def _compare(f1, f2):
        """Multi objectives comparison function.

        Parameters
        ----------
        f1, f2 : array-like
            Fitness list to be compared.

        Returns
        -------
        result : 1, -1 or 0
            Return 1(-1) if every dimension of f1 is better(worse) than f2, otherwise return 0.
        """
        dominate = dominated = True
        for i in range(len(f1)):
            if dominate or dominated:
                if f1[i] > f2[i]:
                    dominated = False
                elif f1[i] < f2[i]:
                    dominate = False
            else:
                return 0
        return 1 if dominate else -1


class PSO(object):
    """Particle Swarm Optimization Algorithm.

    Search optimum solution for multi objectives.

    Parameters
    ----------
    population : int, optional (default = 10)
        Swarm population, number of particles.

    dimension : int, optional (default = 3)
        Dimensions of a particle.

    random_init : boolean, optional (default = False)
        Whether init the particle position randomly.

    lower_bound : array, optional (default=None)
        The low boundary of random range.

    upper_bound : array, optional (default=None)
        The high boundary of random range.

    fit : function, optional (default = None)
        Fitness calculate function.
        Take particle position as input and return the corresponding fitness.

    w : float, optional (default = 0.4)
        Inertia weight.

    c1, c2 : float, optional (default = 2)
        Constant value, usually referred to as cognitive and social factors.

    iteration : int, optional (default = 100)
        Iteration number.

    Attributes
    ----------
    _swarm : array of Particle, shape = [population]
        Init the swarm with {population} particles.
    """
    def __init__(self, population=10, dimension=3, random_init=False, n_interval=10, lower_bound=None, upper_bound=None,
                 fit=None, w=0.4, c1=2, c2=2, iteration=100):
        self._swarm = [
            Particle(
                dimension=dimension,
                random_init=random_init,
                n_interval=n_interval,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ) for i in range(population)]
        self._iteration = iteration
        self._fit = PSO._default_fit if fit is None else fit
        self._w = w
        self._c1 = c1
        self._c2 = c2

    @staticmethod
    def _default_fit(particle):
        """Default fitness calculate function.

         Returns
         -------
         [x1, x2] : array
            Array of length 2, values in range (0, 1).
        """
        return [random.random(), random.random()]

    def run(self):
        """Run PSO algorithm."""
        for i in range(self._iteration):
            j = 0
            for particle in self._swarm:
                j += 1
                print(datetime.datetime.now(), "Iteration: ", i, " Particle: ", j)
                begin = datetime.datetime.now()
                # print("Begin time : %s" % begin)
                print("%s Update fitness" % begin)
                particle.update_fitness(self._fit(particle.get_position()))
                end = datetime.datetime.now()
                print("Cost : %s" % (end - begin).seconds)
                print("Update local best")
                particle.update_local_best()
                print("Update global best")
                particle.update_global_best()
                print("Update velocity")
                particle.update_velocity(self._w, self._c1, self._c2)
                print("Update position")
                particle.update_position()


class PSOForARM(PSO):
    def __init__(self, dataset=None, population=10, dimension=3,
                 random_init=False, n_interval=10, lower_bound=None, upper_bound=None,
                 w=0.4, c1=2, c2=2, iteration=100):
        self.dataset = dataset
        self.n_transaction = len(dataset)
        super(PSOForARM, self).__init__(
            population=population,
            dimension=dimension,
            random_init=random_init,
            n_interval=n_interval,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fit=self.calculate_fitness,
            w=w,
            c1=c1,
            c2=c2,
            iteration=iteration
        )

    def calculate_fitness(self, rule):
        sup_ant = 0
        sup_con = 0
        sup = 0
        n_ant = 0
        n_con = 0
        i = 0
        while i < len(rule):
            if rule[i] <= 0.66:
                if rule[i] < 0.34:
                    n_ant += 1
                else:
                    n_con += 1
            i += 3
        for transaction in self.dataset:
            ant = True
            con = True
            i = 0
            while (ant or con) and i < len(transaction):
                """
                <=0.33 ant
                0.34 - 0.66 con
                >=0.67 none
                """
                if rule[3 * i] > 0.66:
                    i += 1
                    continue
                elif transaction[i] < rule[3 * i + 1] or transaction[i] > rule[3 * i + 2]:
                    if rule[3 * i] <= 0.33:
                        ant = False
                    else:
                        con = False
                i += 1
            if ant:
                sup_ant += 1
            if con:
                sup_con += 1
            if ant and con:
                sup += 1
        if sup_ant == 0 or sup_con == 0 or n_ant == 0 or n_con == 0:
            return [0, 0, 0, 0]
        support = sup / self.n_transaction
        confidence = sup / sup_ant
        comprehensibility = math.log(n_con + 1) / math.log(n_ant + n_con + 1)
        interestingness = confidence * (sup / sup_con) * (1 - support)
        return [support, confidence, comprehensibility, interestingness]


class FileIO(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    """Three data sets from "http://funapp.cs.bilkent.edu.tr/DataSets/".
    HH: House_16H, attributes: 16, Size: 22784
    QU: Quake, attributes: 4, Size: 2178
    SP: Stock Prices, attributes: 10, Size: 950
    """
    print("Begin time : %s" % datetime.datetime.now())
    HH = np.loadtxt("C:/Users/Zhaoxuan/Desktop/Data/HH.dat", dtype=np.float64, delimiter=',')
    QU = np.loadtxt("C:/Users/Zhaoxuan/Desktop/Data/QU.dat", dtype=np.float64, delimiter=',')
    SP = np.loadtxt("C:/Users/Zhaoxuan/Desktop/Data/SP.dat", dtype=np.float64)
    # data = np.loadtxt("C:/Users/Zhaoxuan/Desktop/pso_test_data/pso_test_data_200.txt", dtype=np.float64, delimiter=',')
    data = QU
    low = data.min(axis=0)
    high = data.max(axis=0)
    """every parameter combination test time."""
    time = 1
    """set population every 10 from 0 to 200."""
    # populations = np.arange(0, 210, 10)
    populations = [10]
    """set iteration every 100 from 0 to 2000."""
    # iterations = np.arange(0, 2100, 100)
    iterations = [100]
    ARs = []
    fitness = []
    runTimes = []
    for population in populations:
        print("Population: ", population)
        timeTmp = []
        fitTmp = np.zeros(4)
        for t in range(time):
            p = PSOForARM(dataset=data, population=population, dimension=4, random_init=True, n_interval=20,
                          iteration=10, lower_bound=low, upper_bound=high)
            # begin = datetime.datetime.now()
            # print("Begin time : %s" % begin)
            p.run()
            # end = datetime.datetime.now()
            # print("Finish time : %s" % end)
            # print("Cost : %s" % (end - begin).seconds)
            # timeTmp.append((end - begin).seconds)
            for gb in Particle.global_best:
                fitTmp += gb['fit']
            ARs.append(len(Particle.global_best))
            for record in Particle.global_best:
                print(record['fit'])
                out = np.array(record['pos'])
                print(out.reshape(-1, 3))
        runTimes.append(np.array(timeTmp).sum() / time)
        fitness.append(fitTmp / time)
        # print("Average:", np.array(ARs).sum() / time)
    np.savetxt("C:/Users/Zhaoxuan/Desktop/sp_population_time.txt", runTimes, delimiter=',')
    np.savetxt("C:/Users/Zhaoxuan/Desktop/sp_population_fitness.txt", fitness, delimiter=',')
