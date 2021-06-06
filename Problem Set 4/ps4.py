# Problem Set 4: Bacteria Population and Spreading of Disease
# Name:

from functools import reduce
import math
import numpy as np
import pylab as pl
import random
import tqdm
from typing import *

randomizer = random.Random()

def use_seed(s: int) -> None:
    randomizer.seed = s

def event_happens(probability: float) -> bool:
    """
    Resolve a probability into success (True) or failure (False) using
    randomness.

    Example
    =======
    ```
        if event_happens(0.33):
            print("This event has a 33% chance of happening.")
        else:
            print("This failure has a 67% chance of happening.")
    ```

    Parameters
    ==========
    probability: float
        Probability of the event happening. Should be between 0.0 and 1.0,
        inclusive of both extremes.
    
    Returns
    =======
    bool
        Whether the event happens.
    """
    if probability <= 0.0:
        return False
    elif probability >= 1.0:
        return True
    else:
        return randomizer.random() <= probability

##########################
# helper code
##########################

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleBacteria
    and ResistantBacteria classes to indicate that a bacteria cell does not
    reproduce. You should use NoChildException as is; you do not need to
    modify it or add any code.
    """


def plot_sim_mean(sim: List[List[int]], *args, **kwargs) -> None:
    x_coords = list(range(len(sim[0])))
    y_coords = [calc_pop_avg(sim, j) for j in x_coords]
    make_one_curve_plot(x_coords, y_coords, *args, **kwargs)


def make_one_curve_plot(x_coords, y_coords, x_label, y_label, title):
    """
    Makes a plot of the x coordinates and the y coordinates with the labels
    and title provided.

    Args:
        x_coords (list of floats): x coordinates to graph
        y_coords (list of floats): y coordinates to graph
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.grid()
    pl.show()


def make_two_curve_plot(x_coords,
                        y_coords1,
                        y_coords2,
                        y_name1,
                        y_name2,
                        x_label,
                        y_label,
                        title):
    """
    Makes a plot with two curves on it, based on the x coordinates with each of
    the set of y coordinates provided.

    Args:
        x_coords (list of floats): the x coordinates to graph
        y_coords1 (list of floats): the first set of y coordinates to graph
        y_coords2 (list of floats): the second set of y-coordinates to graph
        y_name1 (str): name describing the first y-coordinates line
        y_name2 (str): name describing the second y-coordinates line
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): the title of the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords1, label=y_name1)
    pl.plot(x_coords, y_coords2, label=y_name2)
    pl.legend()
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.grid()
    pl.show()


##########################
# PROBLEM 1
##########################

class SimpleBacteria(object):
    """A simple bacteria cell with no antibiotic resistance"""

    def __init__(self, birth_prob: float, death_prob: float):
        """
        Args:
            birth_prob (float in [0, 1]): Maximum possible reproduction
                probability
            death_prob (float in [0, 1]): Maximum death probability
        """
        self.alive: bool = True
        self.birth_prob: float = birth_prob
        self.death_prob: float = death_prob

    @classmethod
    def colony(
        cls,
        num: int,
        *args,
        **kwargs
    ) -> List["SimpleBacteria"]:
        """
        Generate a colony of identical bacteria with the same birth
        probability, death probability.

        Parameters
        ==========
        num: int
            Number of bacteria in the colony.
        
        birth_prob: float
            Maximum possible reproduction probability. Must be between 0 and 1.
        
        death_prob: float
            Maximum death probability. Must be between 0 and 1.
        
        Returns
        =======
        List[SimpleBacteria]
        """
        return [cls(*args, **kwargs) for _ in range(num)]

    @property
    def dead(self) -> bool:
        return not self.alive

    def is_killed(self) -> bool:
        """
        Stochastically determines whether this bacteria cell is killed in
        the patient's body at a time step, i.e. the bacteria cell dies with
        some probability equal to the death probability each time step.

        Returns:
            bool: True with probability self.death_prob, False otherwise.
        """
        if self.dead:
            return True
        else:
            self.alive = not event_happens(self.death_prob)
            return not self.alive

    def clone(self) -> "SimpleBacteria":
        return SimpleBacteria(self.birth_prob, self.death_prob)

    def reproduce(self, pop_density: float) -> "SimpleBacteria":
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes.

        The bacteria cell reproduces with probability
        self.birth_prob * (1 - pop_density).

        If this bacteria cell reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleBacteria (which has the same
        birth_prob and death_prob values as its parent).

        Args:
            pop_density (float): The population density, defined as the
                current bacteria population divided by the maximum population

        Returns:
            SimpleBacteria: A new instance representing the offspring of
                this bacteria cell (if the bacteria reproduces). The child
                should have the same birth_prob and death_prob values as
                this bacteria.

        Raises:
            NoChildException if this bacteria cell does not reproduce.
        """
        if self.dead:
            raise NoChildException(
                "Bacteria at location {0} has died.".format(id(self))
            )
        elif event_happens(self.birth_prob * (1.0 - pop_density)):
            return self.clone()
        else:
            raise NoChildException(
                "Bacteria at location {0} cannot reproduce.".format(id(self))
            )


class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any
    antibiotics and his/her bacteria populations have no antibiotic resistance.
    """
    def __init__(self, bacteria: List[SimpleBacteria], max_pop: int):
        """
        Args:
            bacteria (list of SimpleBacteria): The bacteria in the population
            max_pop (int): Maximum possible bacteria population size for
                this patient
        """
        if len(bacteria) > max_pop:
            raise ValueError(
                (
                    "There are more starting bacteria than the allowed maximum"
                    " population. Bacteria: {0}, Max allowed: {1}"
                ).format(len(bacteria), max_pop)
            )
        self.bacteria: List[SimpleBacteria] = bacteria
        self.max_pop: int = max_pop

    @property
    def density(self) -> float:
        return self.get_total_pop() / self.max_pop

    def get_total_pop(self) -> int:
        """
        Gets the size of the current total bacteria population.

        Returns:
            int: The total bacteria population
        """
        return len(self.bacteria)

    def update(self) -> int:
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute the following steps in
        this order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. Calculate the current population density by dividing the surviving
           bacteria population by the maximum population. This population
           density value is used for the following steps until the next call
           to update()

        3. Based on the population density, determine whether each surviving
           bacteria cell should reproduce and add offspring bacteria cells to
           a list of bacteria in this patient. New offspring do not reproduce.

        4. Reassign the patient's bacteria list to be the list of surviving
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        self.bacteria = list(
            filter(lambda b: not b.is_killed(), self.bacteria)
        )
        n_survivors: int = self.get_total_pop()
        density: float = self.density
        for i_survivor in range(n_survivors):
            try:
                new_bacteria: SimpleBacteria = self.bacteria[
                    i_survivor
                ].reproduce(
                    density
                )
            except NoChildException:
                continue
            else:
                self.bacteria.append(new_bacteria)
        return self.get_total_pop()
    
    def elapse(self, steps: int = 1) -> "Patient":
        for _ in range(steps):
            self.update()
        return self
    
    def elapse_with_result(self, steps: int = 1) -> List[int]:
        initial = self.get_total_pop()
        results = [self.update() for _ in range(steps)]
        results.insert(0, initial)
        return results
        

##########################
# PROBLEM 2
##########################

def calc_pop_avg(populations: List[List[int]], n: int) -> float:
    """
    Finds the average bacteria population size across trials at time step n

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j

    Returns:
        float: The average bacteria population size at time step n
    """
    add_up: Callable[[int, List[int]], int] = lambda a, pop_i: a + pop_i[n]
    return reduce(add_up, populations, 0) / len(populations)


def simulation_without_antibiotic(
    num_bacteria: int,
    max_pop: int,
    birth_prob: float,
    death_prob: float,
    num_trials: int,
    timesteps: int = 300,
    do_plot: bool = True,
    *args,
    **kwargs
) -> List[List[int]]:
    """
    Run the simulation and plot the graph for problem 2. No antibiotics
    are used, and bacteria do not have any antibiotic resistance.

    For each of num_trials trials:
        * instantiate a list of SimpleBacteria
        * instantiate a Patient using the list of SimpleBacteria
        * simulate changes to the bacteria population for 300 timesteps,
          recording the bacteria population after each time step. Note
          that the first time step should contain the starting number of
          bacteria in the patient

    Then, plot the average bacteria population size (y-axis) as a function of
    elapsed time steps (x-axis) You might find the make_one_curve_plot
    function useful.

    Args:
        num_bacteria (int): number of SimpleBacteria to create for patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float in [0, 1]): maximum reproduction
            probability
        death_prob (float in [0, 1]): maximum death probability
        num_trials (int): number of simulation runs to execute

    Returns:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    _pbar: bool = False
    try:
        _pbar = kwargs["progress_bar"]
    except:
        pass
    simulations = []
    iterator = tqdm.tqdm(range(num_trials)) if _pbar else range(num_trials)
    for _ in iterator:
        simulations.append(
            Patient(
                SimpleBacteria.colony(num_bacteria, birth_prob, death_prob),
                max_pop
            ).elapse_with_result(timesteps)
        )
    if do_plot:
        plot_sim_mean(
            simulations,
            "Timestep",
            "Number of bacteria",
            "Bacteria over Time"
        )
    return simulations
    


# When you are ready to run the simulation, uncomment the next line
# populations = simulation_without_antibiotic(100, 1000, 0.1, 0.025, 50)

##########################
# PROBLEM 3
##########################

def calc_pop_std(populations: List[List[int]], t: int) -> float:
    """
    Finds the standard deviation of populations across different trials
    at time step t by:
        * calculating the average population at time step t
        * compute average squared distance of the data points from the average
          and take its square root

    You may not use third-party functions that calculate standard deviation,
    such as numpy.std. Other built-in or third-party functions that do not
    calculate standard deviation may be used.

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        float: the standard deviation of populations across different trials at
             a specific time step
    """
    avg = calc_pop_avg(populations, t)
    return math.sqrt(
        reduce(lambda stddev, pop: stddev+(pop[t]-avg)**2, populations, 0)
        / len(populations)
    )


def calc_95_ci(populations: List[List[int]], t: int) -> Tuple[float, float]:
    """
    Finds a 95% confidence interval around the average bacteria population
    at time t by:
        * computing the mean and standard deviation of the sample
        * using the standard deviation of the sample to estimate the
          standard error of the mean (SEM)
        * using the SEM to construct confidence intervals around the
          sample mean

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        mean (float): the sample mean
        width (float): 1.96 * SEM

        I.e., you should return a tuple containing (mean, width)
    """
    mean = calc_pop_avg(populations, t)
    width = 1.96 * calc_pop_std(populations, t) / math.sqrt(len(populations))
    return mean, width


##########################
# PROBLEM 4
##########################

class ResistantBacteria(SimpleBacteria):
    """A bacteria cell that can have antibiotic resistance."""

    def __init__(
        self,
        birth_prob: float,
        death_prob: float,
        resistant: bool,
        mut_prob: float
    ):
        """
        Args:
            birth_prob (float in [0, 1]): reproduction probability
            death_prob (float in [0, 1]): death probability
            resistant (bool): whether this bacteria has antibiotic resistance
            mut_prob (float): mutation probability for this
                bacteria cell. This is the maximum probability of the
                offspring acquiring antibiotic resistance
        """
        super().__init__(birth_prob, death_prob)
        self.resistant: bool = resistant
        self.mut_prob: bool = mut_prob
        self.alive = True

    @classmethod
    def colony(
        cls,
        num: int,
        *args,
        **kwargs
    ):
        """
        Generate a colony of identical bacteria with the same birth
        probability, death probability.

        Parameters
        ==========
        num: int
            Number of bacteria in the colony.
        
        birth_prob: float
            Maximum possible reproduction probability. Must be between 0 and 1.
        
        death_prob: float
            Maximum death probability. Must be between 0 and 1.
        
        resistant: bool
            Whether the bacteria in the colony are resistant.
        
        mut_prob: float
            The probability of mutating a resistance gene.
        
        Returns
        =======
        List[ResistantBacteria]
        """
        return [cls(*args, **kwargs) for _ in range(num)]

    def get_resistant(self) -> bool:
        """Returns whether the bacteria has antibiotic resistance"""
        return self.resistant

    def is_killed(self) -> bool:
        """Stochastically determines whether this bacteria cell is killed in
        the patient's body at a given time step.

        Checks whether the bacteria has antibiotic resistance. If resistant,
        the bacteria dies with the regular death probability. If not resistant,
        the bacteria dies with the regular death probability / 4.

        Returns:
            bool: True if the bacteria dies with the appropriate probability
                and False otherwise.
        """
        if self.dead:
            return True
        else:
            if self.resistant:
                self.alive = not event_happens(self.death_prob)
            else:
                self.alive = not event_happens(self.death_prob / 4)
            return not self.alive

    def clone(self) -> "ResistantBacteria":
        return ResistantBacteria(
            self.birth_prob,
            self.death_prob,
            self.resistant,
            self.mut_prob
        )

    def reproduce(self, pop_density: float) -> bool:
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A surviving bacteria cell will reproduce with probability:
        self.birth_prob * (1 - pop_density).

        If the bacteria cell reproduces, then reproduce() creates and returns
        an instance of the offspring ResistantBacteria, which will have the
        same birth_prob, death_prob, and mut_prob values as its parent.

        If the bacteria has antibiotic resistance, the offspring will also be
        resistant. If the bacteria does not have antibiotic resistance, its
        offspring have a probability of self.mut_prob * (1-pop_density) of
        developing that resistance trait. That is, bacteria in less densely
        populated environments have a greater chance of mutating to have
        antibiotic resistance.

        Args:
            pop_density (float): the population density

        Returns:
            ResistantBacteria: an instance representing the offspring of
            this bacteria cell (if the bacteria reproduces). The child should
            have the same birth_prob, death_prob values and mut_prob
            as this bacteria. Otherwise, raises a NoChildException if this
            bacteria cell does not reproduce.
        """
        if self.dead:
            raise NoChildException(
                "Bacteria at location {0} has died.".format(id(self))
            )
        elif event_happens(self.birth_prob * (1.0 - pop_density)):
            offspring = self.clone()
            if not self.resistant:
                offspring.resistant = event_happens(self.mut_prob)
            return offspring
        else:
            raise NoChildException(
                "Bacteria at location {0} cannot reproduce.".format(id(self))
            )


class TreatedPatient(Patient):
    """
    Representation of a treated patient. The patient is able to take an
    antibiotic and his/her bacteria population can acquire antibiotic
    resistance. The patient cannot go off an antibiotic once on it.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria: The list representing the bacteria population (a list of
                      bacteria instances)
            max_pop: The maximum bacteria population for this patient (int)

        This function should initialize self.on_antibiotic, which represents
        whether a patient has been given an antibiotic. Initially, the
        patient has not been given an antibiotic.

        Don't forget to call Patient's __init__ method at the start of this
        method.
        """
        self.bacteria = bacteria
        self.max_pop = max_pop
        self.on_antibiotics: bool = False

    def set_on_antibiotic(self):
        """
        Administer an antibiotic to this patient. The antibiotic acts on the
        bacteria population for all subsequent time steps.
        """
        self.on_antibiotics = True

    def get_resist_pop(self) -> int:
        """
        Get the population size of bacteria cells with antibiotic resistance

        Returns:
            int: the number of bacteria with antibiotic resistance
        """
        return reduce(
            lambda a, _: a + 1,
            filter(lambda b: b.resistant, self.bacteria),
            0
        )

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute these actions in order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. If the patient is on antibiotics, the surviving bacteria cells from
           (1) only survive further if they are resistant. If the patient is
           not on the antibiotic, keep all surviving bacteria cells from (1)

        3. Calculate the current population density. This value is used until
           the next call to update(). Use the same calculation as in Patient

        4. Based on this value of population density, determine whether each
           surviving bacteria cell should reproduce and add offspring bacteria
           cells to the list of bacteria in this patient.

        5. Reassign the patient's bacteria list to be the list of survived
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        self.bacteria = list(filter(lambda b: not b.is_killed(), self.bacteria))
        if self.on_antibiotics:
            self.bacteria = list(filter(lambda b: b.resistant, self.bacteria))
        n_survivors: int = self.get_total_pop()
        density: float = self.density
        # print(density)
        for i_survivor in range(n_survivors):
            try:
                offspring = self.bacteria[
                    i_survivor
                ].reproduce(
                    density
                )
            except NoChildException:
                continue
            else:
                self.bacteria.append(offspring)
        # print([b.resistant for b in self.bacteria])
        return self.get_total_pop()
    
    def elapse_with_result(self, steps: int) -> List[Tuple[int, int]]:
        initial = self.get_total_pop(), self.get_resist_pop()
        total = [(self.update(), self.get_resist_pop()) for _ in range(steps)]
        total.insert(0, initial)
        return total


##########################
# PROBLEM 5
##########################

def simulation_with_antibiotic(
    num_bacteria: int,
    max_pop: int,
    birth_prob: float,
    death_prob: float,
    resistant: bool,
    mut_prob: float,
    num_trials: int,
    untreated_time: int = 150,
    treated_time: int = 250,
    do_plot: bool = True,
    *args,
    **kwargs
):
    """
    Runs simulations and plots graphs for problem 4.

    For each of num_trials trials:
        * instantiate a list of ResistantBacteria
        * instantiate a patient
        * run a simulation for 150 timesteps, add the antibiotic, and run the
          simulation for an additional 250 timesteps, recording the total
          bacteria population and the resistance bacteria population after
          each time step

    Plot the average bacteria population size for both the total bacteria
    population and the antibiotic-resistant bacteria population (y-axis) as a
    function of elapsed time steps (x-axis) on the same plot. You might find
    the helper function make_two_curve_plot helpful

    Args:
        num_bacteria (int): number of ResistantBacteria to create for
            the patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float int [0-1]): reproduction probability
        death_prob (float in [0, 1]): probability of a bacteria cell dying
        resistant (bool): whether the bacteria initially have
            antibiotic resistance
        mut_prob (float in [0, 1]): mutation probability for the
            ResistantBacteria cells
        num_trials (int): number of simulation runs to execute

    Returns: a tuple of two lists of lists, or two 2D arrays
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    def anaphase_list(
        chromosome: List[List[Tuple[int, int]]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        total = []
        resistant = []
        for test in chromosome:
            total.append([stamp[0] for stamp in test])
            resistant.append([stamp[1] for stamp in test])
        return total, resistant
    _pbar: bool = False
    try:
        _pbar = kwargs["progress_bar"]
    except:
        pass
    simulations = []
    iterator = tqdm.tqdm(range(num_trials)) if _pbar else range(num_trials)
    for _ in iterator:
        patient = TreatedPatient(
            ResistantBacteria.colony(
                num_bacteria,
                birth_prob,
                death_prob,
                resistant,
                mut_prob
            ),
            max_pop
        )
        results = patient.elapse_with_result(untreated_time)
        patient.set_on_antibiotic()
        results.extend(patient.elapse_with_result(treated_time))
        simulations.append(results)
    total, mutated = anaphase_list(simulations)
    n_sims = len(total[0])
    if do_plot:
        make_two_curve_plot(
            x_coords=[i for i in range(n_sims)],
            y_coords1=[calc_pop_avg(total, t) for t in range(n_sims)],
            y_coords2=[calc_pop_avg(mutated, t) for t in range(n_sims)],
            y_name1="Total",
            y_name2="Resistant",
            x_label="Time",
            y_label="Number of bacteria",
            title="Resistance Test"
        )
    return total, mutated


# When you are ready to run the simulations, uncomment the next lines one
# at a time
##total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
##                                                      max_pop=1000,
##                                                      birth_prob=0.3,
##                                                      death_prob=0.2,
##                                                      resistant=False,
##                                                      mut_prob=0.8,
##                                                      num_trials=50)

##total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
##                                                      max_pop=1000,
##                                                      birth_prob=0.17,
##                                                      death_prob=0.2,
##                                                      resistant=False,
##                                                      mut_prob=0.8,
##                                                      num_trials=50)
