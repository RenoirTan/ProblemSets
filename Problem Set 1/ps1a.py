###########################
# Y4CEP Problem Set 1: Transportation
# Name: Renoir Tan

from ps1_partition import get_partitions
import time

from typing import List, Dict, Tuple

# My functions

def sort_animals(
    animals: Dict[str, int],
    descending: bool=True
) -> List[Tuple[str, int]]:
    """
    Sort the animals based on their weight and dump the result into a list of
    tuples (because lists are ordered but dicts are not).

    I'm using the word animals because there are inconsistencies as to
    whether to call them sheep or cows.

    Unfortunately, Python doesn't have macros or inline functions.

    Parameters
    ==========
    animals: Dict[str, int]
        Dictionary of animals with their name as the key and their weight as
        the value.

    descending: bool = true
        Whether to sort in descending order.
    
    Returns
    =======
    List[Tuple[str, int]]
        The sorted animals.
    """
    return [
        (k, v)
        for k, v
        in sorted(
            animals.items(),
            key=lambda item: item[1],
            reverse=descending
        )
    ]

#================================
# Part A: Transporting Space Cows
#================================

# Problem 1
def load_sheep(filename: str):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated sheep name, weight pairs, and return a
    dictionary containing sheep names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of sheep name (string), weight (int) pairs
    """
    # TODO: Your code here
    sheep_data = open(filename, "r")
    sheep = {}
    for line in sheep_data:
        s = line.rstrip("\n").split(",")
        name, weight = s[0], s[1]
        sheep[name] = int(weight)
    return sheep

# Problem 2
def greedy_sheep_transport(
    sheep: Dict[str, int],
    limit: int = 10
) -> List[List[str]]:
    """
    Uses a greedy heuristic to determine an allocation of sheep that attempts to
    minimize the number of spaceship trips needed to transport all the sheep. The
    returned allocation of sheep may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another sheep, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining sheep

    Does not mutate the given dictionary of sheep.

    Time complexity: O(n3)
    Space complexity: O(n)

    Parameters:
    sheep - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of sheep
    transported on a particular trip and the overall list containing all the
    trips
    """ 
    # TODO: Your code here
    trips = []
    # Sheep who have yet to be put into the spaceship
    # Sorted from largest to smallest so that we can access the largest sheep
    # first.
    _sheep = sort_animals(sheep) # O(n log n)
    while len(_sheep) > 0 and limit > _sheep[-1][1]:
        remaining = limit # How much capacity there is left
        trip = [] # This trip
        # Fill trip with the largest sheep that can board this trip one by one.
        while len(_sheep) > 0 and remaining >= _sheep[-1][1]:
            for _s in _sheep:
                if _s[1] <= remaining:
                    trip.append(_s[0])
                    remaining -= _s[1]
                    _sheep.remove(_s)
                    break
        trips.append(trip)
    return trips

# Problem 3
def brute_force_sheep_transport(
    cows: Dict[str, int],
    limit: int=10
) -> List[List[str]]:
    """
    Finds the allocation of sheep that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the sheep can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    sheep - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of sheep
    transported on a particular trip and the overall list containing all the
    trips
    """
    # TODO: Your code here
    def _combination_is_valid(combination: List[List[str]]) -> bool:
        for trip in combination:
            weight = 0
            for cow in trip:
                weight += cows[cow]
            if weight > limit:
                return False
        return True
    names = [cow for cow in cows.keys()]
    trips = []
    minimum = float("inf")
    for combination in get_partitions(names):
        if _combination_is_valid(combination):
            length = len(combination)
            if length < minimum:
                trips = combination
                minimum = length
    return trips
        
# Problem 4
def compare_sheep_transport_algorithms():
    """
    Using the data from ps1_sheep_data.txt and the specified weight limit, run your
    greedy_sheep_transport and brute_force_sheep_transport functions here. Use the
    default weight limits of 10 for both greedy_sheep_transport and
    brute_force_sheep_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    # TODO: Your code here
    start = time.time()
    sheep = load_sheep("ps1_sheep_data.txt")
    # print(sheep)

    gtrips = greedy_sheep_transport(sheep)
    gtrips_time = time.time()
    print(f"Greedy algorithm result: {gtrips}")
    print("Greedy algorithm length: %d" % (len(gtrips),))
    print("Time to complete greedy: %f seconds" % (gtrips_time-start,))

    btrips = brute_force_sheep_transport(sheep)
    btrips_time = time.time()
    print(f"Brute force result: {btrips}")
    print("Brute force length: %d" % (len(btrips),))
    print(
        "Time to complete brute force: %f seconds" %
        (btrips_time-gtrips_time-start,)
    )
    
    end = time.time()
    print("%f seconds used." % (end-start,))

if __name__ == "__main__":
    compare_sheep_transport_algorithms()
