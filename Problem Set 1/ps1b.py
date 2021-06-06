###########################
# Y4CEP Problem Set 1: Transportation
# Name: Renoir Tan

from typing import List


#================================
# Part B: Silver Eggs
#================================

# Problem 1
# This problem is basically the change making algorithm but with eggs.
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always a egg of value 1.

    Time complexity: O(E*T)
    Space complexity: O(E*T)
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    # Basically change-making problem but with variable names updated
    # to match the context.

    # MATRIX
    # Vertical: Consider each coin downwards in ascending value
    # Hortizontal: Coins used to reach that value
    # [[inf, inf, inf, ...],
    #  [0  ,   0,   0, ...],
    #  ...]
    def _change_making_matrix(target: int, eggs: int) -> List[List[int]]:
        matrix = [[0 for _ in range(target+1)] for _ in range(eggs+1)]
        for i in range(1, target+1):
            matrix[0][i] = float("inf") # Maximum possible
        return matrix
    # egg_weights already sorted, so no need to do:
    # egg_weights = sorted(egg_weights)
    matrix = _change_making_matrix(target_weight, len(egg_weights))
    # Consider each egg from smallest to largest
    for egg in range(1, len(egg_weights)+1):
        for subvalue in range(1, target_weight+1):
            # If you can represent one subvalue as a single egg,
            # it means that you have gotten the best result for that
            # subproblem.
            if egg_weights[egg-1] == subvalue:
                matrix[egg][subvalue] = 1
            # If the weight of the egg is more than the sub weight
            # use the solution used for the sub value without considering
            # this egg.
            # Because this egg would have exceeded the solution.
            elif egg_weights[egg-1] > subvalue:
                matrix[egg][subvalue] = matrix[egg-1][subvalue]
            # If there is room for the next egg
            # Use the more efficient solution:
            # 1. Use the previous solution for making subvalue
            #    without this egg
            # 2. Use the previous solution but without using the
            #    previous egg + 1 for this egg
            else:
                matrix[egg][subvalue] = min(
                    matrix[egg - 1][subvalue],
                    1 + matrix[egg][subvalue - egg_weights[egg - 1]]
                )
    # Return the result which has considered all the eggs and is
    # at `target`.
    return matrix[-1][-1]

# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()
