_USE_PBAR: bool = True # Set to False if you do not want the progress bars
_CUSTOM_TEST: bool = True

from typing import Iterator, List, Tuple, Union
import random
import math
if _USE_PBAR:
    from tqdm import tqdm

Scalar = Union[int, float]

def rollDie() -> int:
    return random.choice([1,2,3,4,5,6])

def rollDice() -> int:
    return rollDie() + rollDie()

ONGOING: int = 0
WON: int = 1
LOST: int = -1
DRAW: int = 2

class CrapsHand(Iterator[int]):
    """
    Simulates a craps hand. An object of this class is an iterator, with each
    iteration representing one roll of the dice. If you want to cut to the end
    of the hand, call the `play` method.
    """
    def __init__(self, passLine: bool = True) -> None:
        self.passLine: bool = passLine
        self.status: int = ONGOING
        self.point: int = 0
        self.lastRoll: int = 0
        self.rollCount: int = 0
    
    def __next__(self) -> int:
        if self.roll():
            return self.lastRoll
        else:
            raise StopIteration
    
    def done(self) -> bool:
        return self.status != ONGOING
    
    def won(self) -> bool:
        return self.status == WON
    
    def lost(self) -> bool:
        return self.status == LOST

    def tied(self) -> bool:
        return self.status == DRAW
    
    def __passRoll(self) -> bool:
        if self.done():
            return False
        self.rollCount += 1
        self.lastRoll = rollDice()
        if self.rollCount == 1:
            if self.lastRoll == 7 or self.lastRoll == 11:
                self.status = WON
            elif (
                self.lastRoll == 2 or
                self.lastRoll == 3 or
                self.lastRoll == 12):
                self.status = LOST
            else:
                self.point = self.lastRoll
                self.status = ONGOING
        else:
            if self.lastRoll == self.point:
                self.status = WON
            elif self.lastRoll == 7:
                self.status = LOST
        return True

    def __dPassRoll(self) -> bool:
        if self.done():
            return False
        self.rollCount += 1
        self.lastRoll = rollDice()
        if self.rollCount == 1:
            if self.lastRoll == 7 or self.lastRoll == 11:
                self.status = LOST
            elif self.lastRoll == 2 or self.lastRoll == 3:
                self.status = WON
            elif self.lastRoll == 12:
                self.status = DRAW
            else:
                self.status = ONGOING
                self.point = self.lastRoll
        else:
            if self.lastRoll == 7:
                self.status = WON
            elif self.lastRoll == self.point:
                self.status = LOST
        return True
    
    def roll(self) -> bool:
        if self.passLine:
            return self.__passRoll()
        else:
            return self.__dPassRoll()
    
    def play(self) -> int:
        while self.roll():
            pass
        return self.status

class CrapsGame(object):
    def __init__(self) -> None:
        self.passWins, self.passLosses = 0,0
        self.dpWins, self.dpLosses, self.dpPushes = 0,0,0
        self.uncategorized: int = 0

    def playHand(self):
        pResult: int = CrapsHand(passLine=True).play()
        if pResult == WON:
            self.passWins += 1
        elif pResult == LOST:
            self.passLosses += 1
        else:
            self.uncategorized += 1
        dpResult: int = CrapsHand(passLine=False).play()
        if dpResult == WON:
            self.dpWins += 1
        elif dpResult == LOST:
            self.dpLosses += 1
        elif dpResult == DRAW:
            self.dpPushes += 1
        else:
            self.uncategorized += 1
        #print("CrapsGame.playHand", pResult, dpResult)

    def playHands(self, hands: int) -> "CrapsGame":
        for _ in range(hands):
            self.playHand()
        return self

    def passResults(self):
        return (self.passWins, self.passLosses)

    def passGames(self) -> int:
        return sum(self.passResults())

    def dpResults(self):
        return (self.dpWins, self.dpLosses, self.dpPushes)

    def dpGames(self) -> int:
        return sum(self.dpResults())

class CrapsSimulation(object):
    def __init__(self, hands: int = 100) -> None:
        self.pWon: int = 0
        self.pLost: int = 0
        self.pGames: int = 0
        self.dpWon: int = 0
        self.dpLost: int = 0
        self.dpGames: int = 0
        self.hands: int = hands
    
    def reset(self) -> "CrapsSimulation":
        self.pWon: int = 0
        self.pLost: int = 0
        self.pGames: int = 0
        self.dpWon: int = 0
        self.dpLost: int = 0
        self.dpGames: int = 0
        return self

    def playOnce(self) -> "CrapsSimulation":
        game: CrapsGame = CrapsGame().playHands(self.hands)
        pRes = game.passResults()
        dpRes = game.dpResults()
        #print("CrapsSimulation.playOnce", pRes, dpRes)
        self.pWon += pRes[0]
        self.pLost += pRes[1]
        self.pGames += game.passGames()
        self.dpWon += dpRes[0]
        self.dpLost += dpRes[1]
        self.dpGames += game.dpGames()
        return self

    def playMany(self, games: int = 1000) -> "CrapsSimulation":
        for _ in range(games):
            self.playOnce()
        return self

    def simpleRoi(self) -> Tuple[float, float]:
        return (
            (self.pWon - self.pLost) / self.pGames,
            (self.dpWon - self.dpLost) / self.dpGames
        )

###############for testing purpose########################

def test1():
    #first test
    random.seed(0)
    c = CrapsGame()
    for i in range(100):
        c.playHand()
    print(c.passResults() == (52,48))
    print(c.dpResults() == (46,52,2))
    
    #second test
    random.seed(1000)
    d = CrapsGame()
    for i in range(200):
        d.playHand()
    print(d.passResults() == (96,104))
    print(d.dpResults() == (93,96,11))

#test1()

def testes(games: int = 100):
    c = CrapsGame()
    for i in range(games):
        c.playHand()
    print("After playing {} hands of Craps:".format(games))
    print("\tPass: {}".format(c.passResults()))
    print("\tDon't Pass: {}".format(c.dpResults()))

###############for testing purpose########################

def mean(X: List[Scalar]) -> float:
    '''Assumes X is a list of numbers
    returns the mean of X'''
    return sum(X)/len(X)

def variance(X: List[Scalar]) -> float:
    '''Assumes X is a list of numbers
    returns the variation of X'''
    avg: float = mean(X)
    var: float = 0.0
    for n in X:
        var += (n - avg)**2
    return var/len(X)

def stdDev(X: List[Scalar]) -> float:
    '''Assumes that X is a list of numbers.
    returns the standard deviation of X'''
    return math.sqrt(variance(X))


def crapsSim(handsPerGame: int = 100, numGames: int = 1000) -> None:
    '''Assumes handsPerGame and numGames are ints > 0
    Play numGames games of handsPerGames hands; print results'''
    pRoi: List[float] = []
    dpRoi: List[float] = []
    it = range(numGames)
    if _USE_PBAR:
        it = tqdm(it, desc="Simulating...", total=numGames)
    for _ in it:
        simulator = CrapsSimulation(handsPerGame).playOnce()
        #print(simulator.pWon, simulator.pLost, simulator.dpWon, simulator.dpLost)
        sRoi = simulator.simpleRoi()
        pRoi.append(sRoi[0])
        dpRoi.append(sRoi[1])
    print("Pass: Mean ROI = {}% Std. Dev. = {}%".format(
        mean(pRoi)*100,
        stdDev(pRoi)*100
    ))
    print("Don't Pass: Mean ROI = {}% Std. Dev. = {}%".format(
        mean(dpRoi)*100,
        stdDev(dpRoi)*100
    ))

if __name__ == "__main__" and _CUSTOM_TEST:
    testes(100)
    setups: List[Tuple[int, int]] = [
        (20, 10), # Control
        (20, 1000), # Change number of games
        (1000, 10), # Change number of hands per game
        (1000, 1000) # Change both variables
    ]
    for s in setups:
        print("Trying {1} games of {0} hands.".format(*s))
        crapsSim(*s)
    testes(100)

#With your results, is one of these a better bet than the other? Is either a good bet?
'''
(with random.seed(0))

With my results, there is inconclusive evidence suggesting that either bet
is better than the other.

When simulating with games consisting of less than 100 hands (swapping 
between playing pass and don't pass), one noticeable trend is that if the
shooter chooses to pass the line, they spend a better albeit marginal
chance of winning (see the table below).

However, when the number of hands per game to 1000, not passing the line
becomes slightly more advantageous.

| Games | Hands/Game | Pass Mean | Don't Pass Mean |
| ----- | ---------- | --------- | --------------- |
|   1   |    100     |   6.00%   |      -7.00%     |
|   10  |     20     |  -9.00%   |     -14.00%     |
|  1000 |     20     |  -1.05%   |      -2.26%     |
|   10  |    1000    |  -1.38%   |      -0.02%     |
|  1000 |    1000    |  -1.49%   |      -1.42%     |

In addition, as the seed for the random number generator is preset, it means
that whether the shooter wins a hand or not is also determined by the order
each game is played. Hence, the reliability of the results generated and
collated in the table above is questionable at best

When playing without a preset seed, the difference between passing and not
passing the line comes within 0.1 percentage points for many games with many
hands played per game.

Therefore, neither bet is better than the other.

As seen in the table, the mean rate of investment is negative. Furthermore,
the standard deviation for these games hover at around 3%. Therefore, the
chance of winning a game is less than 50%, and even if you do win, it will
not offset any losses previously made. Therefore playing craps, regardless
of your decision over whether to pass the line, is a bad idea.
'''
