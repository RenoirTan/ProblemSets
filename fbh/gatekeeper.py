"""
Module with functions which generates instructions on how to change to
orientation of caps in a group of people.
"""

from typing import Iterable, List, Tuple, TypeVar

__all__ = ["get_commands", "FRONT", "BACK", "HEAD"]

T = TypeVar("T")

# Hat type enum
FRONT: str = "F"
BACK: str = "B"
HEAD: str = "H"

def __count(group: Iterable[str]) -> Tuple[int, int]:
    """
    Count how many congregations of "front"-cappers and "back"-cappers there
    are in a slice of a queue. This function does not count each person
    in the slice, but each continuous group they form. For example, consider
    a queue like this:

        FFFBBBBFFFBBFFFF
    
    In this queue, there 3 groups of people in a row with front-facing caps
    and 2 groups of people in a row with back-facing caps. If you wanted to
    minimise the number of commands required to convert all the people into
    either F and B, you should choose to convert the cap type with the lowest
    number of groups. In this case, since there are only 2 groups of people
    with "B" caps, you should order those with backwards cap to flip them
    forwards.

    This function returns the number of groups of F and groups of B in a tuple.

    Parameters
    ==========
    group: Iterable[str]
        An object which iterates over a queue of people wearing caps.
    
    Returns
    =======
    Tuple[int, int]
        The first value is the number of forward-facing rows and the second
        value is the number of backward-facing rows.
    """
    _front: int = 0
    _back: int = 0
    _state: str = ""
    for _person in group:
        if _person == FRONT:
            if _state != FRONT:
                _front += 1
                _state = FRONT
        elif _person == BACK:
            if _state != BACK:
                _back += 1
                _state = BACK
        else:
            _state = _person
    return _front, _back

def __length(group: List[str]) -> int:
    """
    The length of a continuous sequence of people with the same cap. It takes
    the value of the first person's cap and checks the people that come after
    them. If a person which has a cap which does not match that of the first
    person, the length of the sequence is returned. For example, a group that
    looks like this:

        FFFFFFBFFFF
    
    will return a result of 6 when passed into this function. This is because
    there are 5 other people with front-facing caps after the first person.

    Parameters
    ==========
    List[str]
        A group of people with(out) caps.
    
    Returns
    =======
    int
        The length of the sequence of people with the same cap as the first
        person, including the first person.
    """
    _length: int = 0
    _state: str = group[0]
    for _person in group:
        if _person == _state:
            _length += 1
        else:
            break
    return _length

def __instruction(first: int, last: int) -> str:
    """
    Converts a range into a string instruction.
    """
    return "People in positions {} through {} please flip your caps.".format(
        first, last
    ) if last > first else "Person in position {} please flip your cap.".format(
        first
    )

def get_commands(group: Iterable[str]) -> List[str]:
    _group: List[str] = [person for person in group]
    _front, _back = __count(_group)
    _pest: str = BACK
    if _front < _back:
        _pest = FRONT
    # Let: '-' be the target orientation
    #      '+' be the deviant orientation (_pest)
    # Group: -------++++++++++++++++++++++++----++++-------
    #               ^ _index
    _index: int = 0
    _instructions: List[str] = []
    while _index < len(_group):
        if _group[_index] == _pest:
            _pest_length = __length(_group[_index:])
            _instructions.append(__instruction(_index, _index+_pest_length-1))
            _index += _pest_length
        else:
            _index += 1
    return _instructions