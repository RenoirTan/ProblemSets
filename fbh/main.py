from typing import List
import gatekeeper as gk
import random

def __queue(length: int, front: int=5, back: int=5, bare: int=1) -> List[str]:
    _queue: List[str] = []
    _front = front
    _back = _front + back
    _bare = _back + bare
    for _ in range(length):
        _rngesus = random.randint(1, _bare)
        if _rngesus <= _front:
            _queue.append(gk.FRONT)
        elif _rngesus <= _back:
            _queue.append(gk.BACK)
        else:
            _queue.append(gk.HEAD)
    return _queue

def __queue_str(queue: List[str]) -> str:
    if len(queue) <= 0:
        return ""
    _output: str = ""
    for _person in queue:
        _output += "{} ".format(_person)
    return _output[:-1]

def main() -> None:
    _queue: List[str] = __queue(20)
    print("Queue:", __queue_str(_queue))
    _commands: List[str] = gk.get_commands(_queue)
    for _command in _commands:
        print(_command)

if __name__ == "__main__":
    main()