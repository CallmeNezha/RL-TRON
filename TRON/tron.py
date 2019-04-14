import numpy as np
from copy import copy
from collections import deque
from typing import Tuple
from collections import namedtuple

CLU_COLOR   = (245,207,82)
FLYNN_COLOR = (126,255,253)

class Grid:
    """
    Grid has custom number of columns and rows .. Playground of TRON GAME
    0------> x
    |      |
    |      |
    V----(1,1)
    y 
    """
    WEST    = (-1,  0)
    EAST    = ( 1,  0)
    SOUTH   = ( 0,  1)
    NORTH   = ( 0, -1)

    def __init__(self, width=20):
        self._width = width
        self._vertices = np.zeros((width+1, width+1), dtype=np.int32)

    @property
    def width(self) -> int:
        return self._width

    def __getitem__(self, key) -> int:
        return self._vertices[key]

    def clear(self):
        self._vertices = np.zeros(self._vertices.shape, dtype=self._vertices.dtype)

    def add_player(self, player:'Player'):
        """
        Stack player onto Grid
        """
        for t in player.tail:
            self._vertices[t] += 1
        #!for
        self._vertices[player.position] += 2



class Player:
    """
    Are you SAM ? Or GLU ?
    """
    def __init__(self, position=(0,0), direction=Grid.EAST, tail=4):
        self._position = position
        self._direction = direction
        self._tail_len = tail
        self._tail = deque()

        self.name = ''
        self.color = (255,255,255)

    @property
    def position(self) -> Tuple[int,int]:
        return self._position

    @property
    def direction(self) -> Tuple[int,int]:
        return self._direction

    @property
    def tail(self):
        return self._tail


    def _counter_direction(self, direction:Tuple[int,int]) -> Tuple[int,int]:
        if direction == Grid.EAST:
            return Grid.WEST
        elif direction == Grid.WEST:
            return Grid.EAST
        elif direction == Grid.NORTH:
            return Grid.SOUTH
        elif direction == Grid.SOUTH:
            return Grid.NORTH

    def move(self, direction:Tuple[int,int]) -> Tuple[int,int]:
        if direction == self._counter_direction(self._direction):
            pass
        else:
            self._direction = direction
        while len(self._tail) >= self._tail_len:
            self._tail.pop()
        #!while
        self._tail.appendleft(self._position)
        self._position = [self.position[0] + self._direction[0],
                          self.position[1] + self._direction[1]]
        return self._position

Box = namedtuple("Box", "shape, dtype, high, low")
Discrete = namedtuple("Discrete", "dtype, n")

class Tron:
    def __init__(self):
        self._grid = Grid(width=20)
        self._env_observation_space = Box(shape=(2, self._grid, self._grid), dtype=np.int32, high=2, low=0)
        self._action_space = Discrete(dtype=np.int32, n=4)
        player0 = Player()
        player0.name = 'SAM'
        player0.color = FLYNN_COLOR
        player1 = Player((self._grid.width, self._grid.width), Grid.WEST)
        player1.name = 'CLU'
        player1.color = CLU_COLOR
        self._players = [player0, player1]

    @property
    def env_observation_space(self) -> Box:
        return self._env_observation_space
    
    @property
    def action_space(self) -> Discrete:
        return self._action_space

    def reset(self):
        self._grid.clear()
        player0 = Player()
        player0.name = 'SAM'
        player0.color = FLYNN_COLOR
        player1 = Player((self._grid.width, self._grid.width), Grid.WEST)
        player1.name = 'CLU'
        player1.color = CLU_COLOR
        self._players = [player0, player1]
        return self.current_state()

    def current_state(self):
        states = []
        for tt in [(1,0), (0,1)]:
            op_grid = Grid(self._grid.width)
            op_grid.add_player(self._players[tt[0]])
            my_grid = Grid(self._grid.width)
            my_grid.add_player(self._players[tt[1]])
            state = np.vstack([op_grid._vertices, my_grid._vertices])
            states.append(state)
        #!for
        return states

    def step(self, action0, action1):
        self._players[0].move(action0)
        self._players[1].move(action1)

        def out(position) -> bool:         
            if position[0] < 0 or position[0] > self._grid.width:
                return True
            elif position[1] < 0 or position[1] > self._grid.width:
                return True
            else:
                return False

        self._grid.clear()
        self._grid.add_player(self._players[0])
        self._grid.add_player(self._players[1])

        p1 = self._players[0].position
        p2 = self._players[1].position
        crash0 = False
        crash1 = False
        if self._grid[p1] > 2 or out(p1):
            crash0 = True
        if self._grid[p2] > 2 or out(p2):
            crash1 = True

        reward0 = -1 if crash0 else 1
        reward1 = -1 if crash1 else 1

        done = crash0 or crash1
        if done:
            return [None,None], [reward0, reward1], True
        else:
            states = self.current_state()
            return states, [reward0, reward1], False

