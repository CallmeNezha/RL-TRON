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
        else:
            raise ValueError("Invalid Parameter")

    def move(self, direction:Tuple[int,int]) -> Tuple[int,int]:
        if direction == self._counter_direction(self._direction):
            pass
        else:
            self._direction = direction
        while len(self._tail) >= self._tail_len:
            self._tail.pop()
        #!while
        self._tail.appendleft(self._position)
        self._position = (self.position[0] + self._direction[0],
                          self.position[1] + self._direction[1])
        return self._position

Box = namedtuple("Box", "shape, dtype, high, low")
Discrete = namedtuple("Discrete", "dtype, n, vals")

class Tron:
    def __init__(self):
        self._grid = Grid(width=20)
        self._observation_space = Box(shape=(2, (self._grid.width+1)*2, self._grid.width+1), 
                                          dtype=np.int32, high=2, low=0)
        self._action_space = Discrete(dtype=np.int32, n=4, vals=[Grid.WEST, Grid.EAST, Grid.SOUTH, Grid.NORTH])
        self._done = True

    @property
    def observation_space(self) -> Box:
        return self._observation_space
    
    @property
    def action_space(self) -> Discrete:
        return self._action_space

    @property
    def done(self) -> bool:
        return self._done

    def reset(self):
        self._done = False
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
        return np.array(states)

    def step(self, action0, action1):
        if self._done:
            raise RuntimeError("Enviroment is done!")
        
        def action_direction(act):
            if act == 0:
                return Grid.WEST
            elif act == 1:
                return Grid.NORTH
            elif act == 2:
                return Grid.EAST
            elif act == 3:
                return Grid.SOUTH
            else:
                raise ValueError("NO SUCH OPTION")

        self._players[0].move(action_direction(action0))
        self._players[1].move(action_direction(action1))

        def out(position) -> bool:         
            if position[0] < 0 or position[0] > self._grid.width:
                return True
            elif position[1] < 0 or position[1] > self._grid.width:
                return True
            else:
                return False


        p1 = self._players[0].position
        p2 = self._players[1].position

        crash0 = False
        crash1 = False
        if out(p1):
            crash0 = True
        if out(p2):
            crash1 = True

        self._grid.clear()
        if not crash0:
            self._grid.add_player(self._players[0])
        if not crash1:
            self._grid.add_player(self._players[1])

        if not crash0 and self._grid[p1] > 2:
            crash0 = True
        if not crash1 and self._grid[p2] > 2:
            crash1 = True

        reward0 = -1 if crash0 else 1
        reward1 = -1 if crash1 else 1

        self._done = crash0 or crash1
        if self._done:
            return None, [reward0, reward1], True
        else:
            states = self.current_state()
            return states, [reward0, reward1], False

class ConsoleRender:
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'

    def __init__(self):
        pass

    def render(self, env:Tron):
        if env.done:
            return
        canvas_width = env._grid.width + 1
        grid = [[' '.join([self.CGREYBG,self.CEND])] * canvas_width for _ in range(canvas_width)]
        for i,p in enumerate(env._players):
            if 0 == i:
                head = '2'.join([self.CBLUEBG,self.CEND])
                tail = '1'.join([self.CBLUEBG,self.CEND])
            elif 1 == i:
                head = '2'.join([self.CYELLOWBG,self.CEND])
                tail = '1'.join([self.CYELLOWBG,self.CEND])
            else:
                raise RuntimeError("Impossible to have three players!")
            x, y = p.position
            grid[y][x] = head
            for t in p.tail:
                x, y = t
                grid[y][x] = tail
            #!for
        #!for
        # lets print it
        for row in grid:
            print(''.join(row))
