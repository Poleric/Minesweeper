import numpy as np
from typing import Iterator

# Optional
SCIPY = True
try:
    from scipy.signal import convolve2d
except ImportError:
    SCIPY = False

CoordT = tuple[int, int]


class Board:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=np.int8)  # create a board with zeros

    def get_adjacent_coords(self, coordinate: CoordT) -> Iterator[CoordT]:
        """Get the coordinates of row and column of adjacent cell given a coordinate."""

        for drow in range(-1, 2):  # -1, 0, 1
            for dcol in range(-1, 2):  # -1, 0, 1
                row, col = coordinate[0] + drow, coordinate[1] + dcol  # adjacent cell

                if 0 <= row < self.height and 0 <= col < self.width and (drow, dcol) != (0, 0):
                    yield row, col

    def coords_to_number(self, coordinate: CoordT) -> int:
        """Translate a 2d coordinate into a flatten index"""

        return coordinate[0] * self.width + coordinate[1]

    def number_to_coords(self, number: int) -> CoordT:
        """Translate the index of the flatten list into a 2d coordinate """

        return divmod(number, self.width)


class MinesweeperBoard(Board):
    MINES_EDGE_DETECTION_KERNEL = np.ones((3, 3), dtype=np.int8) * -1

    def __init__(self, width: int, height: int, mines: int | float, *, starting_tile: CoordT = None, seed=None):
        """Create a minesweeper board

        :param: width <int> - the width of the board
        :param: height <int> - the height of the board
        :param: mines <int> - the number of mines in the board
                      <float> (0, 1) - the mine density of the board, from 0 to 1
        :param: starting_tile <tuple[int]> - the coordinate of the starting tile, usually to exclude mines from
        :param: seed <int> - the seed used for the rng
        """

        super().__init__(width, height)
        self.rng = np.random.default_rng(seed)

        if 0 < mines < 1:  # convert mine density to mines
            self.mines = max(1, int(mines*self.width*self.height))  # min 1 mine
        else:
            self.mines = mines

        assert isinstance(mines, int), "Mines can only be an integer, or a decimal number between 0 and 1 to represent mine density"  # make sure mines is an integer

        if not starting_tile:
            starting_tile = (self.rng.integers(0, self.height-1), self.rng.integers(0, self.width-1))
        self.starting_tile: CoordT = starting_tile

        self.generate_board()

    def get_randomized_coords_for_mines(self, n: int) -> Iterator[CoordT]:
        """Get randomized coordinates for mine placements, will exclude starting position and the adjacent tiles to it."""

        cell_num = list(range(self.width * self.height))  # get position in 1d space

        cell_num.remove(self.coords_to_number(self.starting_tile))  # exclude starting position
        for coord in self.get_adjacent_coords(self.starting_tile):  # exclude the tiles adjacent to the starting position
            cell_num.remove(self.coords_to_number(coord))

        # get random position and convert to 2d coordinates
        yield from map(self.number_to_coords, self.rng.choice(cell_num, size=n, replace=False))

    def generate_board(self) -> None:
        if SCIPY:
            self.convol_generation()
        else:
            self.linear_generation()

    def linear_generation(self) -> None:
        """Generate a board with mines and markings. For loops implementation"""

        for coord in self.get_randomized_coords_for_mines(self.mines):
            self.board[coord] = -1
            for coord in self.get_adjacent_coords(coord):
                if self.board[coord] != -1:
                    self.board[coord] += 1

    def convol_generation(self) -> None:  # 4x faster than for loops implementation, not that it really matters, it's just cool
        """Generate a board with mines and markings. Matrix convolution implementation"""

        # add mines
        for coord in self.get_randomized_coords_for_mines(self.mines):
            self.board[coord] = -1

        mines_positions = self.board == -1
        marked_tiles = convolve2d(self.board, self.MINES_EDGE_DETECTION_KERNEL, mode="same")
        # for spot that is mine, set -1, else keep
        self.board = np.where(mines_positions, -1, marked_tiles)
