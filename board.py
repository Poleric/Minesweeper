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
    def __init__(self, rows: int, columns: int):
        self.rows = rows
        self.columns = columns
        self.board = self.create_board()

    def __repr__(self):
        return repr(self.board)

    def create_board(self) -> np.ndarray:
        """Method that is always called to initialize a board
        Define this when subclassing to produce different kind of board"""

        return np.zeros((self.rows, self.columns), dtype=np.int8)  # create a board with zeros

    def get_adjacent_coords(self, coordinate: CoordT) -> Iterator[CoordT]:
        """Get the coordinates of row and column of adjacent cell given a coordinate."""

        for drow in range(-1, 2):  # -1, 0, 1
            for dcol in range(-1, 2):  # -1, 0, 1
                row, col = coordinate[0] + drow, coordinate[1] + dcol  # adjacent cell

                if 0 <= row < self.rows and 0 <= col < self.columns and (drow, dcol) != (0, 0):
                    yield row, col

    def coords_to_number(self, coordinate: CoordT) -> int:
        """Translate a 2d coordinate into a flatten index"""

        return coordinate[0] * self.columns + coordinate[1]

    def number_to_coords(self, number: int) -> CoordT:
        """Translate the index of the flatten list into a 2d coordinate """

        return divmod(number, self.columns)


class MinesweeperBoard(Board):
    MINES_EDGE_DETECTION_KERNEL = np.ones((3, 3), dtype=np.int8) * -1

    def __init__(self, rows: int, columns: int, mines: int | float, *, starting_tile: CoordT = None, seed=None):
        """Create a minesweeper board

        :param: width <int> - the width of the board
        :param: height <int> - the height of the board
        :param: mines <int> - the number of mines in the board
                      <float> (0, 1) - the mine density of the board, from 0 to 1
        :param: starting_tile <tuple[int]> - the coordinate of the starting tile, usually to exclude mines from
        :param: seed <int> - the seed used for the rng
        """

        super().__init__(rows, columns)
        self.rng = np.random.default_rng(seed)

        if 0 < mines < 1:  # convert mine density to mines
            self.mines = max(1, int(mines * self.rows * self.columns))  # min 1 mine
        else:
            self.mines = mines

        assert isinstance(mines, int), "Mines can only be an integer, or a decimal number between 0 and 1 to represent mine density"  # make sure mines is an integer

        if not starting_tile:
            starting_tile = (self.rng.integers(0, self.columns - 1), self.rng.integers(0, self.rows - 1))
        self.starting_tile: CoordT = starting_tile

        self.generate_board()

    def get_randomized_coords_for_mines(self, n: int) -> Iterator[CoordT]:
        """Get randomized coordinates for mine placements, will exclude starting position and the adjacent tiles to it."""

        cell_num = list(range(self.rows * self.columns))  # get position in 1d space

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

    def get_all_mine_coords(self) -> Iterator[CoordT]:
        return zip(*np.where(self.board == -1))


class TileBoard(Board):
    def __init__(self, rows: int, columns: int):
        """Used to generate visual board for playing

        Created to store tile states (open, flagged) when playing
        self.board: store tile values as the game goes on
        self.opened: tile is opened or not
        self.flagged: tile is flagged or not
        """

        super().__init__(rows, columns)

        # Create arrays with all 0, switch to 1 to denote truthy
        self.opened: np.ndarray = self.create_bool_board()
        self.flagged: np.ndarray = self.create_bool_board()

    def create_bool_board(self) -> np.ndarray:
        return np.zeros((self.rows, self.columns), dtype=bool)

    def is_opened(self, coord: CoordT) -> bool:
        return self.opened[coord]

    def is_flagged(self, coord: CoordT) -> bool:
        return self.flagged[coord]

    def get_all_opened_coords(self) -> Iterator[CoordT]:
        return zip(*np.where(self.opened))

    def get_all_flagged_coords(self) -> Iterator[CoordT]:
        return zip(*np.where(self.flagged))

    def flag_tile(self, coord: CoordT) -> None:
        self.flagged[coord] = True

    def unflag_tile(self, coord: CoordT) -> None:
        self.flagged[coord] = False


class GameBoard(TileBoard):
    """Add actual board alongside the display board and game logic."""
    def __init__(self, rows: int, columns: int, mines: int):
        super().__init__(rows, columns)

        self.mines = mines
        self.minesweeper: MinesweeperBoard | None = None

    def start_game(self, coord: CoordT) -> None:
        self.generate_minesweeper_board(coord)

    def click_tile(self, coord: CoordT) -> None:
        """When a tile is clicked, do stuff based on the tile state"""
        if self.minesweeper is None:  # not initialised yet
            self.start_game(coord)

        if self.is_flagged(coord):
            return

        if not self.is_opened(coord):
            self.open_tile(coord)

            if self.board[coord] == 0:  # no mines around
                self.cascade_tile(coord)
        else:
            if self.can_look_around(coord):
                self.cascade_tile(coord)

        if self.board[coord] == -1:  # pressed mine
            self.game_over(coord)

        if self.is_win():
            self.win()

    def game_over(self, coord: CoordT) -> None:
        """Reveal all mines"""
        mines = self.minesweeper.board == -1
        self.board = np.where(mines, -1, self.board)

    def win(self) -> None:
        """Flag all mines"""
        for mine_coord in self.minesweeper.get_all_mine_coords():
            self.flag_tile(mine_coord)

    def cascade_tile(self, coord) -> None:
        for coord in self.get_adjacent_unopened_and_unflagged_coords(coord):
            self.click_tile(coord)

    def generate_minesweeper_board(self, starting_tile: CoordT, seed=None) -> None:
        self.minesweeper = MinesweeperBoard(self.rows, self.columns, self.mines, starting_tile=starting_tile, seed=seed)

    def is_win(self) -> bool:
        """Check if the number of non opened cells is equal to number of mines"""
        return np.count_nonzero(self.opened == 0) == self.mines

    def open_tile(self, coord: CoordT) -> None:
        """"Open" a tile and set its value to its respective tile type. Ideally called once per tile."""
        self.board[coord] = self.minesweeper.board[coord]
        self.opened[coord] = True

    def get_adjacent_unopened_and_unflagged_coords(self, coord: CoordT) -> Iterator[CoordT]:
        """Used to cascade tiles and "look around" tiles"""

        for coord in self.get_adjacent_coords(coord):
            if self.is_opened(coord) or self.is_flagged(coord):
                continue
            yield coord

    def get_adjacent_flagged_coords(self, coord: CoordT) -> Iterator[CoordT]:
        for coord in self.get_adjacent_coords(coord):
            if self.is_flagged(coord):
                yield coord

    def can_look_around(self, coord: CoordT) -> bool:
        """See if the number on the tile is less than the number of adjacent flags"""
        return self.board[coord] <= len(tuple(self.get_adjacent_flagged_coords(coord)))

    def get_wrong_flag_coords(self) -> Iterator[CoordT]:
        mines = self.minesweeper.board == -1

        # using an A AND NOT B logic gate
        #  flag   mines   out
        #   0      0       0
        #   0      1       0
        #   1      0       1
        #   1      1       0
        return zip(*np.where(self.flagged & ~mines))

    def get_unflagged_mine_coords(self) -> Iterator[CoordT]:
        mines = self.minesweeper.board == -1

        # using a NOT A AND B logic gate
        #  flag   mines   out
        #   0      0       0
        #   0      1       1
        #   1      0       0
        #   1      1       0
        return zip(*np.where(~self.flagged & mines))
