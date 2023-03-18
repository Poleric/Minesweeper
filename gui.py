from tkinter import *
from board import Board, MinesweeperBoard
import numpy as np
import sys
from datetime import timedelta

from dataclasses import dataclass

from typing import Callable, Iterator, Literal

LEFT_CLICK = "<Button-1>"
RIGHT_CLICK = "<Button-2>" if sys.platform == 'darwin' else "<Button-3>"
CoordT = tuple[int, int]


@dataclass
class Tile:
    button: Button

    value: int = None  # Values follows minesweeper board, eg -1: mines, 0: 0 mines adjacent. None means unopened
    is_flagged: bool = False

    @property
    def is_opened(self) -> bool:
        return self.value is not None


class BoardGUI(Frame, Board):

    def __init__(self, tk, width: int, height: int, mines: int, seed=None):
        super().__init__(tk)

        self.width = width
        self.height = height

        self.board: np.ndarray[Tile] = np.empty((height, width), dtype=object)

        self.mines = mines
        self.seed = seed
        self.minesweeper: MinesweeperBoard = None

        self.PLAYING = True  # change to False if lose

        self.images = {
            "closed": PhotoImage(file="./img/closed.png"),
            "opened": PhotoImage(file="./img/opened.png"),
            "flag": PhotoImage(file="./img/flag.png"),
            "flag_wrong": PhotoImage(file="./img/flag_wrong.png"),
            "mine": PhotoImage(file="./img/mine.png"),
            "mine_red": PhotoImage(file="./img/mine_red.png"),
            "tile": [
                PhotoImage(file=f"./img/{n}.png") for n in range(0, 9)
            ]
        }

        self.time = 0
        self.timer = Label(self, text="0:00:00", font=("Arial", 12))
        self.timer.pack()

        self.board_frame = Frame(self)
        self.display_board()

    def display_board(self):
        for row in range(self.height):
            for col in range(self.width):
                # Setup button
                button = Button(self.board_frame, image=self.images["closed"], width=25, height=25)
                button.bind(LEFT_CLICK, self._on_left_click_wrapper(row, col))
                button.bind(RIGHT_CLICK, self._on_right_click_wrapper(row, col))
                button.grid(row=row, column=col)

                tile = Tile(button=button)

                self.board[row, col] = tile
        self.board_frame.pack()

    def generate_minesweeper_board(self, starting_tile: CoordT):
        self.minesweeper = MinesweeperBoard(self.width, self.height, self.mines, starting_tile=starting_tile, seed=self.seed)

    def get_adjacent_flag_coords(self, coord: CoordT) -> Iterator[CoordT]:
        """Get coordinates of surrounding flags"""

        for coord in self.get_adjacent_coords(coord):
            if self.board[coord].is_flagged:
                yield coord

    def get_cascadeable_cells(self, coord: CoordT) -> Iterator[CoordT]:
        """Returns adjacent unopened cells coordinates to open / cascade"""

        for coord in self.get_adjacent_coords(coord):
            if not self.board[coord].is_opened:
                yield coord

    def can_look_around(self, coord: CoordT) -> bool:
        """Check if numbered tiles can open adjacent tiles depending on number of flags. TODO: find the right terminology for this"""

        tile: Tile = self.board[coord]
        if not tile.is_opened:
            return False

        number_of_adj_flags = len(list(self.get_adjacent_flag_coords(coord)))
        if (tile.value - number_of_adj_flags) <= 0:
            return True

        return False

    def look_around(self, coord: CoordT) -> None:

        adj_coords = self.get_adjacent_coords(coord)
        adj_flagged = self.get_adjacent_flag_coords(coord)

        adj_not_flagged = set(adj_coords) - set(adj_flagged)
        for coord in adj_not_flagged:
            if not self.board[coord].is_opened:
                self.open_tile(coord)

    def uncover_tile(self, tile: Tile, tile_type: int) -> None:
        """Set tile value and image"""

        tile.value = tile_type

        if tile_type == -1:  # is mine
            image = self.images["mine_red"]
        else:
            image = self.images["tile"][tile_type]
        tile.button.configure(image=image)

    def flag_tile(self, tile: Tile) -> None:
        tile.is_flagged = True

        image = self.images["flag"]
        tile.button.configure(image=image)

    def unflag_tile(self, tile: Tile) -> None:
        tile.is_flagged = False

        image = self.images["closed"]
        tile.button.configure(image=image)

    def _on_left_click_wrapper(self, row, col) -> Callable:
        return lambda _: self.on_left_click((row, col))

    def _on_right_click_wrapper(self, row, col) -> Callable:
        return lambda _: self.on_right_click((row, col))

    def on_left_click(self, coord: CoordT):
        if not self.PLAYING:
            return

        if not self.minesweeper:
            self.generate_minesweeper_board(coord)
            self.start_timer()

        tile: Tile = self.board[coord]
        if tile.is_flagged:
            return

        if not tile.is_opened:
            self.open_tile(coord)
        elif self.can_look_around(coord):
            self.look_around(coord)

        if self.check_win():
            self.win()

    def on_right_click(self, coord: CoordT):
        if not self.PLAYING:
            return

        tile: Tile = self.board[coord]
        if tile.is_opened:
            return

        if not tile.is_flagged:
            self.flag_tile(tile)
        else:
            self.unflag_tile(tile)

    def open_tile(self, coord: CoordT) -> None:
        tile: Tile = self.board[coord]

        self.uncover_tile(tile, tile_type=self.minesweeper.board[coord])
        if tile.value == -1:  # is mine
            self.game_over()
            return

        if tile.value == 0:
            for coord in self.get_cascadeable_cells(coord):
                self.open_tile(coord)

    def game_over(self):
        self.PLAYING = False
        self.show_stray_mines()
        self.show_wrong_flag()

    def show_stray_mines(self):
        """Show mines that isn't flagged"""

        for coord in zip(*np.where(self.minesweeper.board == -1)):  # get all mine coords
            tile: Tile = self.board[coord]

            if tile.is_flagged or tile.is_opened:  # flagged / mine that was clicked on
                continue

            image = self.images["mine"]
            tile.button.configure(image=image)

    def show_wrong_flag(self):
        """Show tiles that have been falsely flagged"""

        is_flagged = np.vectorize(lambda t: t.is_flagged)
        flags = is_flagged(self.board)

        for coord in zip(*np.where(flags)):
            if self.minesweeper.board[coord] != -1:
                image = self.images["flag_wrong"]
                self.board[coord].button.configure(image=image)

    def flag_all_unflagged(self):
        """Flag all unflagged tiles when win"""

        is_unopened = np.vectorize(lambda t: not t.is_opened)
        unopened = is_unopened(self.board)

        for coord in zip(*np.where(unopened)):
            self.flag_tile(self.board[coord])

    def check_win(self) -> bool:
        """Check if number of unopened tiles == number of mines"""

        is_unopened = np.vectorize(lambda t: not t.is_opened)
        count_unopened = np.count_nonzero(is_unopened(self.board))
        return self.mines == count_unopened

    def win(self):
        self.PLAYING = False
        self.flag_all_unflagged()

    def start_timer(self):
        def update():
            if self.PLAYING:
                self.timer.configure(text=str(timedelta(seconds=self.time)))
                self.time += 1

                self.timer.after(1000, update)

        update()


class MainGUI(Frame):
    def __init__(self, tk):
        super().__init__(tk)

        self.width_frame = Frame(self)
        self.width = IntVar(value=10)
        self.width_label = Label(self.width_frame, text="Width", font=("Arial", 10))
        self.width_input = Entry(self.width_frame, textvariable=self.width, width=5)
        self.width_label.grid(row=0, column=0)
        self.width_input.grid(row=0, column=1)

        self.height_frame = Frame(self)
        self.height = IntVar(value=10)
        self.height_label = Label(self.height_frame, text="Height", font=("Arial", 10))
        self.height_input = Entry(self.height_frame, textvariable=self.height, width=5)
        self.height_label.grid(row=0, column=0)
        self.height_input.grid(row=0, column=1)

        self.mines_frame = Frame(self)
        self.mines = IntVar(value=10)
        self.mines_label = Label(self.mines_frame, text="Mines", font=("Arial", 10))
        self.mines_input = Entry(self.mines_frame, textvariable=self.mines, width=5)
        self.mines_label.grid(row=0, column=0)
        self.mines_input.grid(row=0, column=1)

        self.set_button = Button(self, text="Set", command=self.load_board)

        self.width_frame.grid(row=0, column=0)
        self.height_frame.grid(row=0, column=1)
        self.mines_frame.grid(row=0, column=2)
        self.set_button.grid(row=0, column=3)

        self.board: BoardGUI = None
        self.load_board()

    # def set_mode(self, mode: Literal["Easy", "Intermediate", "Expert"]):
    #     match mode.casefold():
    #         case "easy":
    #             self.width.set(10)
    #             self.height.set(10)
    #             self.mines.set(10)
    #         case "intermediate":
    #             self.width.set(16)
    #             self.height.set(16)
    #             self.mines.set(40)
    #         case "expert":
    #             self.width.set(30)
    #             self.height.set(16)
    #             self.mines.set(99)

    def validate_mines(self):
        area = self.width.get() * self.height.get()
        mine_density = self.mines.get() / area
        if mine_density < 0:
            self.mines.set(int(0.1*area))
        elif mine_density > 0.35:
            self.mines.set(int(0.35*area))

    def load_board(self):
        if self.board:
            self.board.destroy()

        self.validate_mines()
        self.board = BoardGUI(self, self.width.get(), self.height.get(), self.mines.get())
        self.board.grid(row=1, columnspan=4)
        return self.board


def main():
    root = Tk()
    root.title("Minesweeper")
    gui = MainGUI(root)
    gui.pack()
    gui.mainloop()


if __name__ == "__main__":
    main()
