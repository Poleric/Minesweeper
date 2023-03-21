from tkinter import *
from board import GameBoard
import numpy as np
from datetime import timedelta

import sys
from typing import Callable, Literal

LEFT_CLICK = "<Button-1>"
RIGHT_CLICK = "<Button-2>" if sys.platform == 'darwin' else "<Button-3>"
CoordT = tuple[int, int]


class BoardGUI(Frame, GameBoard):
    def __init__(self, tk, rows: int, columns: int, mines: int, seed=None):
        super().__init__(tk)
        GameBoard.__init__(self, rows, columns, mines)

        self.PLAYING = True
        self.seed = seed

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

        self.button_frame = Frame(self)
        self.buttons = self.create_button_board(self.button_frame)

        self.time = 0  # time in seconds
        self.timer = Label(self, text="0:00:00", font=("Arial", 12))

        self.display()

    def display(self) -> None:
        """Display the board and the timer"""

        self.timer.pack()
        self.button_frame.pack()

    def create_button_board(self, tk) -> np.ndarray:
        ar = np.empty((self.rows, self.columns), dtype=object)
        closed_image = self.images["closed"]

        def left_click(row, col) -> Callable:
            coord = (row, col)

            def func(event):
                if not self.PLAYING:
                    return

                self.click_tile(coord)

            return func

        def right_click(row, col) -> Callable:
            coord = (row, col)

            def func(event):
                if not self.PLAYING or self.is_opened(coord):
                    return

                if not self.is_flagged(coord):
                    self.flag_tile(coord)
                else:
                    self.unflag_tile(coord)

            return func

        for row in range(self.rows):
            for col in range(self.columns):
                button = Button(tk, image=closed_image, width=25, height=25)
                button.bind(LEFT_CLICK, left_click(row, col))
                button.bind(RIGHT_CLICK, right_click(row, col))
                button.grid(row=row, column=col)

                ar[row, col] = button
        return ar

    def open_tile(self, coord: CoordT) -> None:
        super().open_tile(coord)

        # set the picture
        tile_type = self.board[coord]
        if tile_type == -1:
            image = self.images["mine_red"]
        else:
            image = self.images["tile"][tile_type]
        self.buttons[coord].configure(image=image)

    def start_game(self, coord: CoordT) -> None:
        super().start_game(coord)
        self.start_timer()

    def start_timer(self) -> None:
        def update_timer():
            if self.PLAYING:
                self.timer.configure(text=str(timedelta(seconds=self.time)))
                self.time += 1
                self.timer.after(1000, update_timer)
        update_timer()

    def flag_tile(self, coord: CoordT) -> None:
        super().flag_tile(coord)
        self.buttons[coord].configure(image=self.images["flag"])

    def unflag_tile(self, coord: CoordT) -> None:
        super().unflag_tile(coord)
        self.buttons[coord].configure(image=self.images["closed"])

    def win(self) -> None:
        """Win game and flag all unflagged mines"""

        self.PLAYING = False
        for mine_coord in self.get_unflagged_mine_coords():
            self.flag_tile(mine_coord)

    def game_over(self, coord: CoordT) -> None:
        """Lose game and show all other mines, also shows wrong flagged
        :param coord: Coordinate of the clicked mines. Clicked mine have different image from revealed mines
        """

        self.PLAYING = False

        red_mine_image = self.images["mine"]
        for mine_coord in self.get_unflagged_mine_coords():
            if mine_coord == coord:
                continue

            self.buttons[mine_coord].configure(image=red_mine_image)

        wrong_flag_image = self.images["flag_wrong"]
        for flag_coord in self.get_wrong_flag_coords():
            self.buttons[flag_coord].configure(image=wrong_flag_image)


class MainGUI(Frame):
    def __init__(self, tk):
        super().__init__(tk)

        self.preset_buttons = Frame(self)
        self.easy_button = Button(self.preset_buttons, text="Easy", command=lambda: self.set_mode("Easy"))
        self.intermediate_button = Button(self.preset_buttons, text="Intermediate", command=lambda: self.set_mode("Intermediate"))
        self.expert_button = Button(self.preset_buttons, text="Expert", command=lambda: self.set_mode("Expert"))
        self.easy_button.grid(row=0, column=0)
        self.intermediate_button.grid(row=0, column=1)
        self.expert_button.grid(row=0, column=2)

        self.board_settings = Frame(self)

        self.rows_frame = Frame(self.board_settings)
        self.rows = IntVar(value=10)
        self.rows_label = Label(self.rows_frame, text="Rows", font=("Arial", 10))
        self.rows_input = Entry(self.rows_frame, textvariable=self.rows, width=5)
        self.rows_label.grid(row=0, column=0)
        self.rows_input.grid(row=0, column=1)

        self.columns_frame = Frame(self.board_settings)
        self.columns = IntVar(value=10)
        self.columns_label = Label(self.columns_frame, text="Columns", font=("Arial", 10))
        self.columns_input = Entry(self.columns_frame, textvariable=self.columns, width=5)
        self.columns_label.grid(row=0, column=0)
        self.columns_input.grid(row=0, column=1)

        self.mines_frame = Frame(self.board_settings)
        self.mines = IntVar(value=10)
        self.mines_label = Label(self.mines_frame, text="Mines", font=("Arial", 10))
        self.mines_input = Entry(self.mines_frame, textvariable=self.mines, width=5)
        self.mines_label.grid(row=0, column=0)
        self.mines_input.grid(row=0, column=1)

        self.set_button = Button(self.board_settings, text="Set", command=self.load_board)

        self.rows_frame.grid(row=1, column=0)
        self.columns_frame.grid(row=1, column=1)
        self.mines_frame.grid(row=1, column=2)
        self.set_button.grid(row=1, column=3)

        self.preset_buttons.pack()
        self.board_settings.pack()

        self.board: BoardGUI | None = None
        self.load_board()

    def set_mode(self, mode: Literal["Easy", "Intermediate", "Expert"]):
        match mode.casefold():
            case "easy":
                self.rows.set(10)
                self.columns.set(10)
                self.mines.set(10)
            case "intermediate":
                self.rows.set(16)
                self.columns.set(16)
                self.mines.set(40)
            case "expert":
                self.rows.set(16)
                self.columns.set(30)
                self.mines.set(99)

    def validate_mines(self):
        area = self.rows.get() * self.columns.get()
        mine_density = self.mines.get() / area
        if mine_density < 0:
            self.mines.set(int(0.1*area))
        elif mine_density > 0.35:
            self.mines.set(int(0.35*area))

    def load_board(self):
        if self.board:
            self.board.destroy()

        self.validate_mines()
        self.board = BoardGUI(self, self.rows.get(), self.columns.get(), self.mines.get())
        self.board.pack()
        return self.board


def main():
    root = Tk()
    root.title("Minesweeper")
    gui = MainGUI(root)
    gui.pack()
    gui.mainloop()


if __name__ == "__main__":
    main()
