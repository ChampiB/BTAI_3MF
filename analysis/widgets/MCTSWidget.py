import tkinter as tk


class MCTSWidget(tk.Frame):
    """
    A widget that display the MCTS information.
    """

    def __init__(self, parent, gui):
        """
        Construct the widget that display the likelihood model.
        :param parent: the parent widget.
        :param gui: the graphical user interface.
        """
        tk.Frame.__init__(self, parent)

        # Store the temporal slice.
        self.gui = gui

        # Create the canvas's title.
        self.text_label = tk.Label(self, text="MCTS information")
        self.text_label.grid(row=0, column=0, sticky=tk.NSEW)

        # Create a new frame to store the MCTS information.
        self.info_frame = tk.Frame(
            self, bg=self.gui.white,
            highlightbackground=self.gui.black, highlightthickness=2
        )
        self.info_frame.grid(row=1, column=0, sticky=tk.NSEW)

        # Create labels to display MCTS information.
        self.action_label = tk.Label(
            self.info_frame, bg=self.gui.white,
            text="Action: {}".format(gui.current_ts.action)
        )
        self.action_label.grid(row=0, column=0, sticky=tk.NSEW)

        self.cost_label = tk.Label(
            self.info_frame, bg=self.gui.white,
            text="Cost: {}".format(round(gui.current_ts.cost, 3))
        )
        self.cost_label.grid(row=1, column=0, sticky=tk.NSEW)

        self.visits_label = tk.Label(
            self.info_frame, bg=self.gui.white,
            text="Visits: {}".format(gui.current_ts.visits)
        )
        self.visits_label.grid(row=2, column=0, sticky=tk.NSEW)

        self.n_actions_label = tk.Label(
            self.info_frame, bg=self.gui.white,
            text="Number of actions: {}".format(gui.current_ts.n_actions)
        )
        self.n_actions_label.grid(row=3, column=0, sticky=tk.NSEW)

    def refresh(self):
        """
        Refresh the MCTS information.
        :return: nothing.
        """
        self.action_label.config(
            text="Action: {}".format(self.gui.current_ts.action)
        )
        self.cost_label.config(
            text="Cost: {}".format(round(self.gui.current_ts.cost, 3))
        )
        self.visits_label.config(
            text="Visits: {}".format(self.gui.current_ts.visits)
        )
        self.n_actions_label.config(
            text="Number of actions: {}".format(self.gui.current_ts.n_actions)
        )
