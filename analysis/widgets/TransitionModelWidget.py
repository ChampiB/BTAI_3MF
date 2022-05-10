import tkinter as tk
from analysis.widgets.FactorGraphCanvasCreator import FactorGraphCanvasCreator as CanvasCreator


class TransitionModelWidget(tk.Frame):

    def __init__(self, parent, gui):
        """
        Construct the widget that display the transition model.
        :param parent: the parent widget.
        :param gui: the graphical user interface.
        """
        tk.Frame.__init__(self, parent)

        # Store the temporal slice.
        self.gui = gui

        # Create the canvas's title.
        self.text_label = tk.Label(self, text="Transition model")
        self.text_label.grid(row=0, column=0, sticky=tk.NSEW)

        # Create the canvas representing the factor graph of the likelihood model.
        cc = CanvasCreator(gui)
        xshift = 30
        yshift = -30
        for i, state_name in enumerate(gui.current_ts.states_prior.keys()):
            cc.add_variable(state_name, xshift + 50 * i, yshift + 100, index=0)
            cc.add_factor(
                "f_" + state_name + "_t",
                xshift + 50 * i, yshift + 50, [(state_name, 0)]
            )
        action_name = self.gui.current_ts.action_name
        cc.add_variable(
            action_name, xshift + 50 * len(gui.current_ts.states_prior),
            yshift + 100, observed=True, index=1
        )
        cc.add_factor(
            "f_" + action_name, xshift + 50 * len(gui.current_ts.states_prior),
            yshift + 50, [(action_name, 1)]
        )
        for i, (state_name, state_parents) in enumerate(gui.current_ts.states_parents.items()):
            cc.add_variable(state_name, xshift + 50 * i, yshift + 200, index=1)
            parents = [(state_name, 0)] + [(state_parent, 1) for state_parent in state_parents]
            cc.add_factor("f_" + state_name + "_t+1", xshift + 50 * i, yshift + 150, parents)
        max_nb_rv = max(len(gui.current_ts.states_prior.keys()), len(gui.current_ts.states_parents.items()))
        self.canvas = cc.get_canvas(self, 2 * xshift + 50 * max_nb_rv, 200)
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW)
