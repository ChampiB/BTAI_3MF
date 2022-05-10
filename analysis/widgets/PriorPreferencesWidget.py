import math
import tkinter as tk
from analysis.widgets.FactorGraphCanvasCreator import FactorGraphCanvasCreator as CanvasCreator


class PriorPreferencesWidget(tk.Frame):

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
        self.text_label = tk.Label(self, text="Prior preferences")
        self.text_label.grid(row=0, column=0, sticky=tk.NSEW)

        # Create the canvas representing the factor graph of the likelihood model.
        cc = CanvasCreator(gui)
        processed_obs = []
        xshift = 5
        yshift = 80
        for obs_name, (rv_names, prior_pref) in gui.current_ts.obs_prior_pref.items():

            # Check if the observation has already been processed.
            if obs_name in processed_obs:
                continue

            # Compute the display the prior preferences of the variables in rv_names.
            n_rvs = len(rv_names)
            radians = [2 * i * math.pi / n_rvs + math.pi / 2 for i in range(n_rvs)]
            xshift += 25 if n_rvs <= 2 else 75
            for i, rv_name in enumerate(rv_names):
                cc.add_variable(
                    rv_name,
                    xshift - 50 * math.cos(radians[i]),
                    yshift - 50 * math.sin(radians[i])
                )
            parents = [(rv_name, 0) for rv_name in rv_names]
            cc.add_factor("f_" + obs_name, xshift, yshift, parents)
            xshift += 25 if n_rvs <= 2 else 75

            # Add the random variable of the subset to the list of processed observation.
            processed_obs += rv_names

        self.canvas = cc.get_canvas(self, xshift, 160, line_style="center")
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW)
