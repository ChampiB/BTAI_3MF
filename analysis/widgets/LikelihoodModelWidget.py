import tkinter as tk
from analysis.widgets.FactorGraphCanvasCreator import FactorGraphCanvasCreator as CanvasCreator


class LikelihoodModelWidget(tk.Frame):

    def __init__(self, parent, gui):
        """
        Construct the widget that display the likelihood model.
        :param parent: the parent widget.
        :param gui: the graphical user interface.
        """
        tk.Frame.__init__(self, parent)

        # Store graphical user interface and parent widget.
        self.gui = gui
        self.parent = parent

        # Create the canvas's title.
        self.text_label = tk.Label(self, text="Likelihood model")
        self.text_label.grid(row=0, column=0, sticky=tk.NSEW)

        # Create the canvas representing the factor graph of the likelihood model.
        self.cc = CanvasCreator(gui)
        xshift = 30
        yshift = -30
        for i, state_name in enumerate(gui.current_ts.states_prior.keys()):
            self.cc.add_variable(state_name, xshift + 50 * i, yshift + 100)
            self.cc.add_factor("f_" + state_name, xshift + 50 * i, yshift + 50, [(state_name, 0)])
        for i, (obs_name, obs_parents) in enumerate(gui.current_ts.obs_parents.items()):
            self.cc.add_variable(obs_name, xshift + 50 * i, yshift + 200, observed=gui.current_ts.parent is None)
            parents = [(obs_name, 0)] + [(obs_parent, 0) for obs_parent in obs_parents]
            self.cc.add_factor("f_" + obs_name, xshift + 50 * i, yshift + 150, parents)
            if gui.current_ts.parent is None:
                self.cc.add_factor("e_" + obs_name, xshift + 50 * i, yshift + 250, [(obs_name, 0)])
        max_nb_rv = max(len(gui.current_ts.states_prior.keys()), len(gui.current_ts.obs_parents.items()))
        self.canvas = self.cc.get_canvas(
            self, 2 * xshift + 50 * (max_nb_rv - 1),
            250 if gui.current_ts.parent is None else 200
        )
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW)
        self.canvas.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        """
        Set the combo boxes of the posterior and message widgets to match
        what the user clicked on.
        :param event: the event that triggered the call to this function.
        :return: nothing.
        """
        # Get the name of the node on which the user clicked.
        name = None
        xshift = 30
        yshift = -30
        for i, state_name in enumerate(self.gui.current_ts.states_prior.keys()):
            if self.clicked_on_node(xshift + 50 * i, yshift + 100, 20, event):
                name = state_name
            if self.clicked_on_node(xshift + 50 * i, yshift + 50, 10, event):
                name = "f_" + state_name
        for i, (obs_name, obs_parents) in enumerate(self.gui.current_ts.obs_parents.items()):
            if self.clicked_on_node(xshift + 50 * i, yshift + 200, 20, event):
                name = obs_name
            if self.clicked_on_node(xshift + 50 * i, yshift + 150, 10, event):
                name = "f_" + obs_name
            if self.gui.current_ts.parent is None:
                if self.clicked_on_node(xshift + 50 * i, yshift + 250, 10, event):
                    name = "e_" + obs_name

        # If no node was clicked on, return.
        if name is None:
            return

        # If the current temporal slice is the one of the present time step.
        if self.gui.current_ts.parent is None:
            if name[0:2] == "O_":
                self.set_message_from_cbox(name)
            elif name[0:2] == "S_":
                self.set_message_from_cbox(name)
                self.set_posterior_rv_cbox(name)
            elif name[0:2] == "f_" or name[0:2] == "e_":
                self.set_message_from_cbox(name)
            return

        # If the node clicked on is a latent variable in the future.
        if name[0:2] != "f_":
            self.set_posterior_rv_cbox(name)

    @staticmethod
    def clicked_on_node(x, y, margin, event):
        """
        Check if the node at position (x, y) was clicked on during an event,
        assuming a vertical and horizontal margin of error around the position (x, y).
        :param x: the x position of the node.
        :param y: the y position of the node.
        :param margin: the vertical and horizontal margin of error around the position (x, y).
        :param event: the event describing where the user clicked.
        :return: True if the node was clicked on, False otherwise.
        """
        return x - margin < event.x < x + margin and y - margin < event.y < y + margin

    def set_message_from_cbox(self, name):
        """
        Set the combo box value of the message widget to match the input name.
        :param name: the name that should become the combo box value of the message widget.
        :return: nothing.
        """
        cbox = self.parent.efe_or_msg_widget.cbox_from
        index = cbox['values'].index(name)
        cbox.current(index)
        self.parent.efe_or_msg_widget.refresh_widget_from(None)

    def set_posterior_rv_cbox(self, name):
        """
        Set the combo box value of the posterior widget to match the input name.
        :param name: the name that should become the combo box value of the posterior widget.
        :return: nothing.
        """
        cbox = self.parent.posterior_widget.cbox
        index = cbox['values'].index(name)
        cbox.current(index)
        self.parent.posterior_widget.refresh_widget(None)
