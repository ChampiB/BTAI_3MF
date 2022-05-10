import tkinter as tk
from analysis.widgets.LikelihoodModelWidget import LikelihoodModelWidget
from analysis.widgets.TransitionModelWidget import TransitionModelWidget
from analysis.widgets.PriorPreferencesWidget import PriorPreferencesWidget
from analysis.widgets.MCTSWidget import MCTSWidget
from analysis.widgets.PosteriorWidget import PosteriorWidget
from analysis.widgets.MessageWidget import MessageWidget
from analysis.widgets.ExpectedFreeEnergyWidget import ExpectedFreeEnergyWidget


class TemporalSliceFrame(tk.Frame):
    """
    Class representing the frame displaying the information of the current
    temporal slice.
    """

    def __init__(self, parent, gui):
        """
        Construct the frame that display the temporal slice information.
        :param parent: the parent widget.
        :param gui: the graphical user interface.
        """
        tk.Frame.__init__(self, parent)

        # Store graphical user interface.
        self.gui = gui

        # Create the canvas containing illustrating the transition model.
        self.transition_canvas = TransitionModelWidget(self, gui)

        # Create the canvas containing illustrating the prior preferences.
        self.prior_preferences_canvas = PriorPreferencesWidget(self, gui)

        # Create the widget displaying the MCTS information.
        self.mcts_widget = MCTSWidget(self, gui)

        # Create the widget displaying the variables posterior.
        self.posterior_widget = PosteriorWidget(self, gui)

        # Create the widget displaying the variables posterior.
        self.efe_or_msg_widget = ExpectedFreeEnergyWidget(self, gui) \
            if self.gui.current_ts.parent is not None else MessageWidget(self, gui)

        # Create the canvas containing illustrating the likelihood model.
        self.likelihood_canvas = LikelihoodModelWidget(self, gui)

        # Set position of all widgets.
        self.set_widgets_position()

    def set_widgets_position(self):
        """
        Set all the widgets position.
        :return: nothing.
        """
        # Create empty rows and columns.
        self.grid_columnconfigure(1, minsize=50)
        self.grid_columnconfigure(3, minsize=50)
        self.grid_rowconfigure(1, minsize=50)

        # Set the position of all widgets.
        self.likelihood_canvas.grid(row=0, column=0)
        self.transition_canvas.grid(row=0, column=2)
        self.prior_preferences_canvas.grid(row=0, column=4)
        self.posterior_widget.grid(row=2, column=0)
        self.mcts_widget.grid(row=2, column=2)
        self.efe_or_msg_widget.grid(row=2, column=4)

    def refresh_callback(self, _):
        """
        Call the refresh function.
        :return: nothing.
        """
        self.refresh()

    def refresh(self):
        """
        Refresh the content of the frame.
        """
        # Refresh the widget displaying the transition model.
        self.transition_canvas.destroy()
        self.transition_canvas = TransitionModelWidget(self, self.gui)

        # Refresh the widget displaying theprior preferences.
        self.prior_preferences_canvas.destroy()
        self.prior_preferences_canvas = PriorPreferencesWidget(self, self.gui)

        # Create the widget displaying the variables posterior.
        self.efe_or_msg_widget.destroy()
        self.efe_or_msg_widget = ExpectedFreeEnergyWidget(self, self.gui) \
            if self.gui.current_ts.parent is not None else MessageWidget(self, self.gui)

        # Refresh the MCTS information.
        self.mcts_widget.refresh()

        # Refresh the posterior widget.
        self.posterior_widget.refresh()

        # Refresh the widget displaying the likelihood model.
        self.likelihood_canvas.destroy()
        self.likelihood_canvas = LikelihoodModelWidget(self, self.gui)

        # Set position of all widgets.
        self.set_widgets_position()

