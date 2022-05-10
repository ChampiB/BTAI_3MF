import tkinter as tk
import tkinter.ttk as ttk


class PosteriorWidget(tk.Frame):
    """
    Widget used to display the posterior beliefs of the latent variable
    of the temporal slice.
    """

    def __init__(self, parent, gui):
        """
        Construct the widget that display the posterior districution over variables.
        :param parent: the parent widget.
        :param gui: the graphical user interface.
        """
        tk.Frame.__init__(self, parent)

        # Store graphical user interface.
        self.gui = gui

        # Create the canvas's title.
        self.text_label = tk.Label(self, text="Posterior beliefs")
        self.text_label.grid(row=0, column=0, sticky=tk.NSEW)

        # Create a new canvas.
        width = 500
        height = 300
        self.canvas = tk.Canvas(
            self, bg=self.gui.white, width=width, height=height,
            highlightbackground=self.gui.black, highlightthickness=2
        )
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW)

        # Draw axis.
        self.x_o = 20
        self.y_o = height - 50
        self.x_a1 = width - 20
        self.y_a1 = height - 50
        self.x_a2 = 20
        self.y_a2 = 50
        self.canvas.create_line(self.x_o, self.y_o, self.x_a1, self.y_a1, arrow=tk.LAST)
        self.canvas.create_line(self.x_o, self.y_o, self.x_a2, self.y_a2, arrow=tk.LAST)

        # Draw axis labels.
        self.y_label = self.canvas.create_text(self.x_a2 - 10, self.y_a2 - 20, text="P(RV|O_t)", anchor=tk.W)
        self.canvas.create_text(self.x_a1 - 150, self.y_a1 + 33, text="RV:")

        # Create the combo box to select a targeted random variable.
        self.cbox = ttk.Combobox(self, state='readonly', width=10)
        self.cbox['values'] = self.get_variables_names()
        self.cbox.current(0)
        self.cbox.bind("<<ComboboxSelected>>", self.refresh_widget)
        self.canvas.create_window(self.x_a1 - 60, self.y_a1 + 33, window=self.cbox)

        # Display the probability distribution.
        self.bar_tags = []
        self.index_tags = []
        self.display_posterior()

    def get_variables_names(self):
        """
        Getter.
        :return: the list of all latent variables in the temporal slice.
        """
        names = list(self.gui.current_ts.states_prior.keys())
        if self.gui.current_ts.parent is not None:
            names += list(self.gui.current_ts.obs_likelihood.keys())
        return names

    def display_posterior(self):
        """
        Display the posterior beliefs graphically.
        :return: nothing.
        """
        # Delete the previous bars from the graph.
        for bar_tag in self.bar_tags:
            self.canvas.delete(bar_tag)
        self.bar_tags = []
        for index_tag in self.index_tags:
            self.canvas.delete(index_tag)
        self.index_tags = []


        # Get posterior parameters.
        var_name = self.cbox.get()
        posterior = self.gui.current_ts.states_posterior[var_name] \
            if var_name[0:2] == "S_" else self.gui.current_ts.obs_posterior[var_name]

        # Get maximum posterior probability, width of each bar, and max vertical space.
        n_values = posterior.size(0)
        max_value = posterior.max()
        total_space = self.x_a1 - self.x_o
        available_space = total_space - 5 * (n_values + 1)
        bar_width = int(available_space / n_values)
        max_vspace = self.y_a2 - self.y_o + 15

        # Display the bars representing posterior probabilities.
        xshift = self.x_o + 5
        for i in range(n_values):
            tags = "bar_{}".format(i)
            self.bar_tags.append(tags)
            bar_height = int(posterior[i] / max_value * max_vspace)
            self.canvas.create_rectangle(
                xshift, self.y_a1 - 1, xshift + bar_width, self.y_a1 + bar_height,
                fill=self.gui.blue, tags=tags
            )
            tags = "index_{}".format(i)
            self.index_tags.append(tags)
            if n_values <= 10 or i % 5 == 0:
                self.canvas.create_text(
                    xshift + int(bar_width / 2), self.y_a1 + 10, text=str(i), tags=tags
                )
            xshift += 5 + bar_width

    def refresh_widget(self, _):
        """
        Call the refresh function.
        :return: nothing.
        """
        self.refresh(False)

    def refresh(self, reset_value=True):
        """
        Refresh the posterior distribution widget.
        :param reset_value: True if the value of the combo box should be reset, False otherwise.
        :return: nothing.
        """
        self.cbox['values'] = self.get_variables_names()
        if reset_value:
            self.cbox.current(0)
        self.display_posterior()
