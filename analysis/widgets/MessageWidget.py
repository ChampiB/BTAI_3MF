import tkinter as tk
import tkinter.ttk as ttk


class MessageWidget(tk.Frame):
    """
    Widget used to display the messages sent between factors and random variable
    of the temporal slice.
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
        self.text_label = tk.Label(self, text="Messages")
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
        self.y_label = self.canvas.create_text(self.x_a2 - 10, self.y_a2 - 20, text="msg(From->To)", anchor=tk.W)
        self.canvas.create_text(self.x_a1 - 340, self.y_a1 + 33, text="From:")
        self.canvas.create_text(self.x_a1 - 150, self.y_a1 + 33, text="To:")

        # Create the combo boxes to select a source and target random variable / factor.
        self.cbox_from = ttk.Combobox(self, state='readonly', width=10)
        self.cbox_from['values'] = self.get_nodes_names()
        self.cbox_from.bind("<<ComboboxSelected>>", self.refresh_widget_from)
        self.canvas.create_window(self.x_a1 - 250, self.y_a1 + 33, window=self.cbox_from)

        self.cbox_to = ttk.Combobox(self, state='disabled', width=10)
        self.cbox_to['values'] = self.get_nodes_names()
        self.cbox_to.bind("<<ComboboxSelected>>", self.refresh_widget_to)
        self.canvas.create_window(self.x_a1 - 60, self.y_a1 + 33, window=self.cbox_to)

        # Display the probability distribution.
        self.bar_tags = []
        self.index_tags = []

    def get_nodes_names(self):
        """
        Getter.
        :return: the list of all nodes in the factor graph, i.e., variables and factors.
        """
        return list(self.gui.current_ts.fg.nodes.keys())

    def display_message(self):
        """
        Display the message graphically.
        :return: nothing.
        """
        # Get message.
        node_from = self.cbox_from.get()
        node_to = self.cbox_to.get()
        if node_from == "" or node_to == "":
            return
        message = self.gui.current_ts.fg.nodes[node_to].in_messages[node_from]

        # Delete the previous bars from the graph.
        for bar_tag in self.bar_tags:
            self.canvas.delete(bar_tag)
        for index_tag in self.index_tags:
            self.canvas.delete(index_tag)

        # Get maximum posterior probability, width of each bar, and max vertical space.
        n_values = message.size(0)
        max_value = message.max()
        total_space = self.x_a1 - self.x_o
        available_space = total_space - 5 * (n_values + 1)
        bar_width = int(available_space / n_values)
        max_vspace = self.y_a2 - self.y_o + 15

        # Display the bars representing posterior probabilities.
        xshift = self.x_o + 5
        for i in range(n_values):
            tags = "bar_{}".format(i)
            self.bar_tags.append(tags)
            bar_height = int(message[i] / max_value * max_vspace)
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

    def get_neighbours(self, node_name):
        """
        Getter.
        :param node_name: the name of the node whose neighbours name should be returned.
        :return: the neighbours name of the node whose name is provided as input.
        """
        return self.gui.current_ts.fg.nodes[node_name].neighbours

    def refresh_widget_from(self, _):
        """
        Call the refresh function.
        :return: nothing.
        """
        self.cbox_to.config(state='normal')
        self.cbox_to.delete(0, "end")
        self.cbox_to.config(state='readonly')
        self.cbox_to['values'] = self.get_neighbours(self.cbox_from.get())
        self.refresh()

    def refresh_widget_to(self, _):
        """
        Call the refresh function.
        :return: nothing.
        """
        self.refresh()

    def refresh(self):
        """
        Refresh the posterior distribution widget.
        :return: nothing.
        """
        self.display_message()
