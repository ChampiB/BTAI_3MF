import tkinter as tk
import math


class ExpectedFreeEnergyWidget(tk.Frame):
    """
    Widget used to display the expected free energy components such as risk and ambiguity.
    """

    def __init__(self, parent, gui):
        """
        Construct the widget that display the posterior districution over variables.
        :param parent: the parent widget.
        :param gui: the graphical user interface.
        """
        tk.Frame.__init__(self, parent)

        # Store the temporal slice.
        self.gui = gui

        # Create the canvas's title.
        self.text_label = tk.Label(self, text="Expected Free Energy")
        self.text_label.grid(row=0, column=0, sticky=tk.NSEW)

        # Create a new canvas.
        width = 500
        height = 300
        self.canvas = tk.Canvas(
            self, bg=self.gui.white, width=width, height=height,
            highlightbackground=self.gui.black, highlightthickness=2
        )
        self.canvas.grid(row=1, column=0, sticky=tk.NSEW)
        self.canvas.bind("<Motion>", self.print_term_name)
        self.canvas.bind("<Button-1>", self.update_display_type)

        # Draw axis.
        self.x_o = 70
        self.y_o = height - 20
        self.x_a1 = width - 20
        self.y_a1 = self.y_o
        self.x_a2 = self.x_o
        self.y_a2 = 50
        self.canvas.create_line(self.x_o, self.y_o, self.x_a1, self.y_a1)
        self.canvas.create_line(self.x_o, self.y_o, self.x_a2, self.y_a2, arrow=tk.LAST)

        # The x and y coordinates of the exit cross.
        self.exit_cross_x = self.x_a1 - 5
        self.exit_cross_y = self.y_a2 - 20

        # Draw axis labels.
        self.y_label = self.canvas.create_text(self.x_a2, self.y_a2 - 20, text="EFE")

        # Display the expected free energy components.
        self.colors = [
            self.gui.blue,
            self.gui.red,
            self.gui.orange,
            self.gui.green,
            self.gui.gray,
            self.gui.white
        ]
        self.display = "EFE"
        self.is_lock = False
        self.data = {}
        self.bar_tags = []
        self.index_tags = []
        self.display_efe()

    def display_efe(self):
        """
        Display the expected free energy graphically.
        :return: nothing.
        """
        # Delete the previous bars from the graph.
        for bar_tag in self.bar_tags:
            self.canvas.delete(bar_tag)
        self.bar_tags = []
        for index_tag in self.index_tags:
            self.canvas.delete(index_tag)
        self.index_tags = []

        # Update the data that will be displyed.
        self.update_data_to_display()

        # Update y-axis label.
        self.canvas.itemconfig(self.y_label, text=self.display)

        # Get most extrem value to display, width of each bar, and max vertical space.
        max_value = -math.inf
        min_value = math.inf
        total_value = 0
        for value in self.data.values():
            total_value += value
            max_value = max(max_value, total_value)
            min_value = min(min_value, total_value)
        ext_value = min_value if abs(min_value) > abs(max_value) else max_value
        bar_width = 30
        max_vspace = self.y_a2 - self.y_o + 10

        # Display the bars representing posterior probabilities.
        xshift = self.x_o + 20
        yshift = self.y_o - 1
        for i, (key, value) in enumerate(self.data.items()):
            # Compute the color of the term being displayed.
            color = self.colors[i % len(self.colors)]

            # Display a bar for each term.
            tags = "bar_{}".format(key)
            self.bar_tags.append(tags)
            bar_height = int(self.data[key] / ext_value * max_vspace)
            self.canvas.create_rectangle(
                xshift, yshift,
                xshift + bar_width, yshift + bar_height,
                fill=color, tags=tags
            )
            yshift += bar_height
            xshift += bar_width

        # Create the horizontal line corresponding to the total value displayed.
        tags = "index_line"
        self.index_tags.append(tags)
        self.canvas.create_line(
            self.x_o, yshift, xshift, yshift,
            width=2, tags=tags, dash=(4, 2)
        )

        # Create text displaying the total value displayed.
        tags = "index_text"
        self.index_tags.append(tags)
        value = round(sum(self.data.values()), 3)
        self.canvas.create_text(self.x_o - 30, yshift, text=str(value), tags=tags)

        # Display the exit cross if within the risk or ambiguity window.
        self.remove_exit_cross()
        if self.display == "Risk" or self.display == "Ambiguity":
            self.display_exit_cross()

    def remove_exit_cross(self):
        """
        Remove the exit cross from the canvas.
        :return: nothing.
        """
        items = self.canvas.find_withtag("exit_cross")
        for item in items:
            self.canvas.delete(item)

    def display_exit_cross(self):
        """
        Display the exit cross.
        :return: nothing.
        """
        self.canvas.create_line(
            self.exit_cross_x - 5, self.exit_cross_y - 5,
            self.exit_cross_x + 5, self.exit_cross_y + 5,
            fill=self.gui.red, tags="exit_cross", width=2
        )
        self.canvas.create_line(
            self.exit_cross_x - 5, self.exit_cross_y + 5,
            self.exit_cross_x + 5, self.exit_cross_y - 5,
            fill=self.gui.red, tags="exit_cross", width=2
        )

    def update_data_to_display(self):
        """
        Update the data to be displayed.
        :return: nothing.
        """
        self.data = {}
        if self.display == "EFE":
            # Load the expected free energy components.
            self.data["Risk"] = sum(self.gui.current_ts.compute_risk_terms())
            self.data["Ambiguity"] = sum(self.gui.current_ts.compute_ambiguity_terms())
        elif self.display == "Risk":
            # Load the risk components.
            risks = self.gui.current_ts.compute_risk_terms()
            processed_modalities = []
            i = 0
            for obs_name, (rv_names, _) in self.gui.current_ts.obs_prior_pref.items():
                if obs_name in processed_modalities:
                    continue
                self.data["risk[{}]".format(",".join(rv_names))] = risks[i]
                i += 1
                processed_modalities += rv_names
        else:
            # Load the ambiguity components.
            ambiguities = self.gui.current_ts.compute_ambiguity_terms()
            for i, rv_name in enumerate(self.gui.current_ts.obs_likelihood.keys()):
                self.data["ambiguity[{}]".format(rv_name)] = ambiguities[i]

    def update_display_type(self, event):
        """
        Update the display type, i.e., if the user clicked on the risk or ambiguity term,
        then additional information about this term will be displayed.
        :param event: the event that triggered the call to this function.
        :return: nothing.
        """
        # If the user clicked on the red cross, go back to the EFE screen.
        if self.display != "EFE" \
                and self.exit_cross_x - 10 < event.x < self.exit_cross_x + 10 \
                and self.exit_cross_y - 10 < event.y < self.exit_cross_y + 10:
            self.is_lock = True
            self.display = "EFE"
            self.display_efe()
            self.is_lock = False
            return

        for i, bar_tag in enumerate(self.bar_tags):
            bar = self.canvas.find_withtag(bar_tag)
            if len(bar) == 0:
                continue
            pos = self.canvas.bbox(bar[0])

            # Check if the user asked details about risk or ambiguity.
            new_display = list(self.data.keys())[i]
            if new_display != "Risk" and new_display != "Ambiguity":
                return

            # If the user clicked on the risk or ambiguity term, then
            # display additional information about this term.
            if pos[0] < event.x < pos[2] and pos[1] < event.y < pos[3]:
                self.is_lock = True
                self.display = new_display
                self.display_efe()
                self.is_lock = False
                return

    def print_term_name(self, event):
        """
        Display the variable name when the mouse is over it.
        :param event: an event describing the event that triggered the call to this function.
        :return: nothing.
        """
        # If the widget is locked, return.
        if self.is_lock:
            return

        # Display the tooltip if the mouse is over a random variable.
        for i, bar_tag in enumerate(self.bar_tags):
            bar = self.canvas.find_withtag(bar_tag)
            if len(bar) == 0:
                continue
            pos = self.canvas.bbox(bar[0])
            if pos[0] < event.x < pos[2] and pos[1] < event.y < pos[3]:
                # If tooltip already exists, delete it.
                if len(self.canvas.find_withtag("tooltip")) != 0:
                    self.delete_tooltip()
                # Crete the tooltip at the current mouse position.
                value = round(list(self.data.values())[i], 3)
                name = list(self.data.keys())[i] + " = " + str(value)
                self.create_tooltip(event, name)
                return

        # Delete the tooltip if the not hovering a variable.
        self.delete_tooltip()

    def create_tooltip(self, event, term_name, anchor='sw'):
        """
        Create the tooltip displaying the term's name.
        :param event: the event that triggered the call to this function.
        :param term_name: the term's name to display.
        :param anchor: the position relative to the mouse where this should be displayed.
        :return: nothing.
        """
        # Check that the tooltip fit in the canvas, if not change the anchor.
        text = self.canvas.create_text(event.x, event.y, text=term_name, tags="tooltip", anchor=anchor)
        pos = self.canvas.bbox(text)
        margin = 3
        if anchor != 'w' and (
                pos[2] + margin > self.canvas.winfo_width()
                or pos[1] - margin < 0
                or pos[0] - margin < 0
        ):
            anchors = ['sw', 'se', 'nw', 'ne', 's', 'n', 'e', 'w']
            anchor = anchors[anchors.index(anchor) + 1]
            self.delete_tooltip()
            self.create_tooltip(event, term_name, anchor=anchor)
            return

        # Create the tooltip.
        self.canvas.create_rectangle(
            pos[0] - margin, pos[1] - margin, pos[2] + margin, pos[3] + margin,
            fill=self.gui.white, tags="tooltip"
        )
        self.canvas.create_text(event.x, event.y, text=term_name, tags="tooltip", anchor=anchor)

    def delete_tooltip(self):
        """
        Delete the tooltip.
        :return: nothing.
        """
        tooltip_id = self.canvas.find_withtag("tooltip")
        for i in tooltip_id:
            self.canvas.delete(i)

    def refresh_widget(self, _):
        """
        Call the refresh function.
        :return: nothing.
        """
        self.refresh()

    def refresh(self):
        """
        Refresh the expected free energy widget.
        :return: nothing.
        """
        self.display_efe()
