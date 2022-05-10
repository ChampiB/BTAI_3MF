import tkinter as tk


class FactorGraphCanvasCreator:
    """
    A class used to create canvas displaying a factor graph.
    """

    def __init__(self, gui):
        """
        Construct a creator of canvas displaying a factor graph.
        :param gui: the graphical user interface.
        """
        self.gui = gui
        self.state_id = 0
        self.obs_id = 0
        self.action_id = 0
        self.variables = {}
        self.factors = {}
        self.canvas = None

    def add_variable(self, var_name, x, y, observed=False, index=0):
        """
        Add a variable to the factor graph to be displayed.
        :param var_name: the name of the variable to be displayed.
        :param x: the x position where the variable should be displayed.
        :param y: the y position where the variable should be displayed.
        :param observed: whether the variable is observed or not.
        :param index: the number of times this variable was added to the factor graph.
        :return: nothing.
        """
        # Get the display name and store the information for later use.
        display_name = self.get_display_name(var_name)
        self.variables[(display_name, index)] = (var_name, x, y, observed)

    def create_square(self, x, y, width, **kwargs):
        """
        Create a square with a specified width at coordinate (x, y).
        :param x: the x position of the square.
        :param y: the y position of the square.
        :param width: the width of the square.
        :param kwargs: optional arguments.
        :return: the index of the object in the canvas.
        """
        return self.canvas.create_rectangle(
            x - width / 2, y - width / 2, x + width / 2, y + width / 2, **kwargs
        )

    def create_circle(self, x, y, r, **kwargs):
        """
        Create a circle with radius r at coordinate (x, y).
        :param x: the x position of the circle.
        :param y: the y position of the circle.
        :param r: the radius of the circle.
        :param kwargs: optional arguments.
        :return: the index of the object in the canvas.
        """
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, **kwargs)

    def get_display_name(self, var_name):
        """
        Getter.
        :param var_name: the real name of the random variable.
        :return: the variable name to be displayed.
        """
        for (d_name, _), (v_name, _, _, _) in self.variables.items():
            if var_name == v_name:
                return d_name
        if var_name[0:2] == "S_":
            self.state_id += 1
            return "S{}".format(self.state_id)
        if var_name[0:2] == "O_":
            self.obs_id += 1
            return "O{}".format(self.obs_id)
        if var_name[0:2] == "A_":
            self.action_id += 1
            return "A{}".format(self.action_id)
        raise Exception("Variable name should start with either S_, O_ or A_.")

    def add_factor(self, factor_name, x, y, var_names):
        """
        Add a factor to the factor graph to be displayed.
        :param factor_name: the name of the factor.
        :param x: the x position where the factor should be displayed.
        :param y: the y position where the factor should be displayed.
        :param var_names: the names of neighbour variables.
        :return: nothing.
        """
        self.factors[factor_name] = (x, y, var_names)

    def get_canvas(self, parent, w, h, line_style="border"):
        """
        Getter.
        :param parent: the parent widget of the canvas.
        :param w: the canvas width.
        :param h: the canvas height.
        :param line_style: the style of connection, either 'border' or 'center'.
        :return: the canvas containing the factor graph specified by the user.
        """
        # Create a new canvas.
        self.canvas = tk.Canvas(
            parent, bg=self.gui.white, width=w, height=h,
            highlightbackground=self.gui.black, highlightthickness=2
        )

        # Display all connections between the factors and the variables.
        for fx, fy, var_names in self.factors.values():
            for var_name in var_names:
                _, (var_name, x, y, _) = next(filter(
                    lambda variable: variable[1][0] == var_name[0] and variable[0][1] == var_name[1],
                    self.variables.items()
                ))
                if line_style == "center" or fy == y:
                    self.canvas.create_line(fx, fy, x, y)
                elif fy > y:
                    self.canvas.create_line(fx, fy - 5, x, y + 20)
                else:
                    self.canvas.create_line(fx, fy + 5, x, y - 20)

        # Display all variables.
        for (display_name, _), (var_name, x, y, observed) in self.variables.items():
            if observed:
                self.create_circle(x, y, 20, fill=self.gui.gray)
            else:
                self.create_circle(x, y, 20, fill=self.gui.white)
            self.canvas.create_text(x, y, text=display_name)

        # Display all factors.
        for factor_name, (x, y, var_names) in self.factors.items():
            self.create_square(x, y, 10, fill=self.gui.black)

        # Bind mouse motion over canvas.
        self.canvas.bind("<Motion>", self.print_variable_name)
        return self.canvas

    def print_variable_name(self, event):
        """
        Display the variable name when the mouse is over it.
        :param event: an event describing the event that triggered the call to this function.
        :return: nothing.
        """
        # Display the tooltip if the mouse is over a random variable.
        for (_, _), (var_name, x, y, _) in self.variables.items():
            if x - 20 < event.x < x + 20 and y - 20 < event.y < y + 20:
                # If tooltip already exists, delete it.
                if len(self.canvas.find_withtag("tooltip")) != 0:
                    self.delete_tooltip()
                # Crete the tooltip at the current mouse position.
                self.create_tooltip(event, var_name)
                return

        # Delete the tooltip if the not hovering a variable.
        self.delete_tooltip()

    def create_tooltip(self, event, var_name, anchor="sw"):
        """
        Create the tooltip displaying the variable name.
        :param event: the event that triggered the call to this function.
        :param var_name: the variable name to display.
        :param anchor: the position relative to the mouse where this should be displayed.
        :return: nothing.
        """
        # Check that the tooltip fit in the canvas, if not change the anchor.
        text = self.canvas.create_text(event.x, event.y, text=var_name, tags="tooltip", anchor=anchor)
        pos = self.canvas.bbox(text)
        margin = 3
        if pos[2] + margin > self.canvas.winfo_width() or pos[1] - margin < 0:
            anchors = ["sw", "se", "nw", "ne"]
            anchor = anchors[anchors.index(anchor) + 1]
            self.delete_tooltip()
            self.create_tooltip(event, var_name, anchor=anchor)
            return

        # Create the tooltip.
        self.canvas.create_rectangle(
            pos[0] - margin, pos[1] - margin, pos[2] + margin, pos[3] + margin,
            fill=self.gui.white, tags="tooltip"
        )
        self.canvas.create_text(event.x, event.y, text=var_name, tags="tooltip", anchor=anchor)

    def delete_tooltip(self):
        """
        Delete the tooltip.
        :return: nothing.
        """
        tooltip_id = self.canvas.find_withtag("tooltip")
        for i in tooltip_id:
            self.canvas.delete(i)

