import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class VisualisationFrame(tk.Frame):
    """
    Class representing the frame used to visualise the agent's planning
    and action selection scheme.
    """

    def __init__(self, parent, gui):
        """
        Construct the visulisation frame.
        :param parent: the parent widget.
        :param gui: the graphical user interface.
        """
        tk.Frame.__init__(self, parent)

        # Store graphical user interface.
        self.gui = gui

        # Attributes for the step by step simulation and analysis of BTAI_3MF.
        self.planning_step = 0
        self.current_action_seq = ""
        self.node_selected_by_uct = None

        # Create the image widget containing the current image of the environment.
        self.image = self.to_photo_image(gui.env.current_frame())
        self.image_label = tk.Label(self, image=self.image)
        self.image_label.grid(row=0, column=0, sticky=tk.NSEW)

        # Create a button to perform the planning iteration.
        self.reset_btn = tk.Button(
            self, text='Reset', height=2, bg=self.gui.white,
            command=self.reset_env_and_agent
        )
        self.reset_btn.grid(row=1, column=0, sticky=tk.NSEW)

        # Create a button to perform the planning iteration.
        self.next_iter_btn = tk.Button(
            self, text='Next planning step', height=2, bg=self.gui.white,
            command=self.perform_next_planning_step
        )
        self.next_iter_btn.grid(row=2, column=0, sticky=tk.NSEW)

        # Create a button to finish planning.
        self.finish_btn = tk.Button(
            self, text='Finish planning', height=2, bg=self.gui.white,
            command=self.finish_planning
        )
        self.finish_btn.grid(row=3, column=0, sticky=tk.NSEW)

        # Create a button to perform the best action.
        self.perform_action_btn = tk.Button(
            self, text='Perform action', height=2, bg=self.gui.white,
            command=self.perform_action
        )
        self.perform_action_btn.grid(row=4, column=0, sticky=tk.NSEW)

        # Create an empty column.
        self.grid_columnconfigure(1, minsize=50)

        # Create the canvas that will display the tree.
        self.canvas = tk.Canvas(self, bg=self.gui.white, width=800, height=400)
        self.canvas.grid(row=0, column=2, rowspan=5, columnspan=1, sticky=tk.NSEW)
        self.canvas.bind("<Button-1>", self.go_to_parent)

        # Create the button representing the root temporal slice.
        self.root_btn = tk.Button(
            self, text=self.get_temporal_slice_name(self.current_action_seq),
            width=6, height=4, bg=self.gui.gray, command=self.display_temporal_slice
        )

        # Create buttons representing the root's children.
        self.children_btns = []
        for i in range(gui.env.n_actions):
            self.children_btns.append(tk.Button(
                self, text=self.get_temporal_slice_name(self.current_action_seq + str(i)),
                width=6, height=4, bg=self.gui.gray, command=self.go_to_child
            ))

        # Lists of the objects' indices in the canvas.
        self.lines_ids = []
        self.children_ids = []

        # Draw the current tree.
        self.arrow_id = self.canvas.create_line(400, 45, 400, 15, arrow=tk.LAST, width=5)
        self.root_id = self.canvas.create_window(400, 100, window=self.root_btn)
        shifts = [-45, -17, 17, 45]
        for i in range(self.gui.env.n_actions):
            lines_id = self.canvas.create_line(400 + shifts[i], 145, 100 + i * 200, 255, width=2)
            child_id = self.canvas.create_window(100 + i * 200, 300, window=self.children_btns[i])
            self.lines_ids.append(lines_id)
            self.children_ids.append(child_id)

    @staticmethod
    def get_temporal_slice_name(action_seq):
        """
        Getter.
        :param action_seq: the action sequence of the temporal slice whose name should be returned.
        :return: the name of the temporal slice corresponding to the action sequence.
        """
        return "TS(t)" if action_seq == "" else "TS({})".format(action_seq)

    @staticmethod
    def to_photo_image(image):
        """
        Transform the input image into a PhotoImage.
        :param image: a pytorch tensor.
        :return: the PhotoImage.
        """
        size = 175
        if image is None:
            return VisualisationFrame.to_photo_image(np.zeros([64, 64, 3]))
        image = np.squeeze(image).astype(np.uint8)
        return ImageTk.PhotoImage(image=Image.fromarray(image).resize((size, size)))

    def highlight_anscestors_of(self, node, descendant, button, lines_id=None):
        """
        Highlight the button and line if node is an anscestor of descendant.
        :param node: the potential anscestor.
        :param descendant: the potential descendant.
        :param button: the button to highlight if node is an anscestor of descendant.
        :param lines_id: the id of the line to highlight if node is an anscestor of descendant.
        :return: nothing.
        """
        if self.is_anscestor_of(node, descendant):
            button.config(highlightbackground=self.gui.red, highlightthickness=2)
            if lines_id is not None:
                self.canvas.itemconfig(lines_id, fill=self.gui.red, width=2)
        else:
            button.config(highlightbackground=self.gui.gray, highlightthickness=1)
            if lines_id is not None:
                self.canvas.itemconfig(lines_id, fill=self.gui.black, width=1)

    @staticmethod
    def is_anscestor_of(anscestor, descendant):
        """
        Check if a node is an anscestor of another node.
        :param anscestor: the potential anscestor.
        :param descendant: the potential descendant.
        :return: if node is an anscestor of descendant.
        """
        while descendant is not None:
            if descendant == anscestor:
                return True
            descendant = descendant.parent
        return False

    def go_to_child(self):
        """
        Go to the requested child.
        :return: nothing.
        """
        # Check if there are children to go to.
        if not self.gui.current_ts.children:
            return

        # Go to the requested child.
        x = self.winfo_pointerx() - self.winfo_rootx()
        for i in range(self.gui.env.n_actions):
            if x < 450 + i * 200:
                self.gui.current_ts = self.gui.current_ts.children[i]
                self.current_action_seq += str(self.gui.current_ts.action)
                break

        # Refresh the visualisation frame.
        self.refresh()

    def go_to_parent(self, _):
        """
        Go to the parent node.
        :return: nothing.
        """
        # Check if there is a parent to go to.
        if self.gui.current_ts.parent is None:
            return

        # Go to the requested child.
        x = self.winfo_pointerx() - self.winfo_rootx()
        y = self.winfo_pointery() - self.winfo_rooty()

        if 630 < x < 650 and 15 < y < 45:
            self.gui.current_ts = self.gui.current_ts.parent
            self.current_action_seq = self.current_action_seq[:-1]

        # Refresh the visualisation frame.
        self.refresh()

    def reset_env_and_agent(self):
        """
        Reset the environment and the agent before to refresh the frame.
        :return: nothing.
        """
        # Reset the environment and the agent.
        obs = self.gui.env.reset()
        self.gui.agent.reset(obs)

        # Update current image.
        self.image = self.to_photo_image(self.gui.env.current_frame())
        self.image_label.config(image=self.image)

        # Update current temporal slice.
        self.gui.current_ts = self.gui.agent.ts
        self.current_action_seq = ""
        self.node_selected_by_uct = None

        # Refresh the visualisation frame.
        self.refresh()

    def perform_next_planning_step(self):
        """
        Perform one planning iteration.
        :return: nothing.
        """
        # Perform one planning iteration.
        self.node_selected_by_uct = self.gui.agent.mcts.select_node(self.gui.agent.ts)
        e_nodes = self.gui.agent.mcts.expansion(self.node_selected_by_uct)
        self.gui.agent.mcts.evaluation(e_nodes)
        self.gui.agent.mcts.propagation(e_nodes)
        self.planning_step += 1

        # Refresh the visualisation frame.
        self.refresh()

    def finish_planning(self):
        """
        Finish the MCTS planning.
        :return: nothing.
        """
        for i in range(self.planning_step, self.gui.agent.max_planning_steps):
            self.perform_next_planning_step()

    def display_temporal_slice(self):
        """
        Display the temporal slice frame.
        :return: nothing.
        """
        self.gui.show_frame("TemporalSliceFrame")

    def perform_action(self):
        """
        Perform the current best action in the environment.
        :return: nothing.
        """
        # Check if at least one planning iteration was performed.
        if not self.gui.agent.ts.children:
            return

        # Select and execute best action.
        action = max(self.gui.agent.ts.children, key=lambda x: x.visits).action
        obs = self.gui.env.execute(action)
        self.gui.agent.update(action, obs)

        # Update current image of the environment.
        self.image = self.to_photo_image(self.gui.env.current_frame())
        self.image_label.config(image=self.image)

        # Update current temporal slice.
        self.planning_step = 0
        self.gui.current_ts = self.gui.agent.ts
        self.current_action_seq = ""
        self.node_selected_by_uct = None

        # Refresh the visualisation frame.
        self.refresh()

    def refresh_callback(self, _):
        """
        Call the refresh function.
        :return: nothing.
        """
        self.refresh()

    def refresh(self):
        """
        Refresh the visualisation frame.
        :return: nothing.
        """
        # Update parent arrow.
        if self.gui.current_ts.parent is None:
            self.canvas.itemconfig(self.arrow_id, fill=self.gui.orange)
        else:
            self.canvas.itemconfig(self.arrow_id, fill=self.gui.gray)

        # Update root node.
        self.root_btn.config(
            bg=self.gui.gray, activebackground=self.gui.light_gray,
            text=self.get_temporal_slice_name(self.current_action_seq)
        )
        self.highlight_anscestors_of(self.gui.current_ts, self.node_selected_by_uct, self.root_btn)

        # Update children nodes and lines.
        for i in range(self.gui.env.n_actions):

            # Reset the buttons border to gray.
            self.children_btns[i].config(highlightbackground=self.gui.gray, highlightthickness=1)

            # If there are no children, then set orange backgrounds.
            if len(self.gui.current_ts.children) == 0:
                self.children_btns[i].config(
                    bg=self.gui.orange, activebackground=self.gui.light_orange, text="None"
                )
                self.canvas.itemconfig(self.lines_ids[i], fill=self.gui.orange)
                continue

            # If there are children, then set gray backgrounds.
            self.children_btns[i].config(
                bg=self.gui.gray,
                activebackground=self.gui.light_gray,
                text=self.get_temporal_slice_name(self.current_action_seq + str(i))
            )

            # Highlit children that are anscestor of the node selected by the UCT criterion.
            self.highlight_anscestors_of(
                self.gui.current_ts.children[i], self.node_selected_by_uct,
                self.children_btns[i], self.lines_ids[i]
            )

        # Set a green background to the action that would be selected to be performed.
        if self.gui.agent.ts == self.gui.current_ts and self.gui.current_ts.children != []:
            selected_action = max(self.gui.current_ts.children, key=lambda x: x.visits).action
            self.children_btns[selected_action].config(
                bg=self.gui.green, activebackground=self.gui.light_green
            )
