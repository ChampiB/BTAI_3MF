import tkinter as tk
from tkinter import messagebox
import torch


#
# Class representing the main navigation bar.
#
class NavBar(tk.Menu):

    def __init__(self, gui):
        """
        Construct the main navigation bar.
        :param gui: the graphical user interface.
        """
        # Call super class contructor.
        super().__init__(gui.window)

        # Store the graphical user interface.
        self.gui = gui

        # Add the visualisation tab to the navigation bar.
        self.add_command(label="Visualisation", command=self.visualisation_cmd)

        # Add the model tab to the navigation bar, if needed.
        self.add_command(label="Temporal Slice", command=self.temporal_slice_cmd)

    def visualisation_cmd(self):
        """
        Display the page used to visualise BTAI_3MF.
        :return: nothing.
        """
        self.gui.show_frame("VisualisationFrame")

    def temporal_slice_cmd(self):
        """
        Display the page used to visualise the latent space of the model.
        :return: nothing.
        """
        self.gui.show_frame("TemporalSliceFrame")
