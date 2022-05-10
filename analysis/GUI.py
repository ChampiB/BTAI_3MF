import tkinter as tk
from analysis.frames.VisualisationFrame import VisualisationFrame
from analysis.frames.TemporalSliceFrame import TemporalSliceFrame
from analysis.widgets.NavBar import NavBar


#
# A class representing the GUI used for analysing the model.
#
class GUI:

    def __init__(self, env, agent):
        """
        Construct the graphical user interface used to analyse the model.
        :param env: the dSprites environment.
        :param agent: the BTAI_3MF agent.
        """

        # Store the environment and the agent.
        self.env = env
        self.agent = agent

        # Get initial observation and initialise the agent.
        obs = env.reset()
        agent.reset(obs)
        self.current_ts = agent.ts

        # Create the main window.
        self.window = tk.Tk()
        self.window.title("BTAI_3MF analysis")
        self.window.geometry(self.get_screen_size())

        # Colors.
        self.white = "#ffffff"
        self.gray = "#d5d5d5"
        self.light_gray = "#e9e9e9"
        self.orange = "#de9b00"
        self.light_orange = "#f2af14"
        self.black = "#000000"
        self.red = "#e00000"
        self.blue = "#003366"
        self.green = "#096100"
        self.light_green = "#1d7500"

        # Create the navigation bar.
        self.navbar = NavBar(self)
        self.window.config(menu=self.navbar)

        # Create the frame container.
        self.container = tk.Frame(self.window)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(2, weight=1)
        self.container.grid_columnconfigure(2, weight=1)

        # The dictionary of frames' constructor.
        self.frames_classes = {
            "VisualisationFrame": VisualisationFrame,
            "TemporalSliceFrame": TemporalSliceFrame
        }

        # The list of currently loaded frames.
        self.frames = {}
        self.current_frame = None

        # Show the page used to load the model and dataset.
        self.show_frame("VisualisationFrame")

    def get_screen_size(self):
        """
        Getter.
        :return: the screen' size.
        """
        screen_size = str(self.window.winfo_screenwidth() - 85)
        screen_size += "x"
        screen_size += str(self.window.winfo_screenheight() - 75)
        screen_size += "+85+35"
        return screen_size

    def update_navbar(self):
        """
        Update the navigation bar.
        :return: nothing.
        """
        self.navbar = NavBar(self)
        self.window.config(menu=self.navbar)

    def show_frame(self, frame_name):
        """
        Show a frame for the given frame name.
        :param frame_name: the name of the frame to show.
        :return: nothing.
        """
        # Construct the frame if it does not already exist.
        if frame_name not in self.frames.keys():
            frame = self.frames_classes[frame_name](parent=self.container, gui=self)
            self.frames[frame_name] = frame

        # Display the requested frame.
        if self.current_frame is not None:
            self.current_frame.grid_forget()
        self.current_frame = self.frames[frame_name]
        self.current_frame.grid(row=1, column=1, sticky="")
        self.current_frame.refresh()
        self.current_frame.tkraise()

    def loop(self):
        """
        Launch the main loop of the graphical user interface.
        :return: nothing.
        """
        self.window.mainloop()
