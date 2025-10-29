###### IMPORTS #########
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox,scrolledtext
from tkinter.simpledialog import askstring
import pandas as pd
from joblib import dump
import warnings
from ..registry.model_registry import ModelRegistry
from .interface import Interface
from .interface_kmeans import InterFaceKMeans

############ REMOVE IF YOU WANT TO SEE WARNINGS SUCH AS RUNTIMEWARNINGS,
#                UNDEFINEDMETRICWARNINGS and CONVERGENCEWARNINGS.

warnings.filterwarnings("ignore")

####################################


class InterFaceTkinter(Interface):
    def __init__(self, root):

        self.root = root
        self.root.title("Neuron Cluster Classifier")
        self.model_type = tk.StringVar(value="KMeans")
        self.create_widgets()
        self.result = None
    

    def create_widgets(self):
        """
        Creates all widgets for tkinter frame
        
        """

        tk.Label(self.root,
                  text="Model type:").grid(row=1, column=2, sticky="w")
        tk.Radiobutton(self.root,
             text="KMeans", variable=self.model_type,
                   value="KMeans").grid(row=2, column=1, sticky="w")
        
        tk.Radiobutton(self.root,
         text="DEC", variable=self.model_type,
                value="DEC").grid(row=2, column=3, sticky="w")

        tk.Button(self.root, text="Continue",
                   command=self.run).grid(row=5, columnspan=3)
        tk.Button(self.root, text="Quit",
                     command=self.root.destroy).grid(row=5, columnspan=1)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def run(self):
        try:
            if self.model_type.get() == 'KMeans':
                print('HEYHEY')
                
                dlg = InterFaceKMeans(self.root)
                new_win = dlg.window
                dlg.setup()

                self.result = dlg.config
                if self.result is None:
                    messagebox.showinfo("Info", "Dialog cancelled")
                else:
                    print("Collected:", self.result)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def setup(self):
        self.root.mainloop()
