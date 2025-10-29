###### IMPORTS #########
from .interface import Interface
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox,scrolledtext
from tkinter.simpledialog import askstring
import pandas as pd
from joblib import dump
import warnings
#from ..registry.model_registry import ModelRegistry


############ REMOVE IF YOU WANT TO SEE WARNINGS SUCH AS RUNTIMEWARNINGS,
#                UNDEFINEDMETRICWARNINGS and CONVERGENCEWARNINGS.

warnings.filterwarnings("ignore")

####################################


class InterFaceKMeans(Interface):
    def __init__(self, root):
        #self.registry = registry
        self.root = root
        self.window = tk.Toplevel(self.root)
        self.root.title("KMeans")
        self.filepath = tk.StringVar()
        self.n_clusters = tk.IntVar(value=8)
        self.max_iter = tk.IntVar(value=300)
        self.tol = tk.DoubleVar(value=0.0001)
        self.algorithm = tk.StringVar(value='lloyd')
        self.scaler = tk.BooleanVar()
        self.use_dummies = tk.BooleanVar()
        self.create_widgets()
        self.config = None
        

    def create_widgets(self):
        """
        Creates all widgets for tkinter frame
        
        """

        tk.Label(self.root, text="CSV-file:").grid(row=0, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.filepath).grid(row=0, column=1)
        tk.Button(self.root, text="Files",
                   command=self.browse_file).grid(row=0, column=2)

        #tk.Label(self.root, text="Choose model:").grid(row=1,
        #                                                 column=0, sticky="w")
        #self.dependent_reg = tk.OptionMenu(self.root,
        #                                     self.dependent_reg, "")
        #self.dependent_reg.grid(row=1, column=1)

        tk.Label(self.root, text="Iterations:").grid(row=2,
                                                         column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.max_iter).grid(row=2, column=1)

        #tk.Checkbutton(self.root, text="Use polynomial features",
        #  variable=self.use_polynomial).grid(row=3, columnspan=1, sticky="w")

        tk.Label(self.root,
                  text="Nr of clusters:").grid(row=7, column=0, sticky="w")
        tk.Entry(self.root, textvariable=self.n_clusters).grid(row=7, column=1)
        tk.Label(self.root, text="Algorithm:").grid(row=4,
                                                     column=0, sticky="w")
        tk.Radiobutton(self.root,
         text="lloyd", variable=self.algorithm,
                value="lloyd").grid(row=4, column=1, sticky="w")
        tk.Radiobutton(self.root,
         text="elkan", variable=self.algorithm,
                value="elkan").grid(row=4, column=2, sticky="w")

        tk.Checkbutton(self.root,
                        text="Use scaler",
             variable=self.scaler).grid(row=5, columnspan=1, sticky="w")

        tk.Label(self.root, text="Tolerance:").grid(row=6, column=0,
                                                             sticky="w")
        tk.Entry(self.root, textvariable=self.tol).grid(row=6,
                                                              column=1)

        tk.Button(self.root, text="Run",
                   command=self.run).grid(row=8, columnspan=3)
        tk.Button(self.root, text="Quit",
                     command=self.root.destroy).grid(row=10, columnspan=1)
        tk.Checkbutton(self.root,
                        text="Use dummy-columns",
             variable=self.use_dummies).grid(row=5, columnspan=1, sticky="w")


        #self.root.grid_rowconfigure(0, weight=1)
        #self.root.grid_columnconfigure(0, weight=1)

    def check_dummies(self,X:pd.DataFrame,use_dummies:bool)->pd.DataFrame:
        """
        Checks for categorical columns in X.
        """
        cat_cols = [col for col in X.columns if X[col].dtype in ('object', 'category')]

        if cat_cols:
            if use_dummies:
                X = pd.get_dummies(X)
            else:
                raise ValueError(
                    f"Columns '{', '.join(cat_cols)}' are categorical.\n"
                    "Please use dummy-columns.")
        return X

    def browse_file(self):
        self.filepath.set(filedialog.askopenfilename())
        #self.load_columns()

    def check_nan(self,df):
        if df.isnull().values.any():
            raise ValueError(f"""Data contains NaN-values in columns:\n 
                    '{', '.join(df.columns[df.isna().any()].tolist())}'.\n
                                             Please fix before training.""")
        self.root.quit
    def run(self):
        try:
            X = pd.read_csv(self.filepath.get())
            self.check_nan(df=X)
            ### Check if dummies ###
            X = self.check_dummies(X, self.use_dummies.get())

            self.config = {'n_clusters':self.n_clusters.get(),
                      'max_iter':self.max_iter.get(),
                      'scaler':self.scaler.get(),
                      'X_train':X,
                      'algorithm':self.algorithm.get(),
                      'tol':self.tol.get()}
            print('DONE')
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def setup(self):
        self.root.mainloop()

#if __name__ == '__main__':
#    root = tk.Tk()
#    app = InterFaceKMeans(root)
#    root.mainloop()