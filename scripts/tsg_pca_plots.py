#%%
import pickle
import time

import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import matrixplot
from matplotlib import animation
from pkg.data import DATA_PATH
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from tqdm.autonotebook import tqdm
from sklearn.decomposition import PCA

DISPLAY_FIGS = True

FILENAME = "tsg_pca_plots"

OUT_PATH = OUT_PATH / FILENAME

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


set_theme()

#%%

neuron_names = np.loadtxt(DATA_PATH / "tsg/savedModelData/neuronNames.csv", dtype=str)

#%%
groups = []
for name in neuron_names:
    group = name.split("-")[0]
    groups.append(group)


reset_index = 179  # ?

n_components = 2


models = [0, 5, 6]
batches = [0, 9, 19, 29]

with tqdm(total=len(models) * len(batches)) as pbar:
    for model_id in models:
        for batch in batches:

            with open(DATA_PATH / f"tsg/savedModelData/{model_id}.pkl", "rb") as f:
                model_data = pickle.load(f)

            rates = model_data["rates"]
            batch_rates = rates[:, :, batch]
            pca = PCA(n_components=n_components)
            points = pca.fit_transform(batch_rates)

            X = points[:-1, 0]
            Y = points[:-1, 1]
            U = points[1:, 0] - points[:-1, 0]
            V = points[1:, 1] - points[:-1, 1]

            fig, axs = plt.subplots(
                1,
                2,
                figsize=(15, 7),
                gridspec_kw=dict(width_ratios=[2, 1]),
                constrained_layout=True,
            )
            fig.set_facecolor("w")

            ax = axs[0]
            matrixplot(
                data=batch_rates.T,
                ax=ax,
                row_sort_class=groups,
                cbar=False,
                tick_fontsize=20,
                cmap="RdBu_r",
            )
            ax.set_xlabel("Time")
            ax.set_xticks([0, 50, 100, 150, 200, 250])
            ax.set_xticklabels([0, 50, 100, 150, 200, 250], rotation=0)
            ax.xaxis.set_label_position("bottom")
            ax.tick_params(which="both", length=5)

            vline = ax.axvline(0, color="darkred")

            fig.text(
                0.5,
                1.03,
                f"Model={model_id}, batch={batch}",
                ha="center",
                fontsize="x-large",
            )

            ax.annotate(
                "Reset",
                (reset_index, 0),
                xytext=(0, 30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-|>"),
                annotation_clip=False,
                ha="center",
            )

            ax = axs[1]
            t = 1
            U_t = U.copy()
            U_t[t:] = 0
            V_t = V.copy()
            V_t[t:] = 0

            q = ax.quiver(
                X,
                Y,
                U_t,
                V_t,
                scale_units="xy",
                angles="xy",
                scale=1,
                color="white",
                linewidths=0,
                headwidth=3,
                width=0.004,
            )

            colors = sns.color_palette()
            reset_index = 178
            colors = reset_index * [colors[0]] + (len(points) - reset_index) * [
                colors[1]
            ]
            time_colors = len(points) * ["w"]
            ax.set(xticks=[0], yticks=[0], xlabel="PCA 1", ylabel="PCA 2")
            ax.tick_params(which="both", length=5)

            def update_quiver(t):

                U_t[t] = U[t]
                V_t[t] = V[t]
                q.set_UVC(U_t, V_t)
                time_colors[t] = colors[t]
                # q.set_array(edgecolors)
                q.set_edgecolors(time_colors)
                q.set_facecolors(time_colors)

                vline.set_xdata([t, t])
                return (q, vline)

            anim = animation.FuncAnimation(
                fig, update_quiver, frames=len(points) - 1, interval=40, blit=True
            )
            f = FIG_PATH / f"activity_heatmap_pca_model={model_id}_batch={batch}.gif"
            writergif = animation.PillowWriter(fps=30)
            anim.save(f, writergif)
            pbar.update(1)
