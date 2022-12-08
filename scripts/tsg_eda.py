#%%
import pickle
import numpy as np
import matplotlib.pylab as plt

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import DATA_PATH
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from tqdm.autonotebook import tqdm

DISPLAY_FIGS = True

FILENAME = "tsg_eda"

OUT_PATH = OUT_PATH / FILENAME

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


#%%

# import napari as nap

# Load model

# Load neuron names
neuron_names = np.loadtxt(DATA_PATH / "tsg/savedModelData/neuronNames.csv", dtype=str)

n_models = 5

all_model_data = {}
all_rate_data = []
for model_id in range(n_models):
    with open(DATA_PATH / f"tsg/savedModelData/{model_id}.pkl", "rb") as f:
        model_data = pickle.load(f)
        all_model_data[model_id] = model_data
        rates = model_data["rates"]
        for batch in range(rates.shape[-1]):
            rates_for_batch = rates[:, :, batch]  # n length vector
            rate_df = pd.DataFrame(data=rates_for_batch, columns=neuron_names)
            rate_df.index.name = "time"
            rate_df.columns.name = "neuron_rate"
            rate_df["batch"] = batch
            rate_df["model_id"] = model_id
            rate_df = rate_df.reset_index().set_index(["model_id", "batch", "time"])

            # rate_data = pd.Series(
            #     index=neuron_names, data=rate_vector, name="rate"
            # ).to_frame()
            # rate_data["batch"] = batch
            # rate_data["t"] = t
            # rate_data["model_id"] = model_id

            all_rate_data.append(rate_df)


#%%
rate_df = pd.concat(all_rate_data)

#%%
rate_df.shape

#%%

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = pca.fit_transform(rate_df)
X_df = pd.DataFrame(data=X, index=rate_df.index)
X_df.head()

#%%

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection


def timeplot(ax, points, cmap="RdBu", zorder=1, color=None):

    t = np.linspace(0, 1, len(points))  # your "time" variable
    # set up a list of (x,y) points
    plot_points = points.transpose().reshape(-1, 1, 2)

    # set up a list of segments
    segs = np.concatenate([plot_points[:-1], plot_points[1:]], axis=1)
    if color is not None:
        colors = len(points) * [color]
    else:
        colors = plt.get_cmap(cmap)(t)
    # make the collection of segments
    lc = LineCollection(segs, colors=colors, lw=1, zorder=zorder)
    lc.set_array(t)  # color the segments by our parameter
    # for i in range(len(points)):
    #     ax.quiver(
    #         points[:-1, 0],
    #         points[:-1, 1],
    #         points[1:, 0] - points[:-1, 0],
    #         points[1:, 1] - points[:-1, 1],
    #         scale_units="xy",
    #         angles="xy",
    #         scale=1,
    #         color=colors[i],
    #     )

    # plot the collection
    ax.add_collection(lc)  # add the collection to the plot

    # x = points[:, 0]
    # y = points[:, 1]
    # pad = 1
    # if ax.get_xlim()[0] > x.min():
    #     ax.set_xlim(x.min() - pad, ax.get_xlim()[1])
    # if ax.get_xlim()[1] < x.max():
    #     ax.set_xlim(ax.get_xlim()[0], x.max() + pad)
    # if ax.get_ylim()[0] > y.min():
    #     ax.set_ylim(y.min() - pad, ax.get_ylim()[1])
    # if ax.get_ylim()[1] < y.max():
    #     ax.set_ylim(ax.get_ylim()[0], y.max() + pad)

    ax.set(xticks=[], yticks=[])


#%%

set_theme()

size = 5
n_cols = 5
n_rows = int(np.ceil(n_models / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * size, n_rows * size))

n_batches = 30

reset_index = 177

colors = sns.color_palette("husl", n_colors=n_batches)


for model in range(n_models):
    model_X_df = X_df.loc[model]
    for batch in [0, 10, 20, 29]:
        batch_X_df = model_X_df.loc[batch]
        timeplot(axs.flat[model], batch_X_df.values[:reset_index], color=colors[batch])

for ax in axs.flat:
    ax.set_xlim(X_df[0].min() - 2, X_df[0].max() + 2)
    ax.set_ylim(X_df[1].min() - 2, X_df[1].max() + 2)

fig.set_facecolor("w")

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
timeplot(ax, batch_X_df.values[:177])
ax.set_xlim(X_df[0].min() - 2, X_df[0].max() + 2)
ax.set_ylim(X_df[1].min() - 2, X_df[1].max() + 2)
#%%
# # Get data
# dt = 0.5
# weights = model_data["weights"]
# rates = model_data["rates"]
# ISIs = model_data["ISIs"]


# KC_rates = model_data["KC_rates"]
# US_rates = model_data["US_rates"]
# US_rates[:, 1, :] = -US_rates[:, 1, :]
# output_valence = model_data["output_valence"]
# target_output = model_data["target_output"]
# adjacencyMatrix = model_data["adjacencyMatrix"]


#%%

out = np.einsum("tib,ij->tijb", rates, weights)

#%%
for t in range(rates.shape[0]):
    for batch in range(rates.shape[-1]):
        rate_vector = rates[t, :, batch]  # n length vector
        flow_weights = rate_vector[:, None] * weights

        einsum_flow_weights = out[t, :, :, batch]

        if np.linalg.norm(flow_weights - einsum_flow_weights) > 0.0:
            raise ValueError


#%%


def timeplot(ax, x, y, cmap="RdBu", zorder=1):
    t = np.linspace(0, 1, x.shape[0])  # your "time" variable

    # set up a list of (x,y) points
    points = np.array([x, y]).transpose().reshape(-1, 1, 2)

    # set up a list of segments
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = plt.get_cmap(cmap)(t)
    # make the collection of segments
    lc = LineCollection(segs, colors=colors, lw=3, zorder=zorder)
    lc.set_array(t)  # color the segments by our parameter
    # for i in range(len(points)):
    #     ax.quiver(
    #         points[:-1, 0],
    #         points[:-1, 1],
    #         points[1:, 0] - points[:-1, 0],
    #         points[1:, 1] - points[:-1, 1],
    #         scale_units="xy",
    #         angles="xy",
    #         scale=1,
    #         color=colors[i],
    #     )

    # plot the collection
    ax.add_collection(lc)  # add the collection to the plot

    ax.set_xlim(
        x.min() - 0.1, x.max() + 0.1
    )  # line collections don't auto-scale the plot
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.set(xticks=[], yticks=[])


from pkg.plot import set_theme

set_theme()

n_batches = 10
n_components = 2
reset_index = 177
fig, axs = plt.subplots(int(np.ceil(n_batches / 5)), 5, figsize=(20, 10))
for batch in range(n_batches):
    batch_rates = rates[:, :, batch]

    pca = PCA(n_components=n_components)
    lowd_rates = pca.fit_transform(batch_rates)

    timeplot(
        axs.flat[batch],
        lowd_rates[reset_index:, 0],
        lowd_rates[reset_index:, 1],
        cmap="Purples",
        zorder=3,
    )
    timeplot(
        axs.flat[batch],
        lowd_rates[:reset_index, 0],
        lowd_rates[:reset_index, 1],
        cmap="GnBu",
    )

fig.set_facecolor("w")

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))


def quiverplot(points, colors=None, ax=None):
    q = ax.quiver(
        points[:-1, 0],
        points[:-1, 1],
        points[1:, 0] - points[:-1, 0],
        points[1:, 1] - points[:-1, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        color=colors,
        linewidths=0,
    )
    return q


model_id = 10
batch = 10
with open(DATA_PATH / f"tsg/savedModelData/{model_id}.pkl", "rb") as f:
    model_data = pickle.load(f)
    rates = model_data["rates"]
    batch_rates = rates[:, :, batch]
    pca = PCA(n_components=n_components)
    lowd_rates = pca.fit_transform(batch_rates)

points = lowd_rates

reset_index = 179  # ?

colors = list(sns.color_palette("Reds", n_colors=len(points)))
quiverplot(points[:reset_index], colors="red", ax=ax)

colors = list(sns.color_palette("Blues", n_colors=len(points)))
quiverplot(points[reset_index:], colors="blue", ax=ax)

#%%
groups = []
for name in neuron_names:
    group = name.split("-")[0]
    groups.append(group)

#%%
from matplotlib import animation

# from graspologic.plot import matrixplot
from giskard.plot import matrixplot


reset_index = 179  # ?

model_id = 0
batch = 0
for model_id in [0, 5, 6]:
    for batch in [0, 9, 19, 29]:
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
            0.5, 1.03, f"Model={model}, batch={batch}", ha="center", fontsize="x-large"
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
        colors = reset_index * [colors[0]] + (len(points) - reset_index) * [colors[1]]
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
        f = FIG_PATH / f"activity_heatmap_pca_model={model}_batch={batch}.gif"
        writergif = animation.PillowWriter(fps=30)
        anim.save(f, writergif)

#%%
fig = plt.figure()
fig.set_facecolor("w")
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
lines = ax.plot([], [], lw=2)
pad = 2
ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)

t = np.linspace(0, 1, len(points))  # your "time" variable
# set up a list of (x,y) points
plot_points = points.transpose().reshape(-1, 1, 2)

# set up a list of segments
segs = np.concatenate([plot_points[:-1], plot_points[1:]], axis=1)


# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return (line,)

# lines =


def animate(i):
    lc = LineCollection(segs[:i], colors=colors, lw=3)
    # x = points[:i]
    # y = points[:i]
    # line.set_data(x, y)
    ax.add_collection(lc)
    return (lc,)


anim = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True)

f = FIG_PATH / "test_animate_line.gif"
writergif = animation.PillowWriter(fps=30)
anim.save(f, writergif)

#%%

# Plot task stimuli
batch = 1
fig, ax = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
ax = ax.flatten()
ax[0].plot(KC_rates[:, :, batch])
ax[0].set_title("Firing rates of Kenyon cells over time")
ax[2].plot(US_rates[:, :, batch])
ax[2].set_title("Valence of unconditioned stimuli (i.e. reward [+] or punishment [-])")
ax[1].plot(output_valence[:, :, batch], label="Output")
ax[1].plot(
    target_output[:, :, batch], "--", alpha=0.5, color="black", label="Target output"
)
ax[1].legend()
ax[1].set_title("Output valence of MBON activity")
ax[3].imshow(rates[:, :, batch].T, aspect="auto")
ax[3].set_title("Firing rates of all neurons over time")
ax[3].set_ylabel("Neuron ID")
[ax[i + 1].set_ylim([-1.1, 1.1]) for i in range(2)]
[ax[i].set_ylabel("Firing rate (0-1)") for i in range(3)]
[ax[i].set_xlabel("Time (seconds)") for i in range(4)]
plt.show()

# # View adjacency matrix
# v = nap.Viewer()
# v.add_image(adjacencyMatrix[:,:,:,batch])
# v.add_image(b[:,:,:,batch])

# View model weights
fig = plt.figure(figsize=(14, 10))
plt.imshow(weights, aspect="auto")
plt.xlabel("Neuron ID")
plt.ylabel("Neuron ID")
plt.title("Synaptic weight matrix")
plt.show()

# %%
