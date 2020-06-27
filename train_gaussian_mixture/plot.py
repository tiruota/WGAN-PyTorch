# -*- coding: utf-8 -*-
import sampler, pylab, os
import seaborn as sns
from opt import get_opt
import numpy as np

sns.set(font_scale=2)
sns.set_style("white")
opt = get_opt()

def plot_kde(data, dir=None, filename="kde", color="Greens"):
    if dir is None:
        raise Exception()
    try:
        os.mkdir(dir)
    except:
        pass
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    bg_color  = sns.color_palette(color, n_colors=256)[0]
    ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
    ax.set_facecolor(bg_color)
    kde = ax.get_figure()
    pylab.xlim(-4, 4)
    pylab.ylim(-4, 4)
    kde.savefig("{}/{}.png".format(dir, filename))

def plot_scatter(data, dir=None, filename="scatter", color="blue"):
    if dir is None:
        raise Exception()
    try:
        os.mkdir(dir)
    except:
        pass
    fig = pylab.gcf()
    fig.set_size_inches(16.0, 16.0)
    pylab.clf()
    pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
    pylab.xlim(-4, 4)
    pylab.ylim(-4, 4)
    pylab.savefig("{}/{}.png".format(dir, filename))

def main():
    num_samples = 10000
    samples_true = sampler.gaussian_mixture_circle(num_samples, num_cluster=8, scale=2, std=0.2)
    samples_true2 = sampler.gaussian_mixture_double_circle(num_samples, num_cluster=8, scale=2, std=0.2)
    plot_scatter(samples_true, "./results", "scatter_true")
    plot_kde(samples_true, "./results", "kde_true")
    plot_scatter(samples_true2, "./results", "scatter_true2")
    plot_kde(samples_true2, "./results", "kde_true2")
if __name__ == "__main__":
    main()