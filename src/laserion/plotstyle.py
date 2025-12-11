from matplotlib import rcParams
from cycler import cycler


def std_plot(**kwargs):
    rcParams["font.family"] = "Helvetica"
    rcParams["mathtext.default"] = "regular"
    rcParams["svg.fonttype"] = "none"
    rcParams["axes.linewidth"] = 2
    rcParams["figure.figsize"] = [10, 8]
    rcParams["font.size"] = 23
    rcParams["lines.linewidth"] = 2
    rcParams["lines.color"] = "k"
    rcParams["xtick.direction"] = "in"
    rcParams["xtick.top"] = True
    rcParams["xtick.minor.visible"] = True
    rcParams["ytick.direction"] = "in"
    rcParams["ytick.right"] = True
    rcParams["ytick.minor.visible"] = True
    rcParams["xtick.major.size"] = 10
    rcParams["xtick.minor.size"] = 5
    rcParams["xtick.major.width"] = 2
    rcParams["xtick.minor.width"] = 2
    rcParams["ytick.major.size"] = 10
    rcParams["ytick.minor.size"] = 5
    rcParams["ytick.major.width"] = 2
    rcParams["ytick.minor.width"] = 2
    rcParams["figure.autolayout"] = True

    rcParams["axes.prop_cycle"] = cycler("color", ["k", (0.8, 0.0, 0.0), (0.0, 0.0, 0.8), (0.0, 0.8, 0.0), (0.8, 0.8, 0.0), (0.8, 0.0, 0.8), (0.0, 0.8, 0.8), "r", "b", "g", "c", "m", "y"])

    # optional overrides
    for k, v in kwargs.items():
        rcParams[k] = v
