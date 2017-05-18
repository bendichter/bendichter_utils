import itertools

import seaborn.apionly as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import matplotlib as mpl
import matplotlib.gridspec as gridspec


def my_light_palette(color, n_colors=6, reverse=False, as_cmap=False,
                  input="rgb", light_val=1):
    """Make a sequential palette that blends from light to ``color``.
    This kind of palette is good for data that range between relatively
    uninteresting low values and interesting high values.
    The ``color`` parameter can be specified in a number of ways, including
    all options for defining a color in matplotlib and several additional
    color spaces that are handled by seaborn. You can also use the database
    of named colors from the XKCD color survey.
    If you are using the IPython notebook, you can also choose this palette
    interactively with the :func:`choose_light_palette` function. Taken from
    seaborn with an additional light_val argument which allows you to control
    the lightness of the lightest value. Is 1, be defualt, which is white
    Parameters
    ----------
    color : base color for high values
        hex code, html color name, or tuple in ``input`` space.
    n_colors : int, optional
        number of colors in the palette
    reverse : bool, optional
        if True, reverse the direction of the blend
    as_cmap : bool, optional
        if True, return as a matplotlib colormap instead of list
    input : {'rgb', 'hls', 'husl', xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.
    light_val : double, optional
        Controls the lightness of the lightest value. Is 1, be defualt,
        which is white
    Returns
    -------
    palette or cmap : seaborn color palette or matplotlib colormap
        List-like object of colors as RGB tuples, or colormap object that
        can map continuous values to colors, depending on the value of the
        ``as_cmap`` parameter.
    See Also
    --------
    dark_palette : Create a sequential palette with dark low values.
    diverging_palette : Create a diverging palette with two colors.
    Examples
    --------
    Generate a palette from an HTML color:
    .. plot::
        :context: close-figs
        >>> import seaborn as sns; sns.set()
        >>> sns.palplot(sns.light_palette("purple"))
    Generate a palette that increases in lightness:
    .. plot::
        :context: close-figs
        >>> sns.palplot(sns.light_palette("seagreen", reverse=True))
    Generate a palette from an HUSL-space seed:
    .. plot::
        :context: close-figs
        >>> sns.palplot(sns.light_palette((260, 75, 60), input="husl"))
    Generate a colormap object:
    .. plot::
        :context: close-figs
        >>> from numpy import arange
        >>> x = arange(25).reshape(5, 5)
        >>> cmap = sns.light_palette("#2ecc71", as_cmap=True)
        >>> ax = sns.heatmap(x, cmap=cmap)
    """
    color = sns.palettes._color_to_rgb(color, input)
    light = sns.set_hls_values(color, l=light_val)
    colors = [color, light] if reverse else [light, color]
    return sns.blend_palette(colors, n_colors, as_cmap)


cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.5, 0.0, 1.0),
                 (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue': ((0.0, 1.0, 1.0),
                  (0.5, 0.0, 1.0),
                  (0.9, 0.0, 0.1),
                  (1.0, 0.0, 0.1)),

         'alpha': ((0.0, 1.0, 1.0),
                   (0.1, 1.0, 1.0),
                   (0.45, 0.0, 0.0),
                   (0.55, 0.0, 0.0),
                   (0.9, 1.0, 1.0),
                   (1.0, 1.0, 1.0))
         }
my_blue_red = LinearSegmentedColormap('my_blue_red', cdict)


def shift_zero_bwr_colormap(z: float, transparent: bool = True):
    """shifted bwr colormap

    cmap =  shift_zero_bwr_colormap(.7)

    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 2*np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X) * np.sin(Y) * 5 + 5
    plt.plot([0, 10*np.pi], [0, 20*np.pi], color='c', lw=20, zorder=-3)
    plt.imshow(Z, interpolation='nearest', origin='lower', cmap=cmap)
    plt.colorbar()

    """
    if (z < 0) or (z > 1):
        raise ValueError('z must be between 0 and 1')

    cdict1 = {'red': ((0.0, max(-2 * z + 1, 0), max(-2 * z + 1, 0)),
                      (z, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, max(-2 * z + 1, 0), max(-2 * z + 1, 0)),
                        (z, 1.0, 1.0),
                        (1.0, max(2 * z - 1, 0), max(2 * z - 1, 0))),

              'blue': ((0.0, 1.0, 1.0),
                       (z, 1.0, 1.0),
                       (1.0, max(2 * z - 1, 0), max(2 * z - 1, 0))),
              }
    if transparent:
        cdict1['alpha'] = ((0.0, 1 - max(-2 * z + 1, 0), 1 - max(-2 * z + 1, 0)),
                           (z, 0.0, 0.0),
                           (1.0, 1 - max(2 * z - 1, 0), 1 - max(2 * z - 1, 0)))

    return LinearSegmentedColormap('shifted_rwb', cdict1)


def multi_colorbar(cmaps, vmins, vmaxs, ax=None, orientation='vertical'):
    """
    plots multiple colorbars with different vmins and vmaxs on the same axis
    elect_colors = ('blue', 'red', 'green')
    vmins=(.5, -.5, .5)
    vmaxs=(3., 3., 2.)

    cmaps = [sns.light_palette(ecolor, as_cmap=True) for ecolor in elect_colors]

    multi_colorbar(cmaps, vmins, vmaxs, orientation='horizontal')
    """

    if orientation not in ('vertical', 'horizontal'):
        raise ValueError('orientation must be either vertical or horizontal')

    if not (len(cmaps) == len(vmins) and (len(cmaps) == len(vmaxs))):
        raise ValueError('cmaps, vmins, and vmaxs must all be iterables of the same length')

    if ax is None:
        if orientation == 'horizontal':
            figsize = (3, 1)
        else:
            figsize = (1, 3)
        fig, ax = plt.subplots(figsize=figsize)

    maxmax = max(vmaxs)
    minmin = min(vmins)

    if orientation == 'vertical':
        extent = [-.5, len(cmaps) - .5, maxmax, minmin]
    else:
        extent = [minmin, maxmax, -.5, len(cmaps) - .5]

    im = []
    for cmap, vmin, vmax in zip(cmaps, vmins, vmaxs):
        yy = np.linspace(minmin, maxmax, 100)
        colors = cmap((yy - vmin) / (vmax - vmin))
        im.append(colors)
    im = np.array(im)
    if orientation == 'vertical':
        im = np.array(im).swapaxes(0, 1)
    ax.imshow(im, interpolation='nearest', extent=extent)
    ax.invert_yaxis()
    plt.axis('tight')
    if orientation == 'vertical':
        ax.set_xticks([])
    else:
        ax.set_yticks([])
    sns.despine(ax=ax)

    return ax


color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_gmm_results(gm):
    """

    :param gm: sklearn.mixture.gaussian_mixture.GaussianMixture
    :return:
    """
    if gm.covariance_type == 'spherical':
        covariances = [np.diag(np.ones(gm.means_.shape[1])) * c for c in gm.covariances_]
    elif gm.covariance_type == 'diag':
        covariances = [np.diag(c) for c in gm.covariances_]
    else:
        covariances = gm.covariances_
    vs = []
    for i, (mean, covar, color) in enumerate(zip(
            gm.means_, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(plt.gca().bbox)
        ell.set_alpha(0.5)
        plt.gca().add_artist(ell)

        vs.append(v)

    return vs


class ColorPanel:
    def __init__(self, image, xlims, ylims):
        """
        image: np.ndarray(xsize, ysize, 3, dtype=uint8) """
        if type(image) is str:
            from scipy.misc import imread
            image = imread(image)
        self.image = image
        self.xlims = xlims
        self.ylims = ylims
        self.xnorm = mpl.colors.Normalize(xlims[0], xlims[1], clip=True)
        self.ynorm = mpl.colors.Normalize(ylims[0], ylims[1], clip=True)

    def get_color(self, xvalue, yvalue):
        return self.image[(self.image.shape[1] - 1) - round(self.ynorm(yvalue) * (self.image.shape[1] - 1)),
                          round(self.xnorm(xvalue) * (self.image.shape[0] - 1))]

    def show_panel(self, xlabel=None, ylabel=None, ax=None, **kwargs):
        if ax is not None:
            out = ax.imshow(self.image, extent=[self.xlims[0], self.xlims[1], self.ylims[0], self.ylims[1]],
                            aspect=(self.xlims[1] - self.xlims[0]) / (self.ylims[1] - self.ylims[0]), **kwargs)
        else:
            out = plt.imshow(self.image, extent=[self.xlims[0], self.xlims[1], self.ylims[0], self.ylims[1]],
                             aspect=(self.xlims[1] - self.xlims[0]) / (self.ylims[1] - self.ylims[0]), **kwargs)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        return out


def cornerplot(datas, labels=None, interact_plot=None, lims=(-.1, .2), highlight=None):
    if interact_plot is None:
        def interact_plot(data2, data1):
            plt.plot(data2, data1, '.')
            if highlight is not None:
                plt.plot(data2[highlight], data1[highlight], 'r.')
            plt.plot(plt.xlim(), plt.xlim(), '--', color='grey')
            sns.despine(ax=plt.gca())

    fig, axs = plt.subplots(len(datas), len(datas), figsize=(8, 8))
    plt.subplots_adjust(hspace=.3, wspace=.3)

    for i, data1 in enumerate(datas):
        if labels:
            axs[i, 0].set_ylabel(labels[i])
        for j, data2 in enumerate(datas):
            if labels and i == (len(datas) - 1):
                axs[len(datas) - 1, j].set_xlabel(labels[j])
            if i == j:
                axs[i, j].hist(data1)
                if lims is not None:
                    axs[i, j].set_xlim(lims)
                sns.despine(ax=axs[i, j])
            elif i > j:
                plt.sca(axs[i, j])
                interact_plot(data2, data1)
                if lims is not None:
                    axs[i, j].set_xlim(lims)
                    axs[i, j].set_ylim(lims)
            elif j > i:
                axs[i, j].axis('off')


def plot_confusion_matrix(cm, labels, cmap=plt.cm.Blues, ax=None, colorbar_label=None, discrete=False):
    if ax is None:
        ax = plt.gca()
    if discrete:
        ax, cbar = discrete_matshow(cm, cmap=cmap, ax=ax)
    else:
        h = ax.pcolormesh(cm, cmap=cmap)
        cbar = plt.colorbar(h)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    tick_marks = np.arange(len(labels))+.5
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    sns.despine(ax=ax)

    return ax


def discrete_matshow(a, cmap=mpl.cm.binary, ax=None):
    if ax is None:
        ax = plt.gca()

    if type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, np.max(a) - np.min(a) + 1))
    cmap = mpl.colors.ListedColormap(colors)
    mat = ax.matshow(a, cmap=cmap, origin='lower', vmin=np.min(a) - .5, vmax=np.max(a) + .5)
    cbar = plt.colorbar(mat, ticks=np.arange(np.min(a), np.max(a) + 1))
    return ax, cbar


def plot_sig(x1, x2, sig, y=1.05, h=.03, color='salmon'):
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    plt.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=color)


def label_axes(ax, label, left=.01, up=.01, weight='bold', fontsize=14, **kwargs):
    fig = ax.get_figure()
    text_kwargs = {}
    text_kwargs.update(**kwargs)
    bounds = ax.get_position().bounds
    x = bounds[0] - left
    y = bounds[1] + bounds[3] + up
    ax.text(x, y, label, transform=fig.transFigure, weight=weight, fontsize=fontsize, **kwargs)


def subplot_spec_labeler(subplot_spec, fig=None):
    if fig is None:
        fig = plt.gcf()
    ax = fig.add_subplot(subplot_spec)
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def use_subplot_spec(nrows, ncols, subplot_spec, fig=None, gs_out=False, **kwargs):

    if fig is None:
        fig = plt.gcf()
    if subplot_spec is not None:
        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=subplot_spec, **kwargs)
    else:
        gs = gridspec.GridSpec(nrows, ncols, **kwargs)

    if gs_out:
        return gs

    return [fig.add_subplot(x) for x in gs]


def anchor_axes(pos, ax_or_subplot_spec, fig=None):
    kwargs = {}
    if type(ax_or_subplot_spec) == mpl.gridspec.GridSpec:
        if fig is None:
            fig = plt.gcf()
        kwargs.update(fig=fig)
    bounds = ax_or_subplot_spec.get_position(**kwargs).bounds

    return [bounds[0] + pos[0] * bounds[2],
            bounds[1] + pos[1] * bounds[3],
            pos[2] * bounds[2],
            pos[3] * bounds[3]]