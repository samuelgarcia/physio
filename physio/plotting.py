import numpy as np


def plot_cyclic_deformation(data, segment_ratios=None, two_cycles=True, ax=None):
    """

    Parameters
    ----------
    data: np.array
        A 2d cyclic deformed array
    segment_ratios: None or list
        Multi multi segment deformation then vertical line are also ploted
    two_cycles: bool (dafult True)
        Plot 2 consecutive cycles.
    ax: None or matplotlib axes
        Optional an external ax
    Returns
    -------

    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
    
    assert data.ndim == 2
    points_per_cycle = data.shape[1]

    if two_cycles:
        data = np.concatenate([data, data], axis=1)

    av = np.mean(data, axis=0)

    ax.plot(data.T, color='k', alpha=0.1)
    ax.plot(np.mean(data, axis=0), color='orange')

    if segment_ratios is not None:
        if np.isscalar(segment_ratios):
            segment_ratios = [segment_ratios]
        for r in segment_ratios:
            ax.axvline(r * points_per_cycle, color='#27AE60')
        if two_cycles:
            ax.axvline(points_per_cycle, color='#27AE60')
            for r in segment_ratios:
                ax.axvline((r + 1) * points_per_cycle, color='#27AE60')
    
    if two_cycles:
        ax.set_xlim(0, points_per_cycle * 2 - 1)
    else:
        ax.set_xlim(0, points_per_cycle - 1)
