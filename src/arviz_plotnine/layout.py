import plotnine as p9
from arviz_plotnine.style import get_theme

def get_figure_size():
    """Return the base figure size of the currently selected style."""
    try:
        return tuple(get_theme().getp("figure_size"))
    except (AttributeError, KeyError, TypeError):
        return tuple(p9.options.figure_size)


def wrap_layout(n_plots, col_wrap):
    """Return the number of facet columns and the figure size for `n_plots` panels.

    Panels wrap at `col_wrap` columns. Width scales with the number of columns,
    with narrower columns once there are four or more. Height grows with the
    number of rows, starting from the active style's base figure size.
    """
    n_plots = max(n_plots, 1)
    if col_wrap is None or n_plots <= col_wrap:
        n_cols = n_plots
        n_rows = 1
    else:
        n_rows, extra = divmod(n_plots, col_wrap)
        n_rows += extra != 0
        n_cols = col_wrap
    default_width, default_height = get_figure_size()
    width = n_cols * (default_width if n_cols < 4 else 0.6 * default_width)
    height = default_height / 4 * (n_rows + 1) ** 1.1
    return n_cols, (width, height)