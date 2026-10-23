"""The ``arviz-variat`` theme.

A white background, no grid, only bottom and left axis lines, and a
color-blind-friendly palette.
"""
from plotnine import (
    element_blank,
    element_line,
    element_rect,
    element_text,
    theme,
    theme_classic,
)

class theme_arviz_variat(theme_classic):
    """The ``arviz-variat`` theme.

    Parameters
    ----------
    base_size : int, default 12
        Base font size. All text sizes are scaled from it.
    base_family : str, optional
        Base font family. If `None`, use :data:`plotnine.options.base_family`.
    """

    palette = [
        "#36acc6",
        "#f66d7f",
        "#fac364",
        "#7c2695",
        "#228306",
        "#a252f4",
        "#63f0ea",
        "#a77e4f",
    ]
    background_color = "#FFFFFF"

    def __init__(self, base_size=12, base_family=None):
        super().__init__(base_size, base_family)
        large = base_size * 1.2
        self._rcParams["image.cmap"] = "viridis"
        self += theme(
            line=element_line(lineend="round"),
            text=element_text(color=".15"),
            plot_background=element_rect(fill="white", color="white"),
            panel_background=element_rect(fill="white"),
            axis_line=element_line(color=".33", size=0.8),
            axis_ticks=element_line(color=".15", size=0.8),
            axis_text=element_text(size=large, color=".15"),
            axis_title=element_text(size=large, color=".15", weight="normal"),
            panel_grid=element_blank(),
            strip_background=element_blank(),
            strip_text=element_text(size=16, weight="bold", color=".15"),
            legend_background=element_blank(),
            legend_key=element_blank(),
            legend_frame=element_blank(),
            legend_text=element_text(size=large),
            legend_title=element_text(size=large),
            panel_spacing=0.03,
            plot_title=element_text(size=base_size * 1.44, weight="bold", color=".15"),
            figure_size=(6, 5),
            dpi=100,
        )
