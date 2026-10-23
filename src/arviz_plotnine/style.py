"""Style/templating helpers."""

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotnine as p9

from arviz_plotnine.themes import theme_arviz_variat

_STYLES = {"arviz-variat": theme_arviz_variat}
_LINESTYLES = ["-", "--", ":", "-."]
_MARKERS = ["o", "+", "^", "x", "d", "s", "."]
_CYCLE_PATTERN = re.compile(r"C[0-9]+")
_BACKGROUND_PATTERN = re.compile(r"B[0-3]")

_current_style = "arviz-variat"


def use(name):
    """Set an arviz-plotnine style as the global default for plotnine.

    The style is applied with :func:`plotnine.theme_set`, so it affects every
    plotnine plot created afterwards, not only the ones made with arviz-plotnine.

    Parameters
    ----------
    name : str
        Name of the style to use. See :func:`available` for the valid names.

    Raises
    ------
    ValueError
        If the style with the given name is not found.
    """
    if name not in _STYLES:
        raise ValueError(f"Style {name} not found. Available styles: {available()}")
    global _current_style  # pylint: disable=global-statement
    _current_style = name
    p9.theme_set(get_theme())


def available():
    """List the available style names.

    Returns
    -------
    list of str
    """
    return list(_STYLES)


def get_theme():
    """Return a new instance of the currently selected style's theme."""
    style = _STYLES[_current_style]
    return style() if isinstance(style, type) else style


def get_palette():
    """Return the color cycle of the currently selected style."""
    return list(get_theme().palette)


def get_background_color():
    """Return the background color of the currently selected style."""
    return get_theme().background_color


def get_contrast_colors(bg_color="#FFFFFF"):
    """Get colors that contrast with `bg_color`.

    Checks the brightness of the provided background color and returns colors
    that contrast with white if brightness is high, and colors that contrast with
    black otherwise. `bg_color` must be a hexadecimal color.
    """
    color = bg_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    yiq = (red * 299 + green * 587 + blue * 114) / 1000
    if yiq < 128:
        return ("#FFFFFF", "#959595", "#5B5B5B")
    return ("#000000", "#595959", "#949494")


def dealias_aes(aes, value):
    """Convert aesthetic aliases like ``"C0"`` or ``"B1"`` to actual values.

    ``"C<N>"`` selects the N-th element of the current style's cycle for the given
    aesthetic (color/fill, linetype or shape), cycling it if needed. ``"B<N>"``
    selects background-dependent colors: B0 is the background, B1 a high-contrast
    color, and B2/B3 progressively more muted colors.
    """
    if not isinstance(value, str):
        return value
    if _CYCLE_PATTERN.fullmatch(value):
        index = int(value[1:])
        if aes in {"color", "fill", "facecolor", "edgecolor"}:
            cycle = get_palette()
        elif aes in {"linetype", "linestyle"}:
            cycle = _LINESTYLES
        elif aes in {"shape", "marker"}:
            cycle = _MARKERS
        else:
            return value
        return cycle[index % len(cycle)]
    if _BACKGROUND_PATTERN.fullmatch(value) and aes in {
        "color",
        "fill",
        "facecolor",
        "edgecolor",
    }:
        index = int(value[1:])
        if index == 0:
            return get_background_color()
        return get_contrast_colors(get_background_color())[index - 1]
    return value


def palette_values(n_values, neutral):
    """Return `n_values` colors from the palette, cycling it if needed.

    When `neutral` is true the first palette color is reserved for variables the
    mapping cannot be applied to, so the mapped values start from the second color.
    """
    palette = get_palette()
    if neutral:
        palette = palette[1:]
    return [palette[idx % len(palette)] for idx in range(max(n_values, 1))]


def aesthetic_values(aesthetic, n_values, neutral):
    """Return `n_values` default values for an aesthetic, cycling if needed."""
    n_values = max(n_values, 1)
    if aesthetic in {"color", "fill", "facecolor", "edgecolor"}:
        return palette_values(n_values, neutral)
    if aesthetic in {"linetype", "linestyle"}:
        cycle = _LINESTYLES
    elif aesthetic in {"shape", "marker"}:
        cycle = _MARKERS
    elif aesthetic == "alpha":
        return list(np.linspace(0.2, 0.7, n_values))
    elif aesthetic == "size":
        return list(np.linspace(1, 6, n_values))
    else:
        return None
    if neutral:
        cycle = cycle[1:]
    return [cycle[idx % len(cycle)] for idx in range(n_values)]


def get_visual(visuals, defaults, shown_by_default, name):
    """Resolve a ``visuals`` entry into a ``(show, kwargs)`` pair.

    `False` hides the visual entirely. Otherwise the (possibly empty) mapping of
    kwargs is merged on top of that visual's defaults, so users only need to
    specify the properties they want to change, e.g. ``{"color": "darkorchid"}``.
    """
    value = visuals.get(name, shown_by_default.get(name, True))
    if value is False:
        return False, {}
    kwargs = dict(defaults[name])
    if value is not True and isinstance(value, Mapping):
        kwargs.update(value)
    kwargs = {key: dealias_aes(key, val) for key, val in kwargs.items()}
    return True, kwargs


class VisualBuilder:
    """Helper class to manage adding layers to a plot."""

    def __init__(self, p, visuals, defaults, shown_by_default, aes, aes_by_visuals, ds):
        self.p = p
        self.visuals = visuals
        self.defaults = defaults
        self.shown_by_default = shown_by_default
        self.aes = aes
        self.aes_by_visuals = aes_by_visuals
        self.ds = ds
        self.aes_levels = {aesthetic: {} for aesthetic in aes}

    def add_visual(self, geom_cls, df, mapping, name, **kwargs):
        """Add one or two ``geom_cls`` layers for visual `name`, mapping `aes` where it applies.

        Only the aesthetics listed in ``aes_by_visuals[name]`` are mapped for this visual.
        Rows missing one of the dimensions those aesthetics map to (e.g. a variable that does
        not have the dimension mapped to an aesthetic) are drawn in a separate, unmapped
        layer using `name`'s plain fixed style instead of being mapped with a missing/NaN
        value -- this keeps them out of the legend entirely rather than showing up as a
        spurious "None" entry.

        Extra keyword arguments are merged over the visual's `defaults` and `visuals`
        entries, so each visual can be customized at the call site.
        """
        show, visual_kwargs = get_visual(
            self.visuals, self.defaults, self.shown_by_default, name
        )
        if not show:
            return
        visual_kwargs.update(kwargs)
        visual_kwargs = {
            key: dealias_aes(key, val) for key, val in visual_kwargs.items()
        }
        visual_aes = {
            key: dim
            for key, dim in self.aes.items()
            if key in self.aes_by_visuals.get(name, ())
        }
        for aesthetic, dim in visual_aes.items():
            column = df[dim]
            if pd.api.types.is_string_dtype(column) or isinstance(
                column.dtype, pd.CategoricalDtype
            ):
                for level in column.dropna().unique():
                    self.aes_levels[aesthetic].setdefault(level, None)
        if not visual_aes:
            self.p += geom_cls(df, p9.aes(**mapping), **visual_kwargs)
        else:
            aes_dims = list(dict.fromkeys(visual_aes.values()))
            has_aes = df[aes_dims].notna().all(axis=1)
            mapped_df, unmapped_df = df[has_aes], df[~has_aes]
            if not mapped_df.empty:
                mapped_kwargs = {
                    key: val
                    for key, val in visual_kwargs.items()
                    if key not in visual_aes
                }
                self.p += geom_cls(
                    mapped_df, p9.aes(**mapping, **visual_aes), **mapped_kwargs
                )
            if not unmapped_df.empty:
                self.p += geom_cls(unmapped_df, p9.aes(**mapping), **visual_kwargs)

    def add_scales(self):
        """Add manual scales for the categorical aesthetics used by any visual."""
        for aesthetic, levels in self.aes_levels.items():
            if not levels:
                continue
            scale_cls = getattr(p9, f"scale_{aesthetic}_manual", None)
            if scale_cls is None:
                continue
            dim = self.aes[aesthetic]
            neutral = any(dim not in da.dims for da in self.ds.data_vars.values())
            values = aesthetic_values(aesthetic, len(levels), neutral)
            if values is None:
                continue
            self.p += scale_cls(values=values, limits=list(levels))