"""Marginal distribution plots (kde, ecdf, hist and dot)."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotnine as p9
import xarray as xr
import arviz_stats as azs
from arviz_base import rcParams, references_to_dataset
from arviz_base.labels import BaseLabeller
from arviz_base.validate import (
    validate_ci_prob,
    validate_dict_argument,
    validate_or_use_rcparam,
)
from arviz_stats.base.stats_utils import round_num

from arviz_plotnine.layout import wrap_layout
from arviz_plotnine.data_manipulation import dataset_to_long_dataframe, prepare_dataset
from arviz_plotnine.style import VisualBuilder, get_visual, get_theme

defaults = {
    "dist": {"color": "C0"},
    "face": {"fill": "C0", "alpha": 0.4},
    "credible_interval": {"color": "B2", "size": 1},
    "point_estimate": {"color": "B2", "size": 3},
    "point_estimate_text": {"color": "B2", "va": "bottom"},
    "rug": {"color": "B1", "shape": "|", "size": 1.5},
    "rope": {"fill": "B1", "alpha": 0.2, "ymin": -np.inf, "ymax": np.inf},
    "rope_text": {"color": "B1", "ha": "center", "va": "center"},
    "title": {},
}

shown_by_default = {"face": False, "rug": False}


def _unify_grid_dim(ds, kind):
    """Rename the density grid dimension to a shared name across all variables."""
    if kind == "kde":
        rename = {"kde_dim": "grid_dim"}
    elif kind == "dot":
        rename = {"qd_dim": "grid_dim"}
    elif kind == "hist":
        rename = {f"hist_dim_{var_name}": "grid_dim" for var_name in ds.data_vars}
    else:
        rename = {f"ecdf_dim_{var_name}": "grid_dim" for var_name in ds.data_vars}
    return ds.rename({old: new for old, new in rename.items() if old in ds.dims})


def _hist_step_dataframe(df, aes_dims):
    """Build x/y coordinates for a step histogram from left/right bin edges.

    One extra row is appended per group so the last bin closes at its right edge,
    producing a continuous staircase.
    """
    frames = []
    for _, group in df.groupby(["label", *aes_dims], dropna=False, sort=False):
        group = group.sort_values("left_edges")
        closing_row = group.iloc[[-1]].copy()
        closing_row["left_edges"] = closing_row["right_edges"]
        frames.append(pd.concat([group, closing_row], ignore_index=True))
    step_df = pd.concat(frames, ignore_index=True)
    step_df["x"] = step_df["left_edges"]
    step_df["y"] = step_df["histogram"]
    return step_df


def plot_dist(
    dt,
    var_names=None,
    filter_vars=None,
    group="posterior",
    coords=None,
    sample_dims=None,
    kind=None,
    point_estimate=None,
    rope=None,
    ci_kind=None,
    ci_prob=None,
    round_to=None,
    col_wrap=4,
    figure_size=None,
    labeller=None,
    aes: Mapping[
        Literal["color", "fill", "linetype", "alpha", "shape", "size"],
        str,
    ] = None,
    aes_by_visuals: Mapping[
        Literal[
            "dist",
            "face",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "rug",
            "rope",
            "rope_text",
            "title",
        ],
        Sequence[str],
    ] = None,
    visuals: Mapping[
        Literal[
            "dist",
            "face",
            "credible_interval",
            "point_estimate",
            "point_estimate_text",
            "rug",
            "rope",
            "rope_text",
            "remove_axis",
            "title",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    stats: Mapping[
        Literal["dist", "credible_interval", "point_estimate", "rope"],
        Mapping[str, Any] | xr.Dataset,
    ] = None,
):
    """Plot 1D marginal densities in the style of John K. Kruschke's book [1]_.

    Generate faceted plots with: a graphical representation of 1D marginal
    densities (as KDE, histogram, ECDF or dotplot), a credible interval and a point
    estimate.

    Parameters
    ----------
    dt : DataTree, Dataset or dict of {str : DataTree}
        Input data. In case of dictionary input, the keys are taken to be model names.
        In such cases, a dimension "model" is generated and can be used to map to
        aesthetics.
    var_names : str or list of str, optional
        One or more variables to be plotted.
        Prefix the variables by ~ when you want to exclude them from the plot.
    filter_vars : {None, "like", "regex"}, default=None
        If None, interpret var_names as the real variables names.
        If "like", interpret var_names as substrings of the real variables names.
        If "regex", interpret var_names as regular expressions on the real variables names.
    group : str, default "posterior"
        Group to be plotted.
    coords : dict, optional
    sample_dims : str or sequence of hashable, optional
        Dimensions to reduce.
        Defaults to ``rcParams["data.sample_dims"]``.
    kind : {"auto", "kde", "hist", "dot", "ecdf"}, optional
        How to represent the marginal density.
        Defaults to ``rcParams["plot.density_kind"]``. "auto" is treated as "kde".
    point_estimate : {"mean", "median", "mode"}, optional
        Which point estimate to plot. Defaults to rcParam :data:`stats.point_estimate`.
    rope : tuple of float, or dict of {str : tuple of float}, optional
        If provided, a region of practical equivalence (ROPE) is plotted as a shaded area.
        The annotation shows the % of the credible interval that falls within the ROPE.
        For the % of the full posterior that falls within the ROPE pass
        ``stats={"rope": {"ci_prob": 1}},``
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``.
    ci_prob : float, optional
        Indicates the probability that should be contained within the plotted credible
        interval. Defaults to ``rcParams["stats.ci_prob"]``.
    round_to : int or str, optional
        Number of decimals used to round the point estimate text.
        Defaults to ``rcParams["stats.round_to"]``.
    col_wrap : int or None, default 4
        Maximum number of facet columns before wrapping to a new row. `None` puts
        every panel on a single row.
    figure_size : (float, float), optional
        Figure size in inches. Defaults to the size computed from the number of rows
        and columns and the active style's base figure size.
    labeller : labeller, optional
        Defaults to :class:`~arviz_base.labels.BaseLabeller`.
    aes : mapping of {str : str}, optional
        Map dimensions of `dt` to plotnine aesthetics. Keys are aesthetic names
        and values are dimension names, one dimension per aesthetic. Categorical
        dimensions mapped to color or fill use the active style palette;
        linetype, shape, size and alpha use cycling style defaults. Numeric
        dimensions use the corresponding plotnine continuous scale.
    aes_by_visuals : mapping of {str : sequence of str}, optional
        Mapping of visuals to aesthetics that should use their mapping when plotted.
        Valid keys are the same as for `visuals`, except `remove_axis`.

        With a single model, no aesthetic mappings are generated by default, so each
        variable+coord combination gets a plot but they all look the same, unless
        there are user provided aesthetic mappings.
        With multiple models, "color" is mapped to the "model" dimension.

        By default, all aesthetics in `aes` are mapped to the density
        representation. The face uses the same mappings except "linetype".
        "credible_interval", "point_estimate" and "point_estimate_text" keep
        their fixed styles, unless multiple models are present, in which case
        "color" is mapped to the "model" dimension.

        The "title" mapping is accepted for API compatibility but not applied;
        plotnine uses a single style for all facet strips.

        When "point_estimate" key is provided but "point_estimate_text" isn't,
        the values assigned to the first are also used for the second.
    visuals : mapping of {str : mapping or bool}, optional
        Valid keys are:

        * dist -> depending on the value of `kind` passed to:

          * "kde" -> passed to :func:`~plotnine.geom_line`
          * "ecdf" -> passed to :func:`~plotnine.geom_step`
          * "hist" -> passed to :func:`~plotnine.geom_step`
          * "dot" -> passed to :func:`~plotnine.geom_point`

        * face -> visual that fills the area under the marginal distribution
          representation. Defaults to False. Depending on the value of `kind` it is
          passed to:

          * "kde", "ecdf" or "dot" -> passed to :func:`~plotnine.geom_ribbon`
          * "hist" -> passed to :func:`~plotnine.geom_rect`

        * credible_interval -> passed to :func:`~plotnine.geom_segment`
        * point_estimate -> passed to :func:`~plotnine.geom_point`
        * point_estimate_text -> passed to :func:`~plotnine.geom_text`
        * title -> passed to the facet strip text. Defaults to True.
        * rug -> passed to :func:`~plotnine.geom_point`. Defaults to False.
        * rope -> passed to :func:`~plotnine.geom_rect`. Defaults to True.
        * rope_text -> passed to :func:`~plotnine.geom_text`. Defaults to True.
          Both rope and rope_text are ignored if the main rope argument is None.
        * remove_axis -> set to ``False`` to keep the y axis.
    stats : mapping of {str : mapping or Dataset}, optional
        Valid keys are:

        * dist -> passed to :func:`~arviz_stats.kde`, :func:`~arviz_stats.histogram`,
          :func:`~arviz_stats.ecdf`, or :func:`~arviz_stats.qds` depending on `kind`
        * credible_interval -> passed to :func:`~arviz_stats.eti` or
          :func:`~arviz_stats.hdi`
        * point_estimate -> passed to mean, median or mode. Defaults to
          round the result according to ``rcParams["stats.round_to"]``.
        * rope -> passed to :func:`~arviz_stats.ci_in_rope`

        In case a :class:`~xarray.Dataset` is provided, it will be interpreted
        as pre-computed values for that statistic.

    Returns
    -------
    plotnine.ggplot

    See Also
    --------
    arviz_plotnine.style :
        Style and template helpers for arviz-plotnine.

    Examples
    --------
    Map the color to a dimension of the data:

    .. code-block:: python

        >>> from arviz_plotnine import plot_dist, style
        >>> style.use("arviz-variat")
        >>> from arviz_base import load_arviz_data
        >>> non_centered = load_arviz_data('non_centered_eight')
        >>> plot = plot_dist(
        ...     non_centered,
        ...     coords={"school": ["Choate", "Deerfield", "Hotchkiss"]},
        ...     aes={"color": "school"},
        ... )

    Because ``plot_dist`` returns a plotnine plot, it can be customized further
    with plotnine layers. For example, add a reference line at zero:

    .. code-block:: python

        >>> import plotnine as p9
        >>> plot = plot_dist(non_centered, var_names=["mu", "tau"])
        >>> plot + p9.geom_vline(xintercept=0, linetype="dashed")

    Add a region of practical equivalence (ROPE) and show the percentage of
    the credible interval that falls within it:

    .. code-block:: python

        >>> plot = plot_dist(
        ...     non_centered,
        ...     var_names=["mu", "tau"],
        ...     rope={"mu": (-0.5, 0.9), "tau": (0.2, 9.4)},
        ... )

    References
    ----------
    .. [1] Kruschke. Doing Bayesian Data Analysis, Second Edition: A Tutorial with R,
        JAGS, and Stan. Academic Press, 2014. ISBN 978-0-12-405888-0.
        https://www.sciencedirect.com/book/9780124058880
    """
    kind = validate_or_use_rcparam(kind, "plot.density_kind")
    if kind == "auto":
        kind = "kde"
    if kind not in ("kde", "ecdf", "hist", "dot"):
        raise ValueError(
            f"kind must be one of 'auto', 'kde', 'hist', 'dot' or 'ecdf', got '{kind}'"
        )
    point_estimate = validate_or_use_rcparam(point_estimate, "stats.point_estimate")
    ci_kind = validate_or_use_rcparam(ci_kind, "stats.ci_kind")
    ci_prob = validate_ci_prob(ci_prob)
    if round_to is None:
        round_to = rcParams["stats.round_to"]
    if labeller is None:
        labeller = BaseLabeller()
    visuals = validate_dict_argument(visuals, (plot_dist, "visuals"))
    aes = validate_dict_argument(aes, (plot_dist, "aes"))
    aes_by_visuals = validate_dict_argument(aes_by_visuals, (plot_dist, "aes_by_visuals"))
    stats = validate_dict_argument(stats, (plot_dist, "stats"))
    if kind == "ecdf":
        visuals.setdefault("remove_axis", False)

    ds, sample_dims = prepare_dataset(dt, group, var_names, filter_vars, coords, sample_dims)

    rope_dataset = None
    rope_pct = None
    if rope:
        rope_dataset = references_to_dataset(
            rope, ds, sample_dims=sample_dims, ref_dim="band_dim"
        )
        rope_stats = stats.get("rope", {}).copy()
        rope_stats.setdefault("ci_prob", ci_prob)
        rope_stats.setdefault("ci_kind", ci_kind)
        rope_pct = azs.ci_in_rope(ds, rope=rope, dim=sample_dims, **rope_stats)

    if "model" in ds.dims:
        aes.setdefault("color", "model")

    aes_dims = list(dict.fromkeys(aes.values()))
    for dim in aes_dims:
        if dim not in ds.dims:
            raise ValueError(f"'{dim}' in `aes` is not a dimension of `dt`")
        if dim in sample_dims:
            raise ValueError(
                f"'{dim}' in `aes` is also in `sample_dims`; it is reduced away and "
                "cannot be mapped to an aesthetic"
            )

    aes_by_visuals = {key: list(value) for key, value in aes_by_visuals.items()}
    aes_by_visuals.setdefault("dist", list(aes))
    if get_visual(visuals, defaults, shown_by_default, "face")[0]:
        aes_by_visuals.setdefault(
            "face", [key for key in aes_by_visuals["dist"] if key != "linetype"]
        )
    if "model" in ds.dims:
        for visual in ("credible_interval", "point_estimate"):
            aes_by_visuals.setdefault(visual, ["color"])
    if "point_estimate" in aes_by_visuals and "point_estimate_text" not in aes_by_visuals:
        aes_by_visuals["point_estimate_text"] = list(aes_by_visuals["point_estimate"])

    rug_df = None
    if get_visual(visuals, defaults, shown_by_default, "rug")[0]:
        rug_df = dataset_to_long_dataframe(
            ds, value_dims=sample_dims, labeller=labeller, value_name="x"
        )
        rug_df["y"] = 0.0

    dist_stats = stats.get("dist", {})
    if isinstance(dist_stats, xr.Dataset):
        dens = dist_stats
    else:
        dist_func = {
            "kde": azs.kde,
            "ecdf": azs.ecdf,
            "hist": azs.histogram,
            "dot": azs.qds,
        }[kind]
        dens = dist_func(ds, dim=sample_dims, **dist_stats)

    ci_stats = stats.get("credible_interval", {})
    if isinstance(ci_stats, xr.Dataset):
        ci = ci_stats
    else:
        ci_stats = {**ci_stats, "prob": ci_prob}
        ci = (azs.eti if ci_kind == "eti" else azs.hdi)(ds, dim=sample_dims, **ci_stats)

    pe_stats = stats.get("point_estimate", {})
    if isinstance(pe_stats, xr.Dataset):
        pe = pe_stats
    else:
        pe_stats = {**pe_stats, "round_to": "none"}
        pe = getattr(azs, point_estimate)(ds, dim=sample_dims, **pe_stats)

    dens = _unify_grid_dim(dens, kind)

    dens_df = (
        dataset_to_long_dataframe(dens, value_dims=["plot_axis", "grid_dim"], labeller=labeller)
        .pivot(index=["label", "grid_dim", *aes_dims], columns="plot_axis", values="value")
        .reset_index()
    )

    step_df = _hist_step_dataframe(dens_df, aes_dims) if kind == "hist" else None
    face_df = None
    if kind == "dot" and get_visual(visuals, defaults, shown_by_default, "face")[0]:
        if isinstance(dist_stats, xr.Dataset):
            face_df = dens_df
        else:
            face_density = _unify_grid_dim(
                azs.qds(ds, dim=sample_dims, top_only=True, **dist_stats), kind, "grid_dim"
            )
            face_df = (
                dataset_to_long_dataframe(
                    face_density, value_dims=["plot_axis", "grid_dim"], labeller=labeller
                )
                .pivot(index=["label", "grid_dim", *aes_dims], columns="plot_axis", values="value")
                .reset_index()
            )

    ci_df = dataset_to_long_dataframe(ci, value_dims=["ci_bound"], labeller=labeller)
    ci_df = ci_df.pivot(
        index=["label", *aes_dims], columns="ci_bound", values="value"
    ).reset_index()
    ci_df["y"] = 0.0

    pe_df = dataset_to_long_dataframe(pe, value_dims=[], labeller=labeller, value_name="x")[
        ["label", *aes_dims, "x"]
    ]
    pe_df["y"] = 0.0
    density_y = "histogram" if kind == "hist" else "y"
    max_y = dens_df.groupby("label", observed=True)[density_y].max()
    pe_df["ytext"] = 0.1 * pe_df["label"].map(max_y)
    pe_df["text"] = [f"{round_num(val, round_to)} {point_estimate}" for val in pe_df["x"]]

    rope_df = None
    rope_text_df = None
    if rope_pct is not None:
        rope_long = dataset_to_long_dataframe(
            rope_dataset, value_dims=["band_dim"], labeller=labeller, value_name="rope"
        )
        rope_index = ["label", *[dim for dim in aes_dims if dim in rope_long.columns]]
        rope_df = (
            rope_long.pivot(index=rope_index, columns="band_dim", values="rope")
            .reset_index()
            .rename(columns={0: "xmin", 1: "xmax"})
        )
        rope_text_df = dataset_to_long_dataframe(
            rope_pct, value_dims=[], labeller=labeller, value_name="rope_pct"
        )
        rope_text_df = rope_text_df.merge(
            rope_df.assign(x=(rope_df["xmin"] + rope_df["xmax"]) / 2)[["label", "x"]],
            on="label",
            how="left",
        )
        rope_text_df["y"] = 0.35 * rope_text_df["label"].map(max_y)
        rope_text_df["text"] = [
            f"{pct:.1f}% in ROPE" for pct in rope_text_df["rope_pct"]
        ]

    order = list(dict.fromkeys(dens_df["label"]))
    n_cols, default_figure_size = wrap_layout(len(order), col_wrap)
    if figure_size is None:
        figure_size = default_figure_size
    label_dfs = [dens_df, ci_df, pe_df]
    if step_df is not None:
        label_dfs.append(step_df)
    if face_df is not None and face_df is not dens_df:
        label_dfs.append(face_df)
    if rug_df is not None:
        label_dfs.append(rug_df)
    if rope_df is not None:
        label_dfs.append(rope_df)
    if rope_text_df is not None:
        label_dfs.append(rope_text_df)
    for df in label_dfs:
        df["label"] = pd.Categorical(df["label"], categories=order)

    plotter = VisualBuilder(
        p9.ggplot(), visuals, defaults, shown_by_default, aes, aes_by_visuals, ds
    )
    if kind == "kde":
        plotter.add_visual(
            p9.geom_ribbon,
            dens_df,
            {"x": "x", "ymin": 0, "ymax": "y"},
            "face",
        )
        plotter.add_visual(
            p9.geom_line,
            dens_df,
            {"x": "x", "y": "y"},
            "dist",

        )
    elif kind == "hist":
        plotter.add_visual(
            p9.geom_rect,
            dens_df,
            {"xmin": "left_edges", "xmax": "right_edges", "ymin": 0, "ymax": "histogram"},
            "face",
        )
        plotter.add_visual(
            p9.geom_step,
            step_df,
            {"x": "x", "y": "y"},
            "dist",
        )
    elif kind == "dot":
        if face_df is not None:
            plotter.add_visual(
                p9.geom_ribbon,
                face_df,
                {"x": "x", "ymin": 0, "ymax": "y"},
                "face",
            )
        plotter.add_visual(
            p9.geom_point,
            dens_df,
            {"x": "x", "y": "y"},
            "dist",
        )
    else:
        plotter.add_visual(
            p9.geom_ribbon,
            dens_df,
            {"x": "x", "ymin": 0, "ymax": "y"},
            "face",
        )
        plotter.add_visual(
            p9.geom_step,
            dens_df,
            {"x": "x", "y": "y"},
            "dist",
        )
    if rug_df is not None:
        plotter.add_visual(
            p9.geom_point,
            rug_df,
            {"x": "x", "y": "y"},
            "rug",
        )
    plotter.add_visual(
        p9.geom_segment,
        ci_df,
        {"x": "lower", "xend": "upper", "y": "y", "yend": "y"},
        "credible_interval",
    )
    plotter.add_visual(
        p9.geom_point,
        pe_df,
        {"x": "x", "y": "y"},
        "point_estimate",
    )
    plotter.add_visual(
        p9.geom_text,
        pe_df,
        {"x": "x", "y": "ytext", "label": "text"},
        "point_estimate_text",
    )
    if rope_df is not None:
        plotter.add_visual(
            p9.geom_rect,
            rope_df,
            {"xmin": "xmin", "xmax": "xmax"},
            "rope",
        )
    if rope_text_df is not None:
        plotter.add_visual(
            p9.geom_text,
            rope_text_df,
            {"x": "x", "y": "y", "label": "text"},
            "rope_text",
        )
    plotter.p += p9.facet_wrap("~label", scales="free", ncol=n_cols)

    plotter.add_scales()

    theme_kwargs = {"figure_size": figure_size}
    if kind in ["kde", "hist", "dot"] and visuals.get("remove_axis", True) is not False:
        theme_kwargs.update(
            {
                "axis_ticks_y": p9.element_blank(),
                "axis_text_y": p9.element_blank(),
                "axis_ticks_major_y": p9.element_blank(),
                "axis_title": p9.element_blank(),
            }
        )
    show_title, title_kwargs = get_visual(visuals, defaults, shown_by_default, "title")
    if not show_title:
        theme_kwargs["strip_text"] = p9.element_blank()
    elif title_kwargs:
        theme_kwargs["strip_text"] = p9.element_text(**title_kwargs)
    plotter.p += get_theme() + p9.theme(**theme_kwargs)
    return plotter.p
