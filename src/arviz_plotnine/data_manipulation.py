"""Shared helpers for reshaping arviz data into plotnine-friendly long DataFrames."""

import pandas as pd
import xarray as xr
from arviz_base.validate import validate_sample_dims
from arviz_base.utils import _var_names
from arviz_base.labels import BaseLabeller


def dataset_to_long_dataframe(ds, value_dims, labeller=None, label_type="vert", value_name="value"):
    """Turn a Dataset into a tidy/long DataFrame ready for plotnine faceting.

    Every panel's values are kept in a single `value_name` column and a ``label``
    column identifies which panel each row belongs to, which is the shape that
    plotnine's ``aes``/``facet_wrap``/``facet_grid`` expect. Every dimension of each
    variable that is not in `value_dims` is kept as its own DataFrame column, so it
    remains available for mapping to a plotnine aesthetic.

    Each variable is converted independently, so variables can have different
    dimensions. For example, ``mu`` with no extra dimensions and ``theta`` with an
    extra ``school`` dimension can be combined. Rows for variables missing a
    dimension present in other variables get `NaN` in that column.

    Parameters
    ----------
    ds : Dataset
        Input data.
    value_dims : sequence of hashable
        Dimensions shared by every variable that should be kept as real DataFrame
        columns and never folded into ``label`` (e.g. the KDE/ECDF grid dimension).
    labeller : labeller, optional
        Defaults to :class:`~arviz_base.labels.BaseLabeller`.
    label_type : {"flat", "vert"}, default "vert"
        Whether to use :meth:`~arviz_base.labels.BaseLabeller.make_label_flat` or
        :meth:`~arviz_base.labels.BaseLabeller.make_label_vert` to build ``label``.
    value_name : str, default "value"
        Name of the resulting values column.

    Returns
    -------
    pandas.DataFrame
    """
    if labeller is None:
        labeller = BaseLabeller()
    make_label = labeller.make_label_vert if label_type == "vert" else labeller.make_label_flat

    frames = []
    for var_name, da in ds.data_vars.items():
        # drop scalar/non-dim coords (e.g. per-variable KDE bandwidths); they are not
        # necessarily shared across variables and would otherwise pollute the columns
        da = da.reset_coords(drop=True)
        if da.ndim == 0:
            # `to_dataframe` rejects scalar DataArrays (e.g. a variable with no
            # dimensions left after reducing over all sample dims)
            df = pd.DataFrame({value_name: [da.item()]})
        else:
            df = da.to_dataframe(name=value_name).reset_index()
        facet_dims = [dim for dim in da.dims if dim not in value_dims]
        if facet_dims:
            combos = df[facet_dims].drop_duplicates().reset_index(drop=True)
            pos_lookup = {
                dim: {val: pos for pos, val in enumerate(da.indexes[dim])} for dim in facet_dims
            }
            combos["label"] = [
                make_label(
                    var_name,
                    dict(zip(facet_dims, combo)),
                    {dim: pos_lookup[dim][val] for dim, val in zip(facet_dims, combo)},
                )
                for combo in combos[facet_dims].itertuples(index=False, name=None)
            ]
            df = df.merge(combos, on=facet_dims, how="left")
        else:
            df["label"] = make_label(var_name, {}, {})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _get_group(data, group):
    """Return `group` from a DataTree/Dataset as a Dataset."""
    if isinstance(data, xr.Dataset):
        return data
    group_data = data[group]
    return group_data if isinstance(group_data, xr.Dataset) else group_data.dataset


def prepare_dataset(dt, group, var_names=None, filter_vars=None, coords=None, sample_dims=None):
    if isinstance(dt, dict):
        distribution = {}
        selected_vars = []
        all_vars = []
        for model_name, model_dt in dt.items():
            model_ds = _get_group(model_dt, group)
            model_var_names = _var_names(
                var_names, model_ds, filter_vars, check_if_present=False
            )
            if model_var_names is not None:
                available_vars = [var for var in model_ds.data_vars if var in model_var_names]
                distribution[model_name] = model_ds[available_vars]
                selected_vars.extend(model_var_names)
                all_vars.extend(model_ds.data_vars)
            else:
                distribution[model_name] = model_ds
        if var_names is not None:
            missing_vars = set(selected_vars).difference(all_vars)
            if missing_vars:
                plural = "" if len(missing_vars) == 1 else "s"
                raise KeyError(f"variable{plural} {missing_vars} not found in any dataset")
        ds = xr.concat(
            [model_ds.sel(coords or {}) for model_ds in distribution.values()],
            dim="model",
            join="outer",
        ).assign_coords(model=list(distribution))
    else:
        ds = _get_group(dt, group)
        var_names = _var_names(var_names, ds, filter_vars)
        if var_names is not None:
            ds = ds[var_names]
        if coords:
            ds = ds.sel(coords)
    sample_dims = list(validate_sample_dims(sample_dims, data=ds))
    return ds, sample_dims