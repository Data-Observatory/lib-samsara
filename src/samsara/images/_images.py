"""
Functions for processing of data and writing images within SAMSARA... perhaps 'images' isn't the best name, but taken from samsara-training-files/images.py.
"""

import numpy as np
from datacube.utils import masking
from datacube.utils.cog import write_cog
from odc.algo import mask_cleanup, to_f32

__all__ = ["write_to_cogs", "mask_and_calculate_ndvi", "xr_transform"]


def write_to_cogs(img, dim=None, **kwargs):
    if kwargs.get("overwrite") is None:
        kwargs["overwrite"] = True
    if kwargs.get("nodata") is None:
        kwargs["nodata"] = np.nan
    if dim is None:
        write_cog(img, **kwargs)
    else:
        fname = kwargs["fname"]
        for i in img[dim].values:
            kwargs["fname"] = f"{fname}_dim-{dim}-{i}.tif"
            write_cog(img.sel({dim: i}), **kwargs)


def mask_and_calculate_ndvi(ds):
    reflectance_bands = ["red", "nir08"]
    quality_band = "qa_pixel"
    good_pixel_flags = {
        "snow": "not_high_confidence",
        "cloud": "not_high_confidence",
        # "cirrus": "not_high_confidence",
        "cloud_shadow": "not_high_confidence",
        "nodata": False,
    }

    cloud_free_mask = masking.make_mask(ds[quality_band], **good_pixel_flags)
    # Apply morphological processing on cloud mask
    # See: https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/Frequently_used_code/Cloud_and_pixel_quality_masking.html#Applying-morphological-processing-on-the-cloud-mask and https://docs.dea.ga.gov.au/notebooks/How_to_guides/Using_load_ard/
    filters = [("opening", 2), ("dilation", 2)]
    cloud_free_mask = mask_cleanup(cloud_free_mask, mask_filters=filters)

    # Should morphological processing be applied to this mask as well?
    cloud_medium_conf_mask = masking.make_mask(
        ds[quality_band], cloud_confidence="medium"
    )

    cloud_masks = cloud_free_mask + cloud_medium_conf_mask

    masked = ds[reflectance_bands].where(cloud_masks)
    masked = to_f32(masked, scale=0.0000275, offset=-0.2)
    ndvi = (masked.nir08 - masked.red) / (masked.nir08 + masked.red)
    ndvi = ndvi.where((ndvi > -1) & (ndvi < 1))
    ndvi.attrs = ds.attrs

    return ndvi


def xr_transform(xarray, levels, dtype=None):
    # Esta función reduce la resolución radiométrica de la imagen
    xr = xarray.copy()
    if dtype is None:
        dtype = xarray.dtype
    min = np.nanmin(xr.data)
    max = np.nanmax(xr.data)
    zi = (xr - min) / (max - min)
    li = zi * (levels - 1)
    if xr.isnull().any().data == True:
        li = np.nan_to_num(li + 1)
    xr.data = li.round().astype(dtype)
    return xr
