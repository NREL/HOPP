(resource:ak-wind-resource)=
# Wind Resource for Alaska (API)

Wind resource data can downloaded for Alaska from the NREL Developer Network hosted Wind Integration National Dataset (WIND) Toolkit dataset [Wind Toolkit Data - Alaska V1.0.0](https://developer.nrel.gov/docs/wind/wind-toolkit/wtk-alaska-v1-0-0-download/). Using this functionality requires an NREL API key.

Wind resource data for Alaska can only be downloaded for wind resource years 2018-2020 and is only downloaded if the `wind_resource_region` input to [SiteInfo](../site_info.md) is set to "ak". For example:

```yaml
site:
    wind_resource_region: "ak"
```

```{eval-rst}
.. autoclass:: hopp.simulation.technologies.resource.alaska_wind.AlaskaWindData
    :members:
    :exclude-members: _abc_impl, check_download_dir
```