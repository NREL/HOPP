(resource:bc-hrrr-wind-resource)=
# Wind Resource from the BC-HRRR Dataset (API)

Wind resource data can be downloaded from the NREL-processed bias-corrected High-Resolution Rapid Refresh (HRRR) dataset from the National Oceanic and Atmospheric Administration (NOAA) for the years 2015-2023. This data is available as an hourly operational forecast product. The data is bias-corrected so that it can be used in continuity with the legacy WIND toolkit data (2007-2013). The data is available over CONUS from the NREL Developer Network hosted Wind Integration National Dataset (WIND) Toolkit dataset [Wind Toolkit Data - BC-HRRR CONUS 60-minute (NOAA + NREL)](https://developer.nrel.gov/docs/wind/wind-toolkit/wtk-bchrrr-v1-0-0-download/). Using this functionality requires an NREL API key.

Wind resource data from the BC-HRRR dataset can only be downloaded for wind resource years 2015-2023 and is only downloaded if the `wind_resource_origin` input to [SiteInfo](../site_info.md) is set to "BC-HRRR". For example:

```yaml
site:
    wind_resource_origin: "BC-HRRR"
```

```{eval-rst}
.. autoclass:: hopp.simulation.technologies.resource.alaska_wind.BCHRRRWindData
    :members:
    :exclude-members: _abc_impl, check_download_dir
```