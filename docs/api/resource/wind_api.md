(resource:wind-resource)=
# Wind Resource for Continental U.S. (API)

By default, wind resource data is downloaded from the NREL Developer Network hosted Wind Integration National Dataset (WIND) Toolkit dataset [Wind Toolkit Data - SAM format (srw)](https://developer.nrel.gov/docs/wind/wind-toolkit/wtk-srw-download/). 

Wind resource data for the continental U.S. can be downloaded from Wind Toolkit for wind resource years 2007 - 2014. Using this functionality requires an NREL API key.

Wind data for the continental U.S. for wind resource years 2015 - 2023 can be downloaded from the BC-HRRR dataset, see [BC-HRRR Dataset Wind Resource (API)](resource:bc-hrrr-wind-resource).

```{eval-rst}
.. autoclass:: hopp.simulation.technologies.resource.wind_resource.WindResource
    :members:
    :exclude-members: _abc_impl, check_download_dir
```