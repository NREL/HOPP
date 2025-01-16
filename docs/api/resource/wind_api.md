(resource:wind-resource)=
# Wind Resource (API)

By default, wind resource data is downloaded from the NREL Developer Network hosted Wind Integration National Dataset (WIND) Toolkit dataset [Wind Toolkit Data - SAM format (srw)](https://developer.nrel.gov/docs/wind/wind-toolkit/wtk-srw-download/). Using this functionality requires an NREL API key.

```{eval-rst}
.. autoclass:: hopp.simulation.technologies.resource.wind_resource.WindResource
    :members:
    :exclude-members: _abc_impl, check_download_dir
```