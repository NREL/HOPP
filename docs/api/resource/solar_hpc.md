(resource:nsrdb-data)=
# Solar Resource (NSRDB Dataset on NREL HPC)

If enabled, solar resource data can be loaded from the NREL HPC (Kestrel) hosted National Solar Radiation Database (NSRDB) dataset. This functionality leverages the [NREL REsource eXtraction (rex) tool](https://github.com/NREL/rex). Information on NREL HPC file systems and datasets can be found [here](https://nrel.github.io/HPC/Documentation/Systems/Kestrel/Filesystems/#projectfs).

```{eval-rst}
.. autoclass:: hopp.simulation.technologies.resource.nsrdb_data.HPCSolarData
    :members:
    :undoc-members:
    :exclude-members: _abc_impl, check_download_dir, call_api
```