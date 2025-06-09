# Resource Data

These are the primary methods for accessing wind and solar resource data.

- [Solar Resource (API)](resource:solar-resource)
- [Conus Wind Resource (API)](resource:wind-resource)
- [Alaska Wind Resource (API)](resource:ak-wind-resource)
- [BC-HRRR Dataset Wind Resource (API)](resource:bc-hrrr-wind-resource)
- [Solar Resource (NSRDB Dataset on NREL HPC)](resource:nsrdb-data)
- [Wind Resource (Wind Toolkit Dataset on NREL HPC)](resource:wtk-data)
- [Wave Resource (Data)](resource:wave-resource)
- [Tidal Resource (Data)](resource:tidal-resource)

## NREL API Keys

An NREL API key is required to use the functionality for [Solar Resource (API)](resource:solar-resource) and [Wind Resource (API)](resource:wind-resource).

An NREL API key can be obtained from [here](https://developer.nrel.gov/signup/).

Once an API key is obtained, create a file ".env" in the HOPP root directory (/path/to/HOPP/.env) that contains the lines:

```bash
NREL_API_KEY=key
NREL_API_EMAIL=your.name@email.com
```

where `key` is your API key and `your.name@email.com` is the email that was used to get the API key.

## NREL HPC Datasets

To load resource data from datasets hosted on NREL's HPC, HOPP must be installed and run from the NREL HPC. Currently, loading resource data from HPC is only enabled for [wind](resource:wtk-data) and [solar](resource:nsrdb-data) resource.


(resource:resource-base)=
## Resource Base Class

Base class for resource data

```{eval-rst}
.. autoclass:: hopp.simulation.technologies.resource.Resource
    :members:
    :exclude-members: copy, plot, _abc_impl
```
