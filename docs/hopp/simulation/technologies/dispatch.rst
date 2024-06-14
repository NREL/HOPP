.. _Dispatch:

Dispatch Strategies
===================

These are the dispatch strategies that may be used for a standard HOPP simulation. Dispatch 
settings can be defined through :class:`.HybridDispatchOptions`. 

Storage Dispatch
----------------

.. toctree::
    :maxdepth: 1

    dispatch/power_storage/simple_battery_dispatch.rst
    dispatch/power_storage/simple_battery_dispatch_heuristic.rst
    dispatch/power_storage/heuristic_load_following_dispatch.rst
    dispatch/power_storage/linear_voltage_convex_battery_dispatch.rst
    dispatch/power_storage/linear_voltage_nonconvex_battery_dispatch.rst
    dispatch/power_storage/one_cycle_battery_dispatch_heuristic.rst

The above dispatch classes inherit from the :py:class:`.PowerStorageDispatch` class.

.. toctree::
    :maxdepth: 1

    dispatch/power_storage/power_storage_dispatch.rst

Technology Dispatch
-------------------

Dispatch classes are made for each technology where their specific components of the objectives,
their parameters, and other technology specific dispatch properties are defined.

.. toctree::
    :maxdepth: 1

    dispatch/power_sources/pv_dispatch.rst
    dispatch/power_sources/wind_dispatch.rst
    dispatch/power_sources/wave_dispatch.rst
    dispatch/power_sources/trough_dispatch.rst
    dispatch/power_sources/tower_dispatch.rst
    dispatch/power_sources/csp_dispatch.rst

The above technology classes inherit from the :py:class:`.PowerSourceDispatch` class.

.. toctree::
    :maxdepth: 1

    dispatch/power_sources/power_source_dispatch.rst
