# Power Sources and Storage Technologies

These are the primary technologies that may be configured for a standard HOPP simulation.

- [Simple PV Plant](tech:simple-pv)
- [Detailed PV Plant](tech:detailed-pv)
- [Wind Plant](tech:wind)
- [CSP Plant](tech:csp)
  - [Molten Salt Tower CSP Plant](tech:csp-molten-tower)
  - [Parabolic Trough CSP Plant](tech:csp-parabolic-trough)
- [Battery](tech:battery)
  - [Stateful Battery](tech:battery-stateful)
  - [Stateless Battery](tech:battery-stateless)
- [Grid](tech:grid)
- [Wave Plant](tech:wave)

(tech:power-source)=
## Power Source Base Class

Base class for power generation technologies.

```{eval-rst}
.. autoclass:: hopp.simulation.technologies.power_source.PowerSource
    :members:
    :exclude-members: copy, plot
```
