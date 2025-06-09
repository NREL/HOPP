(tech:battery)=
# Battery Storage

(tech:battery-stateful)=
## Stateful Battery Storage

Battery Storage class based on PySAM's `BatteryStateful` Model

### Battery Model

```{eval-rst} 
.. autoclass:: hopp.simulation.technologies.battery.battery.Battery
    :members:
```

### Battery Configuration

```{eval-rst} 
.. autoclass:: hopp.simulation.technologies.battery.battery.BatteryConfig
    :members:
```

### Battery Outputs

```{eval-rst} 
.. autoclass:: hopp.simulation.technologies.battery.battery.BatteryOutputs
    :members:
```

(tech:battery-stateless)=
## Stateless Battery Storage

Battery Storage class with no system model for tracking the state of the battery.

### Stateless Battery Model

```{eval-rst} 
.. autoclass:: hopp.simulation.technologies.battery.battery_stateless.BatteryStateless
    :members:
```

### Stateless Battery Configuration

```{eval-rst} 
.. autoclass:: hopp.simulation.technologies.battery.battery_stateless.BatteryStatelessConfig
    :members:
```

### Stateless Battery Outputs
```{eval-rst} 
.. autoclass:: hopp.simulation.technologies.battery.battery_stateless.BatteryStatelessOutputs
    :members:
```
