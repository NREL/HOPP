
# On-turbine hydrogen storage modeling

## Implementation

In this module, we create a model for storing hydrogen in a wind turbine tower
We follow, largely, the work of Kottenstette (see NREL/TP-500-34656), although
various assumptions in their study are not marked, and our goal is to flesh out
some of their assumptions.

`PressurizedTower` is an object model that represents a pressurized wind turbine
tower with geometry specified on input. The Kottenstette work assumes a wind
turbine tower whose thickness is set by a constant diameter-thickness ratio,
which defaults to 320. The tower is specified by diameter/height pairs, between
which a linear taper is assumed. The material of the tower is assumed to be
steel with the following properties:

- ultimate tensile strength: 636 MPa
- yield strength: 350 MPa
- welded joint efficiency: 0.80 (see ASME Boiler and Pressure Vessel Code for details)
- density: 7817 kg/m<sup>3</sup>
- cost per kg: $1.50

These can be modified for alternative studies by variable access on the `PressurizedTower` object before running an analysis. Refer to the `__init__()`
function for definitions. Inner volume of the tower is computed by conic frustum
volume according to each section, assuming thin walls (s.t. $d \gg t$). Wall
material is computed by assuming the wall thickness is centered at the diameter
dimension (outer or inner thickness placement is available by specification).

## Hydrogen storage

### Wall increment

When hydrogen is stored, a Goodman's equation thickness increment is assumed for
the vertical tower walls (see Kottenstette) in order to handle the additional
pressure stress contribution which is a zero-intercept linear function of
diameter, see `PressurizedTower.get_thickness_increment_const` for the leading
coefficient calculation. Hydrogen is assumed to be stored at the crossover
pressure where pressurized burst strength and aerodynamic moment fatigue are
balanced, see Kottenstette for theory and
`PressureizedTower.get_crossover_pressure` for implementation.

### End cap sizing

End caps are necessary for a pressure vessel tower, which are sized according to
the ASME Boiler and Pressure Vessel code. Caps are assumed to be welded with
flat pressure heads affixed by a rounded corner. Implementation in
`PressurizedTower.compute_cap_thickness` contains details on thickness
computation. Following Kottenstette, we use 2.66 \$/kg to cost the endcap
material.

### Hydrogen

A pressurized tower, then, is assumed to hold a volume of $\mathrm{H}_2$, stored
at pressure and the ambient temperature, and the ideal gas law is used to relate
the resulting mass of the stored hydrogen.

### Summary and additional costs

Above the baseline (non-pressurized) tower, hydrogen storage entails, then:
- increased wall thickness
- the addition of 2 pressure vessel endcaps
- additional fixed non-tower expenses, given in Kottenstette
    - additional ladder
    - conduit for weatherized wiring
    - mainframe extension for safe addition of external ladder
    - additional interior access door
    - access mainway & nozzles for pressure vessel

## Expenditure models

### Capital expenditure (CapEx) model

Capital expenses in addition to the baseline tower are given by:
- additional steel costs above baseline tower for wall reinforcement & pressure
        end caps
- additional hardware requirements for/to facilitate hydrogen storage

### Operational expenditure (OpEx) model

Operational expenditure on pressure vessel is modeled roughly following the
relevant costs from `hopp/hydrogen/h2_storage/pressure_vessel`. The resulting
estimates are _rough_. Basically the annual operational expenditures are modeled
as:
$$
OPEX= R_{\mathrm{maint}} \times CAPEX + \mathrm{Hours}_{\mathrm{staff}} \times \mathrm{Wage}
$$
where $R_{\mathrm{maint}}$ is a maintenance rate. We assume:
- $R_{\mathrm{maint}}= 0.03$; i.e.: 3\% of the capital costs must be re-invested each year to cover maintenance
- $\mathrm{Wage}= \$36/\mathrm{hr}$
- $\mathrm{Hours}= 60$: 60 man-hours of maintenance on the pressure vessel per year; this number very roughly derived from the other code

## Unit testing

Unit testing of this module consists of three tests under two approaches:
- comparison with simple geometries
    - cylindrical tower
    - conical tower
- comparison with Kottenstette results
    - issue: Kottenstette results have some assumptions secreted away, so these
            _at best_ are just to ensure the model remains in the ballpark of
            the Kottenstette results
        - specifically, Kottenstette's results table (Table 3 in NREL report)
                imply a partial tower pressure vessel, inducing various
                differences between this code and their results
        - I can't figure out how to size the pressure vessel caps (heads) to get the
                costs that are reported in Kottenstette
    - tests against Table 3:
        - traditional/non-pressurized tower:
            - tower costs within 5%
            - non-tower costs within 5%
        - pressurized tower:
            - wall costs within 10%
            - _top cap within 100%_
            - _bottom cap within 200%_
            - non-tower cost within 10%
            - capacity within 10%
