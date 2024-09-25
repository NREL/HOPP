# Proton Exchange Membrane Water Electrolysis Balance-of-Plant

This balance-of-plant (BOP) model is derived from Wang et. al (2023). It is represented as an electrical BOP efficiency curve (kWh/kg) at different operating ratios. The operating ratios are the percentage of rated power provided to the electrolyzer.

The electrical BOP curve includes a transformer and a rectifier to condition alternating current power.

**NOTE**: BOP assumes AC current as an input and assumes power electronics for AC to DC conversion. BOP efficiency curve has not been optimized for economies of scale or other electrical infrastructure connections.


Citation for BOP model.
```
@Article{en16134964,
AUTHOR = {Wang, Xiaohua and Star, Andrew G. and Ahluwalia, Rajesh K.},
TITLE = {Performance of Polymer Electrolyte Membrane Water Electrolysis Systems: Configuration, Stack Materials, Turndown and Efficiency},
JOURNAL = {Energies},
VOLUME = {16},
YEAR = {2023},
NUMBER = {13},
ARTICLE-NUMBER = {4964},
URL = {https://www.mdpi.com/1996-1073/16/13/4964},
ISSN = {1996-1073},
DOI = {10.3390/en16134964}
}
```