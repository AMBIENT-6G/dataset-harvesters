
# Scheme
<img src="https://github.com/AMBIENT-6G/dataset-harvesters/blob/main/sSUHFIPTIVA0_875/scheme/harvester.png" width="600">

# Design
- Vdda output capacitors 12 pF (50V) (reel) and 200 pF (50V)
- 3V3 and 1V1 capacitors 10 nF (reel)
<img src="https://github.com/AMBIENT-6G/dataset-harvesters/blob/main/sSUHFIPTIVA0_875/scheme/nxp-harvester-board-875mhz.jpg" width="600">

# Features
- Cable loss taken into account --> 0.55 dB loss [915 MHz]

# Restrictions
- The RF signal generator output power is currently limited to 0 dBm due to the limitations of the energy profiler. (EP V1.2 (2e design))
- The target voltage accuracy is currently set to ±50 mV. This accuracy could be improved in a future energy profiler update once the dynamic range of the programmable load is increased.
