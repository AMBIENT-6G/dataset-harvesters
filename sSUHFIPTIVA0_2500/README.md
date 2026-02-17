## Design
- Vdda output capacitors 12 pF (50V) (reel) and 200 pF (50V)
- 3V3 and 1V1 capacitors 10 nF (reel)
<img src="https://github.com/AMBIENT-6G/dataset-harvesters/blob/main/sSUHFIPTIVA0_2500/scheme/nxp-harvester-board-2500mhz.jpg" width="600">

## Measurement Conditions
### Features
- Cable loss taken into account --> 1.06 dB loss [2450 MHz] --> ToDo --> Check with calibrated VNA

### Restrictions
- The RF signal generator output power is currently limited to 0 dBm due to the limitations of the energy profiler. (EP V1.2 (2e design))
- The target voltage accuracy is currently set to ±50 mV. This accuracy could be improved in a future energy profiler update once the dynamic range of the programmable load is increased.

## Scheme
<img src="https://github.com/AMBIENT-6G/dataset-harvesters/blob/main/sSUHFIPTIVA0_2500/scheme/harvester.png" width="600">
