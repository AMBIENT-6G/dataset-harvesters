# Dataset Several (RF) Harvesters

This repository presents the evaluation and data collection of multiple energy harvesters, focusing on their performance under representative operating conditions. The measurements aim to provide a comparative overview of the capabilities and limitations of the different harvesting solutions. An overview of the evaluated harvesters and their key characteristics is provided in the table below.

| Harvester | Crucial component(s) | Tuning [MHz] | Link | Charger (PMU) | Voltage regulator (PMU) |
|-|-|-|-|-|-|
| NXP | sSUHFIPTIVA0 | 875 | [sSUHFIPTIVA0_875](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/sSUHFIPTIVA0_875) | No | No |
| NXP | sSUHFIPTIVA0 | 2500 | [sSUHFIPTIVA0_2500](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/sSUHFIPTIVA0_2500) | No | No |

## Static Webtool (GitHub Pages)

- Webtool path on Pages: `https://ambient-6g.github.io/dataset-harvesters/webtool/`
- Static app source: `webtool/`
- Term glossary source of truth: `terms.json`

### Local preview

1. Start a static server from repository root:

```bash
python3 -m http.server 8000
```

2. Open:

`http://localhost:8000/webtool/`

### Maintenance note

When `terms.json` is updated, the webtool will pick up the changes directly after redeploy (or immediately during local static preview).
