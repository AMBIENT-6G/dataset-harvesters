# Dataset Several (RF) Harvesters

This repository presents the evaluation and data collection of multiple energy harvesters, focusing on their performance under representative operating conditions. The measurements aim to provide a comparative overview of the capabilities and limitations of the different harvesting solutions. An overview of the evaluated harvesters and their key characteristics is provided in the table below.

| Harvester | Crucial component(s) | Tuning [MHz] | Link | Charger (PMU) | Voltage regulator (PMU) |
|-|-|-|-|-|-|
| 2AAEM30940C031 rectifier (low) | SMS7630005LF | 915 | [SMS7630005LF_915](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/SMS7630005LF_915) | No | No |
| 2AAEM30940C031 rectifier (high) | SMS7621005LF | 915 | [SMS7621005LF_915](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/SMS7621005LF_915) | No | No |
| 2AAEM30940C041 rectifier (low) | SMS7630005LF | 2450 | [SMS7630005LF_2450](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/SMS7630005LF_2450) | No | No |
| 2AAEM30940C041 rectifier (high) | SMS7621005LF | 2450 | [SMS7621005LF_2450](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/SMS7621005LF_2450) | No | No |
| NXP | sSUHFIPTIVA0 | 875 | [sSUHFIPTIVA0_875](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/sSUHFIPTIVA0_875) | No | No |
| NXP | sSUHFIPTIVA0 | 2500 | [sSUHFIPTIVA0_2500](https://github.com/AMBIENT-6G/dataset-harvesters/tree/main/sSUHFIPTIVA0_2500) | No | No |

## Static Webtool (GitHub Pages)

- Webtool path on Pages: `https://ambient-6g.github.io/dataset-harvesters/webtool/`
- Static app source: `webtool/`
- Term glossary source of truth: `terms.json`

### Generate `json-export` from SQLite

1. Open `data.db` (or `032026_data.db`) in **[DB Browser for SQLite](https://sqlitebrowser.org/)**.
2. Export each harvester table as a JSON file.
3. Save the files in `webtool/json-export/` using these exact names:
   `AEM40940.json`, `P1110B.json`, `P2110B.json`, `SMS7621005LF.json`,
   `SMS7630005LF.json`, `sSUHFIPTIVA0.json`.
4. Validate JSON locally (optional): `python3 -m json.tool webtool/json-export/<file>.json > /dev/null`
5. Commit and push your changes. A push to `main` with changes in `webtool/**` will trigger the Pages workflow and publish the updated data.
