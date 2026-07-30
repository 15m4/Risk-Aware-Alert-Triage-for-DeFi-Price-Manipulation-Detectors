# Data for "Risk-Aware Alert Triage for DeFi Price-Manipulation Detectors"

`incident_table_VERIFIED_v2.csv` — the 54-incident corpus used in the paper,
reconstructed from Table 1 of Xie et al., "DeFort" (ISSTA'24), with each
incident's vulnerability-class label independently verified (see the paper
for methodology and limitations).

## Columns

- `idx`, `name`, `date`, `loss_usd`, `chain` — transcribed from DeFort's Table 1.
- `vclass` — vulnerability class (RSF taxonomy: SCL1-5, PL1-5, ASL1-3).
- `confidence` — H / M / L, per-incident confidence in the `vclass` label.
- `rank` — RSF Table 7 priority rank (1 = highest) for that class.

## Citation

- Xie et al., "DeFort: Automatic Detection and Analysis of Price Manipulation
  Attacks in DeFi Applications," ISSTA 2024.
- Arora et al., "A Risk Scoring Framework for Critical Infrastructure of
  Decentralized Finance (DeFi)," Blockchain: Research and Applications, 2026.
