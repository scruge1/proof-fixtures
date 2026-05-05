# Benchmark — Document Ops extraction (proof-fixtures)

Sets scored: 16

## Per-set summary

| Set | N | Pass | Bounce | STR | Bounce % | Silent fail | Avg s/doc |
|---|---|---|---|---|---|---|---|
| 00-test | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 3.1 |
| synth-corrupted-bank_statement__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 231.9 |
| synth-corrupted-cafe_receipt__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 97.6 |
| synth-corrupted-construction_supplier__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 186.0 |
| synth-corrupted-credit_note__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 178.8 |
| synth-corrupted-gp_medical__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 129.3 |
| synth-corrupted-handwritten_override__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 102.9 |
| synth-corrupted-mixed_rate_retailer__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 177.0 |
| synth-corrupted-photographed_receipt__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 95.2 |
| synth-corrupted-professional_services__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 121.2 |
| synth-corrupted-restaurant_thermal__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 155.3 |
| synth-corrupted-solicitor_loe__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 203.0 |
| synth-corrupted-supermarket_per_letter_vat__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 119.2 |
| synth-corrupted-tradesman_rct__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 217.1 |
| synth-corrupted-utility_bill__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 185.5 |
| synth-corrupted-vet__pilot1-light | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 254.4 |

## Field-level accuracy (when not flagged)

| Set | vendor | total | vat | date |
|---|---|---|---|---|
| 00-test | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| synth-corrupted-bank_statement__pilot1-light | 0/1 (0%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) |
| synth-corrupted-cafe_receipt__pilot1-light | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-construction_supplier__pilot1-light | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-credit_note__pilot1-light | 0/1 (0%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| synth-corrupted-gp_medical__pilot1-light | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-handwritten_override__pilot1-light | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| synth-corrupted-mixed_rate_retailer__pilot1-light | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-photographed_receipt__pilot1-light | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-professional_services__pilot1-light | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-restaurant_thermal__pilot1-light | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-solicitor_loe__pilot1-light | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-supermarket_per_letter_vat__pilot1-light | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-tradesman_rct__pilot1-light | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) | 1/1 (100%) |
| synth-corrupted-utility_bill__pilot1-light | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |
| synth-corrupted-vet__pilot1-light | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) | 1/1 (100%) |

## Targets at GA

- Straight-through-rate ≥ 97%
- Field-level-when-not-flagged ≥ 99%
- Silent-failure-rate < 2%

Run `python scripts/extract.py --set <slug>` then `python scripts/score.py --set <slug>` to refresh.
