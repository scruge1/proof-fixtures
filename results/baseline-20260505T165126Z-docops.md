# Benchmark — Document Ops extraction (proof-fixtures)

Sets scored: 17

## Per-set summary

| Set | N | Pass | Bounce | STR | Bounce % | Silent fail | Avg s/doc |
|---|---|---|---|---|---|---|---|
| 00-test | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 3.1 |
| synth-full-bank_statement__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 44.9 |
| synth-full-cafe_receipt__pilot5 | 5 | 5 | 0 | 100.0% | 0.0% | 0 (0.0%) | 0.1 |
| synth-full-construction_supplier__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |
| synth-full-credit_note__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 20.1 |
| synth-full-foreign_currency_mixed_vat__pilot5 | 5 | 5 | 0 | 100.0% | 0.0% | 0 (0.0%) | 0.2 |
| synth-full-gp_medical__pilot5 | 5 | 5 | 0 | 100.0% | 0.0% | 1 (20.0%) | 0.1 |
| synth-full-handwritten_override__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 75.5 |
| synth-full-mixed_rate_retailer__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 10.8 |
| synth-full-photographed_receipt__pilot5 | 5 | 5 | 0 | 100.0% | 0.0% | 0 (0.0%) | 0.1 |
| synth-full-professional_services__pilot5 | 5 | 5 | 0 | 100.0% | 0.0% | 1 (20.0%) | 0.1 |
| synth-full-restaurant_thermal__pilot5 | 5 | 1 | 4 | 20.0% | 80.0% | 0 (0.0%) | 0.2 |
| synth-full-solicitor_loe__pilot5 | 5 | 2 | 3 | 40.0% | 60.0% | 1 (50.0%) | 0.2 |
| synth-full-supermarket_per_letter_vat__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |
| synth-full-tradesman_rct__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 11.6 |
| synth-full-utility_bill__pilot5 | 5 | 2 | 3 | 40.0% | 60.0% | 2 (100.0%) | 4.0 |
| synth-full-vet__pilot5 | 5 | 0 | 5 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |

## Field-level accuracy (when not flagged)

| Set | vendor | total | vat | date |
|---|---|---|---|---|
| 00-test | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| synth-full-bank_statement__pilot5 | 0/5 (0%) | 5/5 (100%) | 0/5 (0%) | 5/5 (100%) |
| synth-full-cafe_receipt__pilot5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-construction_supplier__pilot5 | 0/5 (0%) | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-credit_note__pilot5 | 5/5 (100%) | 0/5 (0%) | 0/5 (0%) | 5/5 (100%) |
| synth-full-foreign_currency_mixed_vat__pilot5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-gp_medical__pilot5 | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-handwritten_override__pilot5 | 5/5 (100%) | 0/5 (0%) | 0/5 (0%) | 5/5 (100%) |
| synth-full-mixed_rate_retailer__pilot5 | 1/5 (20%) | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-photographed_receipt__pilot5 | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-professional_services__pilot5 | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-restaurant_thermal__pilot5 | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-solicitor_loe__pilot5 | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-supermarket_per_letter_vat__pilot5 | 5/5 (100%) | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-tradesman_rct__pilot5 | 0/5 (0%) | 5/5 (100%) | 0/5 (0%) | 5/5 (100%) |
| synth-full-utility_bill__pilot5 | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) |
| synth-full-vet__pilot5 | 0/5 (0%) | 0/5 (0%) | 5/5 (100%) | 5/5 (100%) |

## Targets at GA

- Straight-through-rate ≥ 97%
- Field-level-when-not-flagged ≥ 99%
- Silent-failure-rate < 2%

Run `python scripts/extract.py --set <slug>` then `python scripts/score.py --set <slug>` to refresh.
