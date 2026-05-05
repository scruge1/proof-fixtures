# Benchmark — Document Ops extraction (proof-fixtures)

Sets scored: 17

## Per-set summary

| Set | N | Pass | Bounce | STR | Bounce % | Silent fail | Avg s/doc |
|---|---|---|---|---|---|---|---|
| 00-test | 1 | 0 | 1 | 0.0% | 100.0% | 0 (0.0%) | 3.1 |
| synth-full-bank_statement | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.3 |
| synth-full-cafe_receipt | 32 | 32 | 0 | 100.0% | 0.0% | 1 (3.1%) | 0.1 |
| synth-full-construction_supplier | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |
| synth-full-credit_note | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |
| synth-full-foreign_currency_mixed_vat | 32 | 32 | 0 | 100.0% | 0.0% | 0 (0.0%) | 0.3 |
| synth-full-gp_medical | 32 | 32 | 0 | 100.0% | 0.0% | 7 (21.9%) | 0.2 |
| synth-full-handwritten_override | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.3 |
| synth-full-mixed_rate_retailer | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |
| synth-full-photographed_receipt | 32 | 32 | 0 | 100.0% | 0.0% | 0 (0.0%) | 0.1 |
| synth-full-professional_services | 32 | 32 | 0 | 100.0% | 0.0% | 2 (6.2%) | 0.2 |
| synth-full-restaurant_thermal | 32 | 5 | 27 | 15.6% | 84.4% | 2 (40.0%) | 0.2 |
| synth-full-solicitor_loe | 32 | 16 | 16 | 50.0% | 50.0% | 3 (18.8%) | 0.2 |
| synth-full-supermarket_per_letter_vat | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |
| synth-full-tradesman_rct | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |
| synth-full-utility_bill | 32 | 10 | 22 | 31.2% | 68.8% | 10 (100.0%) | 0.2 |
| synth-full-vet | 32 | 0 | 32 | 0.0% | 100.0% | 0 (0.0%) | 0.2 |

## Field-level accuracy (when not flagged)

| Set | vendor | total | vat | date |
|---|---|---|---|---|
| 00-test | 1/1 (100%) | 0/1 (0%) | 0/1 (0%) | 1/1 (100%) |
| synth-full-bank_statement | 0/32 (0%) | 31/32 (97%) | 0/32 (0%) | 32/32 (100%) |
| synth-full-cafe_receipt | 31/32 (97%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-construction_supplier | 0/32 (0%) | 0/32 (0%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-credit_note | 32/32 (100%) | 0/32 (0%) | 0/32 (0%) | 32/32 (100%) |
| synth-full-foreign_currency_mixed_vat | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-gp_medical | 25/32 (78%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-handwritten_override | 30/32 (94%) | 0/32 (0%) | 0/32 (0%) | 32/32 (100%) |
| synth-full-mixed_rate_retailer | 21/32 (66%) | 0/32 (0%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-photographed_receipt | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-professional_services | 30/32 (94%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-restaurant_thermal | 28/32 (88%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-solicitor_loe | 22/32 (69%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-supermarket_per_letter_vat | 32/32 (100%) | 0/32 (0%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-tradesman_rct | 0/32 (0%) | 32/32 (100%) | 0/32 (0%) | 32/32 (100%) |
| synth-full-utility_bill | 0/32 (0%) | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) |
| synth-full-vet | 0/32 (0%) | 0/32 (0%) | 32/32 (100%) | 32/32 (100%) |

## Targets at GA

- Straight-through-rate ≥ 97%
- Field-level-when-not-flagged ≥ 99%
- Silent-failure-rate < 2%

Run `python scripts/extract.py --set <slug>` then `python scripts/score.py --set <slug>` to refresh.
