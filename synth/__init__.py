"""v0.4.2 synth corpus generation package.

Layout:
    contracts/       — Step 0 schema + taxonomy + VAT + manifest + leakage gate
    templates/       — Step 1 HTML+CSS layouts (one .html + .css per template family)
    content_engine.py — Step 2 Faker en_IE + Mimesis Eire + schwifty IBAN/BIC + IE VAT-num
    render.py        — Step 3 WeasyPrint HTML+CSS → PDF (with GTK3 DLL discovery)
    corrupt.py       — Step 4 Augraphy 30/45/20/5 corruption pipeline (NOT YET WRITTEN)

Adam-keyboard interlock per PDR §7: Step 1 prototype = tradesman_rct ONLY,
5 smoke docs, eyeball, then batch other 15 templates.
"""
__version__ = "0.4.2-step1-prototype"
