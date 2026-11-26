# KMZ : réduction de modèles Koopman–Mori-Zwanzig, preuves formelles Lean et bibliothèque Python

*KMZ: Koopman–Mori-Zwanzig model reduction, Lean formal proofs and Python library*

**Auteurs / Authors :** Nouha DRAME ([ORCID 0009-0003-6018-9802](https://orcid.org/0009-0003-6018-9802)), Déthié DIONE, encadrant / supervisor ([ORCID 0000-0002-9872-2479](https://orcid.org/0000-0002-9872-2479)) — PIED / UNCHK

**Dépôt / Repository :** https://github.com/oscarmedra/koopman-mori-zwanzig · **DOI :** `[DOI Zenodo, à ajouter après la première release]`

Ce dépôt accompagne l'article de N. DRAME et D. DIONE (en préparation, 2026). Il contient :
This repository accompanies the paper by N. DRAME and D. DIONE (in preparation, 2026). It contains:

| Dossier / Folder | Contenu / Content |
|---|---|
| `lean/` | Preuves formelles Lean 4 + Mathlib (`KMZ.lean`, `MZdisc.lean`), sans `sorry` / Lean 4 + Mathlib formal proofs, `sorry`-free |
| `python/` | Bibliothèque Python `kmz` 0.2.2 (numpy), tests, tutoriel, exemples / Python library `kmz` 0.2.2, tests, tutorial, examples |

---

## Correspondance théorèmes ↔ Lean / Theorem ↔ Lean correspondence

Numérotation de l'article (document de démonstrations). Les énoncés Lean portent sur un opérateur **borné**
`L : E →L[ℝ] E` sur un espace de Hilbert réel complet `E`, avec `P` projection orthogonale
(`star P = P`, `P * P = P`), `Q = 1 - P`, et `star` l'adjoint. Le passage aux générateurs non bornés est fait sur papier uniquement.
Numbering follows the paper. Lean statements concern a **bounded** operator; the extension to unbounded generators is on paper only.

| Énoncé (article) / Statement | Nom Lean / Lean name | Fichier / File | Énoncé formalisé / Formal statement |
|---|---|---|---|
| Théorème A | `KMZ.thmA` (avec `norm_PLQ`) | `KMZ.lean` | ‖PLQ·U·QLP‖ ≤ ‖QL*P‖‖U‖‖QLP‖ pour tout `U` (U remplace S(s)) |
| Théorème B | `KMZ.K0_eq_neg_gram`, `KMZ.norm_K0`, `KMZ.thmB` | `KMZ.lean` | L* = −L ⇒ PLQLP = −(QLP)*(QLP), ‖PLQLP‖ = ‖QLP‖², ‖PLQ exp(sQLQ) QLP‖ ≤ ‖QLP‖² |
| (lemme auxiliaire) | `KMZ.norm_PLQ`, `KMZ.norm_PLQ_skew` | `KMZ.lean` | ‖PLQ‖ = ‖QL*P‖, et = ‖QLP‖ si L* = −L |
| Théorème G (mémoire = résidu) | `KMZ.thmG`, `KMZ.thmG_general` | `KMZ.lean` | PLQ exp(sQLQ) QLP = −(QLP)* exp(sQLQ) (QLP) ; cas général PLQ·U·QLP = (QL*P)*·U·QLP |
| Proposition 7 (MZ discret exact) | `KMZ.dyson_disc`, `KMZ.mz_disc` | `MZdisc.lean` | PK^{k+1}P = Σ_{l≤k} (PK^{k−l}P)(PK(QK)^l P) dans tout anneau avec P² = P |
| Théorème H (i) (mémoire discrète et résidu) | `KMZ.omega_residu` | `MZdisc.lean` | PK(QK)^{l+1}P = PKQ (QKQ)^l QKP |

Non formalisés (papier + vérification numérique) / Not formalised: Lemmes 1–5, Prop. 1–6 et 8, Théorèmes C, D, E, F, Cor. 2–4, Th. H (ii)–(iii).
Dans le cas borné le noyau est écrit PLQ exp(sQLQ) QLP ; l'égalité avec PL e^{sQL} QL|_V (Lemme « dynamique orthogonale ») n'est pas formalisée.

## Vérifier les preuves / Checking the proofs

Versions figées / pinned: Lean `v4.37.0-rc1` (`lean/lean-toolchain`), Mathlib tag `v4.37.0-rc1`,
commit `31ce06eaadf4a200c0e1407c5f10953cf606b8f8` (`lean/lakefile.toml`, `lean/lake-manifest.json`).

```bash
# 1. Installer elan (gestionnaire de versions Lean) / install elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
source ~/.elan/env

# 2. Compiler / build
cd lean
lake exe cache get     # télécharge Mathlib précompilé (~5 Go) / downloads prebuilt Mathlib
lake build             # compile KMZ.lean et MZdisc.lean

# 3. Vérifier l'absence de sorry / check there is no sorry
grep -n "sorry" *.lean || echo "aucun sorry / no sorry"
```

`lake build` doit finir par `Build completed successfully` et afficher, via `#print axioms`, uniquement
les axiomes standard `propext`, `Classical.choice`, `Quot.sound` (aucun `sorryAx`).
The build must end with `Build completed successfully` and `#print axioms` lists only the standard axioms (no `sorryAx`).

## Utiliser le code Python / Using the Python code

```bash
cd python
pip install -e ".[tests]"      # numpy, matplotlib, pytest
pytest -q tests/test_unit.py   # tests unitaires / unit tests
python tutorial/tutorial_duffing.py
```

Documentation : `python/README.md` et `python/tutorial/TUTORIAL.md`.

## Citer / How to cite

Voir `CITATION.cff` (bouton « Cite this repository » sur GitHub) et le DOI Zenodo : `[DOI Zenodo]`.
See `CITATION.cff` and the Zenodo DOI.

## Licence / License

MIT, voir `LICENSE`.
