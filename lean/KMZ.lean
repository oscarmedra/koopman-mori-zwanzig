import Mathlib

/-!
# Résidu de fermeture et noyau de mémoire de Mori–Zwanzig (cas borné)

Cadre : `E` espace de Hilbert réel, `L : E →L[ℝ] E` borné (modèle de dimension finie /
Galerkin du générateur de Koopman), `P` projection orthogonale (`star P = P`, `P * P = P`),
`Q = 1 - P`. L'adjoint est `star`.

Noyau de mémoire : `K s = P L Q · exp (s • Q L Q) · Q L P`.
-/

open scoped InnerProductSpace

namespace KMZ

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

local notation "Op" => E →L[ℝ] E

/-- `Q = 1 - P` est autoadjoint et idempotent dès que `P` l'est. -/
lemma Q_props {P : Op} (hPs : star P = P) (hPP : P * P = P) :
    star (1 - P) = 1 - P ∧ (1 - P) * (1 - P) = 1 - P := by
  refine ⟨by simp [star_sub, hPs], ?_⟩
  noncomm_ring
  rw [hPP]; abel

/-- Lemme 1 (cas antiadjoint) : `P L Q L P = -(Q L P)† (Q L P)`. -/
theorem K0_eq_neg_gram {P L : Op} (hPs : star P = P) (hPP : P * P = P)
    (hL : star L = -L) :
    P * L * (1 - P) * L * P = -(star ((1 - P) * L * P) * ((1 - P) * L * P)) := by
  obtain ⟨hQs, hQQ⟩ := Q_props hPs hPP
  have : star ((1 - P) * L * P) = -(P * L * (1 - P)) := by
    simp only [star_mul, hPs, hQs, hL]; noncomm_ring
  rw [this]
  calc P * L * (1 - P) * L * P = P * L * ((1 - P) * (1 - P)) * L * P := by rw [hQQ]
    _ = -(-(P * L * (1 - P)) * ((1 - P) * L * P)) := by noncomm_ring

/-- Lemme 2 (identité C*) : `‖P L Q L P‖ = ‖Q L P‖²` dans le cas antiadjoint. -/
theorem norm_K0 {P L : Op} (hPs : star P = P) (hPP : P * P = P) (hL : star L = -L) :
    ‖P * L * (1 - P) * L * P‖ = ‖(1 - P) * L * P‖ ^ 2 := by
  rw [K0_eq_neg_gram hPs hPP hL, norm_neg, sq]
  exact CStarRing.norm_star_mul_self

/-- Lemme 3 : `‖P L Q‖ = ‖Q L† P‖` (cas général) ; dans le cas antiadjoint `= ‖Q L P‖`. -/
theorem norm_PLQ {P L : Op} (hPs : star P = P) (hPP : P * P = P) :
    ‖P * L * (1 - P)‖ = ‖(1 - P) * star L * P‖ := by
  obtain ⟨hQs, -⟩ := Q_props hPs hPP
  rw [← norm_star (P * L * (1 - P))]
  simp only [star_mul, hPs, hQs, mul_assoc]

theorem norm_PLQ_skew {P L : Op} (hPs : star P = P) (hPP : P * P = P)
    (hL : star L = -L) : ‖P * L * (1 - P)‖ = ‖(1 - P) * L * P‖ := by
  rw [norm_PLQ hPs hPP, hL]; simp [mul_neg, neg_mul, norm_neg]

/-- Théorème A (cas borné) : `‖P L Q U Q L P‖ ≤ ‖Q L† P‖ ‖U‖ ‖Q L P‖` pour tout `U`. -/
theorem thmA {P L : Op} (hPs : star P = P) (hPP : P * P = P) (U : Op) :
    ‖P * L * (1 - P) * U * ((1 - P) * L * P)‖
      ≤ ‖(1 - P) * star L * P‖ * ‖U‖ * ‖(1 - P) * L * P‖ := by
  rw [← norm_PLQ hPs hPP]
  calc _ ≤ ‖P * L * (1 - P) * U‖ * ‖(1 - P) * L * P‖ := norm_mul_le _ _
    _ ≤ ‖P * L * (1 - P)‖ * ‖U‖ * ‖(1 - P) * L * P‖ := by
        gcongr; exact norm_mul_le _ _

/-- La dynamique orthogonale `Q L Q` est antiadjointe si `L` l'est. -/
lemma QLQ_skew {P L : Op} (hPs : star P = P) (hPP : P * P = P) (hL : star L = -L) :
    star ((1 - P) * L * (1 - P)) = -((1 - P) * L * (1 - P)) := by
  obtain ⟨hQs, -⟩ := Q_props hPs hPP
  simp only [star_mul, hQs, hL]; noncomm_ring

/-- Théorème B (cas borné) : pour tout `s`, `‖K s‖ ≤ ‖Q L P‖²`, et `‖K 0‖ = ‖Q L P‖²`. -/
theorem thmB {P L : Op} (hPs : star P = P) (hPP : P * P = P) (hL : star L = -L) (s : ℝ) :
    ‖P * L * (1 - P) * NormedSpace.exp (s • ((1 - P) * L * (1 - P))) * ((1 - P) * L * P)‖
      ≤ ‖(1 - P) * L * P‖ ^ 2 := by
  have hskew : s • ((1 - P) * L * (1 - P)) ∈ skewAdjoint Op := by
    rw [skewAdjoint.mem_iff, star_smul, QLQ_skew hPs hPP hL, star_trivial, smul_neg]
  let _ : NormedAlgebra ℚ (E →L[ℝ] E) := NormedAlgebra.restrictScalars ℚ ℝ (E →L[ℝ] E)
  have hU := NormedSpace.exp_mem_unitary_of_mem_skewAdjoint hskew
  have hUn : ‖NormedSpace.exp (s • ((1 - P) * L * (1 - P)))‖ ≤ 1 := by
    by_cases h : Nontrivial Op
    · rw [CStarRing.norm_of_mem_unitary hU]
    · simp only [not_nontrivial_iff_subsingleton] at h
      rw [Subsingleton.elim (NormedSpace.exp _) 0, norm_zero]; exact zero_le_one
  calc _ ≤ ‖(1 - P) * star L * P‖ * ‖NormedSpace.exp (s • ((1 - P) * L * (1 - P)))‖
            * ‖(1 - P) * L * P‖ := thmA hPs hPP _
    _ ≤ ‖(1 - P) * star L * P‖ * 1 * ‖(1 - P) * L * P‖ := by gcongr
    _ = ‖(1 - P) * L * P‖ ^ 2 := by
        rw [hL]; simp [mul_neg, neg_mul, norm_neg, sq]


/-- Théorème G (cas borné, conservatif) : le noyau de mémoire est l'autocorrélation du résidu
sous la dynamique orthogonale : `P L Q exp(s QLQ) Q L P = -(QLP)† exp(s QLQ) (QLP)`. -/
theorem thmG {P L : Op} (hPs : star P = P) (hPP : P * P = P) (hL : star L = -L) (s : ℝ) :
    P * L * (1 - P) * NormedSpace.exp (s • ((1 - P) * L * (1 - P))) * ((1 - P) * L * P)
      = -(star ((1 - P) * L * P) * NormedSpace.exp (s • ((1 - P) * L * (1 - P)))
          * ((1 - P) * L * P)) := by
  obtain ⟨hQs, -⟩ := Q_props hPs hPP
  have : star ((1 - P) * L * P) = -(P * L * (1 - P)) := by
    simp only [star_mul, hPs, hQs, hL]; noncomm_ring
  rw [this]; noncomm_ring

/-- Cas général (borné) : le noyau est `(Q L† P)† exp(s QLQ) (Q L P)` : il est entièrement déterminé
par le résidu de Koopman `QLP`, le résidu adjoint `QL†P` et la dynamique orthogonale. -/
theorem thmG_general {P L : Op} (hPs : star P = P) (hPP : P * P = P) (U : Op) :
    P * L * (1 - P) * U * ((1 - P) * L * P)
      = star ((1 - P) * star L * P) * U * ((1 - P) * L * P) := by
  obtain ⟨hQs, -⟩ := Q_props hPs hPP
  simp only [star_mul, hPs, hQs, star_star, mul_assoc]

end KMZ

#print axioms KMZ.K0_eq_neg_gram
#print axioms KMZ.norm_K0
#print axioms KMZ.norm_PLQ_skew
#print axioms KMZ.thmA
#print axioms KMZ.thmB
#print axioms KMZ.thmG
#print axioms KMZ.thmG_general
