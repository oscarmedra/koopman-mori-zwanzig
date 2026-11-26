import Mathlib

/-! Identite de Mori–Zwanzig discrete (exacte, purement algebrique). -/
namespace KMZ

variable {R : Type*} [Ring R]

/-- Dyson discret : `K^(k+1) P = Σ_{l≤k} K^(k-l) P K (QK)^l P + (QK)^(k+1) P`. -/
theorem dyson_disc (P K : R) (k : ℕ) :
    K ^ (k + 1) * P
      = (∑ l ∈ Finset.range (k + 1), K ^ (k - l) * P * K * ((1 - P) * K) ^ l * P)
        + ((1 - P) * K) ^ (k + 1) * P := by
  induction k with
  | zero => simp; noncomm_ring
  | succ k ih =>
    rw [pow_succ', mul_assoc, ih, mul_add, Finset.mul_sum, Finset.sum_range_succ _ (k + 1)]
    have h1 : ∀ l ∈ Finset.range (k + 1),
        K * (K ^ (k - l) * P * K * ((1 - P) * K) ^ l * P)
          = K ^ (k + 1 - l) * P * K * ((1 - P) * K) ^ l * P := by
      intro l hl
      have : k + 1 - l = (k - l) + 1 := by
        have := Finset.mem_range.mp hl; omega
      rw [this, pow_succ']; noncomm_ring
    rw [Finset.sum_congr rfl h1]
    simp only [Nat.sub_self, pow_zero, one_mul]
    rw [pow_succ' ((1 - P) * K) (k + 1)]
    noncomm_ring

/-- MZ discret : `P K^(k+1) P = Σ_{l≤k} (P K^(k-l) P) Ω_l`, `Ω_l = P K (QK)^l P`. -/
theorem mz_disc (P K : R) (hPP : P * P = P) (k : ℕ) :
    P * K ^ (k + 1) * P
      = ∑ l ∈ Finset.range (k + 1), (P * K ^ (k - l) * P) * (P * K * ((1 - P) * K) ^ l * P) := by
  have hPQ : P * (1 - P) = 0 := by rw [mul_sub, mul_one, hPP, sub_self]
  rw [mul_assoc, dyson_disc, mul_add, Finset.mul_sum]
  have hlast : P * (((1 - P) * K) ^ (k + 1) * P) = 0 := by
    rw [pow_succ', ← mul_assoc, ← mul_assoc, ← mul_assoc, hPQ, zero_mul, zero_mul, zero_mul]
  rw [hlast, add_zero]
  refine Finset.sum_congr rfl fun l _ => ?_
  calc P * (K ^ (k - l) * P * K * ((1 - P) * K) ^ l * P)
      = P * K ^ (k - l) * (P * P) * K * ((1 - P) * K) ^ l * P := by rw [hPP]; noncomm_ring
    _ = _ := by noncomm_ring

/-- Forme « résidu » du noyau : `Ω_{l+1} = (P K Q) (Q K Q)^l (Q K P)`. -/
theorem omega_residu (P K : R) (hPP : P * P = P) (l : ℕ) :
    P * K * ((1 - P) * K) ^ (l + 1) * P
      = (P * K * (1 - P)) * ((1 - P) * K * (1 - P)) ^ l * ((1 - P) * K * P) := by
  have hQQ : (1 - P) * (1 - P) = 1 - P := by noncomm_ring; rw [hPP]; abel
  -- (QK)^(m+1) = Q (KQ)^m K
  have h1 : ∀ m : ℕ, ((1 - P) * K) ^ (m + 1) = (1 - P) * (K * (1 - P)) ^ m * K := by
    intro m
    induction m with
    | zero => simp
    | succ m ih => rw [pow_succ, ih, pow_succ]; noncomm_ring
  -- Q (KQ)^m = (QKQ)^m Q
  have h2 : ∀ m : ℕ, (1 - P) * (K * (1 - P)) ^ m = ((1 - P) * K * (1 - P)) ^ m * (1 - P) := by
    intro m
    induction m with
    | zero => simp
    | succ m ih =>
      rw [pow_succ, ← mul_assoc, ih, pow_succ]; simp only [mul_assoc, hQQ]
  rw [h1]
  calc P * K * ((1 - P) * (K * (1 - P)) ^ l * K) * P
      = P * K * (((1 - P) * (1 - P)) * (K * (1 - P)) ^ l) * K * P := by rw [hQQ]; noncomm_ring
    _ = P * K * (1 - P) * ((1 - P) * (K * (1 - P)) ^ l) * K * P := by noncomm_ring
    _ = _ := by rw [h2]; noncomm_ring

#print axioms mz_disc
#print axioms omega_residu

end KMZ
