import Mathlib

/-!
# Normative threshold theory for the overcaution project

Unbiased drift-diffusion model, noise `s = 1`, drift `v > 0`, threshold `a ≥ 0`.
We take as given (Bogacz et al., 2006) the closed forms
  * error rate      `ER = 1 / (1 + exp (v a))`
  * decision time   `DT = a/(2v) · tanh(v a / 2) = a/(2v) · (1 - 2 ER)`
  * reward rate     `RR = (R (1 - ER) - P ER) / (DT + t)`,  `t = t0 + ITI > 0`.

Verified here:
  1. `tanh_half`            : the `tanh` form and the `1 - 2 ER` form of `DT` agree.
  2. `RR_decomp`            : `RR = g a - P · h a` with `h = ER / (DT + t)`.
  3. `h_strictAntiOn`       : `h` is strictly decreasing on `a ≥ 0`.
  4. `argmax_mono`          : any reward-rate maximiser is monotone in the penalty `P`
                              (no uniqueness, differentiability or concavity needed).
  5. `aversion_overcaution` : error aversion (`λ > 1`) can only raise the chosen threshold.
  6. `compress_low/high`    : compressed penalty sensitivity (`γ < 1`) implies overcaution
                              below the reference penalty and undercaution above it.
-/

open Real Set

namespace DDM

noncomputable def ER (v a : ℝ) : ℝ := 1 / (1 + exp (v * a))

noncomputable def DT (v a : ℝ) : ℝ := a / (2 * v) * (1 - 2 * ER v a)

noncomputable def T (v t a : ℝ) : ℝ := DT v a + t

noncomputable def RR (R P v t a : ℝ) : ℝ := (R * (1 - ER v a) - P * ER v a) / T v t a

noncomputable def h (v t a : ℝ) : ℝ := ER v a / T v t a

/-- `tanh (x/2) = 1 - 2/(1 + e^x)`, so `DT` above is the usual `a/(2v) tanh(va/2)`. -/
theorem tanh_half (x : ℝ) : tanh (x / 2) = 1 - 2 / (1 + exp x) := by
  have hx : exp x = exp (x / 2) * exp (x / 2) := by rw [← exp_add]; ring_nf
  have h1 : 0 < exp (x / 2) := exp_pos _
  have h2 : 0 < exp (-(x / 2)) := exp_pos _
  have hneg : exp (-(x / 2)) = (exp (x / 2))⁻¹ := exp_neg _
  rw [tanh_eq_sinh_div_cosh, sinh_eq, cosh_eq, hx, hneg]
  have h3 : exp (x / 2) + (exp (x / 2))⁻¹ ≠ 0 := by positivity
  have h4 : 1 + exp (x / 2) * exp (x / 2) ≠ 0 := by positivity
  field_simp
  ring

theorem ER_pos (v a : ℝ) : 0 < ER v a := by
  unfold ER; positivity

theorem ER_le_half {v a : ℝ} (hv : 0 < v) (ha : 0 ≤ a) : ER v a ≤ 1 / 2 := by
  unfold ER
  have : 1 ≤ exp (v * a) := one_le_exp (mul_nonneg hv.le ha)
  rw [div_le_div_iff₀ (by positivity) (by norm_num)]
  linarith

theorem ER_strictAnti {v : ℝ} (hv : 0 < v) : StrictAnti (ER v) := by
  intro a b hab
  unfold ER
  apply one_div_lt_one_div_of_lt (by positivity)
  have : exp (v * a) < exp (v * b) := exp_lt_exp.mpr (mul_lt_mul_of_pos_left hab hv)
  linarith

theorem DT_nonneg {v a : ℝ} (hv : 0 < v) (ha : 0 ≤ a) : 0 ≤ DT v a := by
  unfold DT
  have := ER_le_half hv ha
  apply mul_nonneg (div_nonneg ha (by linarith)) (by linarith)

theorem DT_mono {v a b : ℝ} (hv : 0 < v) (ha : 0 ≤ a) (hab : a ≤ b) : DT v a ≤ DT v b := by
  unfold DT
  have hE : ER v b ≤ ER v a := (ER_strictAnti hv).antitone hab
  have hEa := ER_le_half hv ha
  apply mul_le_mul
  · exact div_le_div_of_nonneg_right hab (by linarith)
  · linarith
  · linarith
  · exact div_nonneg (ha.trans hab) (by linarith)

theorem T_pos {v t a : ℝ} (hv : 0 < v) (ht : 0 < t) (ha : 0 ≤ a) : 0 < T v t a := by
  unfold T; linarith [DT_nonneg hv ha]

/-- The reward rate is affine in the penalty: `RR = g - P h`. -/
theorem RR_decomp (R P v t a : ℝ) :
    RR R P v t a = R * (1 - ER v a) / T v t a - P * h v t a := by
  unfold RR h; ring

theorem h_strictAntiOn {v t : ℝ} (hv : 0 < v) (ht : 0 < t) :
    StrictAntiOn (h v t) (Ici 0) := by
  intro a ha b hb hab
  simp only [mem_Ici] at ha hb
  unfold h
  have hTa := T_pos hv ht ha
  have hTle : T v t a ≤ T v t b := by unfold T; linarith [DT_mono hv ha hab.le]
  calc ER v b / T v t b ≤ ER v b / T v t a :=
        div_le_div_of_nonneg_left (ER_pos v b).le hTa hTle
    _ < ER v a / T v t a := div_lt_div_of_pos_right (ER_strictAnti hv hab) hTa

/-- **Main theorem.** Over any admissible set of thresholds `S ⊆ [0, ∞)`, if `a₁` maximises
the reward rate under penalty `P₁` and `a₂` under `P₂ > P₁`, then `a₁ ≤ a₂`. -/
theorem argmax_mono {R v t P₁ P₂ a₁ a₂ : ℝ} {S : Set ℝ} (hS : S ⊆ Ici 0)
    (hv : 0 < v) (ht : 0 < t) (hP : P₁ < P₂) (h₁ : a₁ ∈ S) (h₂ : a₂ ∈ S)
    (opt₁ : ∀ a ∈ S, RR R P₁ v t a ≤ RR R P₁ v t a₁)
    (opt₂ : ∀ a ∈ S, RR R P₂ v t a ≤ RR R P₂ v t a₂) : a₁ ≤ a₂ := by
  by_contra hlt
  push Not at hlt
  have hh : h v t a₁ < h v t a₂ := h_strictAntiOn hv ht (hS h₂) (hS h₁) hlt
  have e₁ := opt₁ a₂ h₂
  have e₂ := opt₂ a₁ h₁
  rw [RR_decomp, RR_decomp] at e₁ e₂
  nlinarith [mul_pos (sub_pos.mpr hP) (sub_pos.mpr hh)]

/-- Error aversion: optimising for an inflated penalty `λ P` (`λ > 1`, `P > 0`) never
yields a lower threshold than the true optimum. -/
theorem aversion_overcaution {R v t P lam a_true a_avers : ℝ} {S : Set ℝ} (hS : S ⊆ Ici 0)
    (hv : 0 < v) (ht : 0 < t) (hP : 0 < P) (hlam : 1 < lam)
    (h₁ : a_true ∈ S) (h₂ : a_avers ∈ S)
    (opt₁ : ∀ a ∈ S, RR R P v t a ≤ RR R P v t a_true)
    (opt₂ : ∀ a ∈ S, RR R (lam * P) v t a ≤ RR R (lam * P) v t a_avers) :
    a_true ≤ a_avers :=
  argmax_mono hS hv ht (by nlinarith) h₁ h₂ opt₁ opt₂

/-- Effective penalty under compressed sensitivity, reference penalty `P₀`. -/
noncomputable def Peff (lam γ P₀ P : ℝ) : ℝ := lam * P₀ * (P / P₀) ^ γ

theorem Peff_gt_of_low {lam γ P₀ P : ℝ} (hlam : 1 ≤ lam) (hγ : γ < 1)
    (hP₀ : 0 < P₀) (hP : 0 < P) (hlow : P < P₀) : P < Peff lam γ P₀ P := by
  unfold Peff
  set x := P / P₀ with hx
  have hx0 : 0 < x := div_pos hP hP₀
  have hx1 : x < 1 := (div_lt_one hP₀).mpr hlow
  have hpow : x < x ^ γ := by
    have := rpow_lt_rpow_of_exponent_gt hx0 hx1 hγ
    simpa [rpow_one] using this
  have hPx : P = P₀ * x := by rw [hx]; field_simp
  have hxγ : 0 < x ^ γ := rpow_pos_of_pos hx0 γ
  calc P = P₀ * x := hPx
    _ < P₀ * x ^ γ := mul_lt_mul_of_pos_left hpow hP₀
    _ = 1 * P₀ * x ^ γ := by ring
    _ ≤ lam * P₀ * x ^ γ := by
        apply mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hlam hP₀.le) hxγ.le

theorem Peff_lt_of_high {lam γ P₀ P : ℝ} (hlam : lam ≤ 1) (hγ : γ < 1)
    (hP₀ : 0 < P₀) (hhigh : P₀ < P) : Peff lam γ P₀ P < P := by
  unfold Peff
  set x := P / P₀ with hx
  have hx1 : 1 < x := (one_lt_div hP₀).mpr hhigh
  have hpow : x ^ γ < x := by
    have := rpow_lt_rpow_of_exponent_lt hx1 hγ
    simpa [rpow_one] using this
  have hPx : P = P₀ * x := by rw [hx]; field_simp
  have hxγ : 0 < x ^ γ := rpow_pos_of_pos (by linarith) γ
  calc lam * P₀ * x ^ γ ≤ 1 * P₀ * x ^ γ := by
        apply mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hlam hP₀.le) hxγ.le
    _ = P₀ * x ^ γ := by ring
    _ < P₀ * x := mul_lt_mul_of_pos_left hpow hP₀
    _ = P := hPx.symm

/-- Compressed sensitivity with `λ ≥ 1`: at every penalty below the reference the chosen
threshold is at least the reward-rate optimum (overcaution at low stakes). -/
theorem compress_low {R v t lam γ P₀ P a_opt a_c : ℝ} {S : Set ℝ} (hS : S ⊆ Ici 0)
    (hv : 0 < v) (ht : 0 < t) (hlam : 1 ≤ lam) (hγ : γ < 1)
    (hP₀ : 0 < P₀) (hP : 0 < P) (hlow : P < P₀) (h₁ : a_opt ∈ S) (h₂ : a_c ∈ S)
    (opt₁ : ∀ a ∈ S, RR R P v t a ≤ RR R P v t a_opt)
    (opt₂ : ∀ a ∈ S, RR R (Peff lam γ P₀ P) v t a ≤ RR R (Peff lam γ P₀ P) v t a_c) :
    a_opt ≤ a_c :=
  argmax_mono hS hv ht (Peff_gt_of_low hlam hγ hP₀ hP hlow) h₁ h₂ opt₁ opt₂

/-- Compressed sensitivity with `λ ≤ 1`: above the reference penalty the chosen threshold
is at most the optimum (undercaution at high stakes). -/
theorem compress_high {R v t lam γ P₀ P a_opt a_c : ℝ} {S : Set ℝ} (hS : S ⊆ Ici 0)
    (hv : 0 < v) (ht : 0 < t) (hlam : lam ≤ 1) (hγ : γ < 1)
    (hP₀ : 0 < P₀) (hhigh : P₀ < P) (h₁ : a_opt ∈ S) (h₂ : a_c ∈ S)
    (opt₁ : ∀ a ∈ S, RR R P v t a ≤ RR R P v t a_opt)
    (opt₂ : ∀ a ∈ S, RR R (Peff lam γ P₀ P) v t a ≤ RR R (Peff lam γ P₀ P) v t a_c) :
    a_c ≤ a_opt :=
  argmax_mono hS hv ht (Peff_lt_of_high hlam hγ hP₀ hhigh) h₂ h₁ opt₂ opt₁

#print axioms argmax_mono
#print axioms aversion_overcaution
#print axioms compress_low
#print axioms compress_high
#print axioms tanh_half

end DDM
