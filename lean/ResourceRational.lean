import Mathlib

/-!
# Resource-rational threshold choice

A participant chooses a threshold `a` from an admissible set `S` to maximise
  `J c a = f a - c * (a - a₀)^2`,
where `f` is the (normalised) reward rate and `c ≥ 0` the cost of moving away from a default `a₀`.
Nothing about `f` is assumed except where stated (no concavity, no differentiability).

Verified:
  1. `closer_to_default`   : the chosen threshold is never farther from the default than the
                             reward-rate optimum (`|â - a₀| ≤ |a* - a₀|`).
  2. `cost_mono`           : a higher adjustment cost moves the choice (weakly) closer to the default.
  3. `between`             : if `f` is unimodal with peak `a*`, the choice lies between `a₀` and `a*`.
  4. `over_low` / `under_high` : consequently, the choice is overcautious when the optimum is below
                             the default (low stakes) and undercautious when it is above (high stakes).
  5. `penalty_mono`        : with `f = g - P h`, `h` strictly decreasing, the choice is monotone in `P`
                             for any fixed cost and default (the cost term does not depend on `P`).
-/

open Set

namespace RRat

variable {S : Set ℝ} {f : ℝ → ℝ} {a₀ : ℝ}

/-- Objective: reward rate minus quadratic adjustment cost. -/
def J (f : ℝ → ℝ) (a₀ c a : ℝ) : ℝ := f a - c * (a - a₀) ^ 2

theorem closer_to_default {c aopt ahat : ℝ} (hc : 0 < c)
    (hopt : ∀ a ∈ S, f a ≤ f aopt) (hhat : ∀ a ∈ S, J f a₀ c a ≤ J f a₀ c ahat)
    (h₁ : aopt ∈ S) (h₂ : ahat ∈ S) :
    (ahat - a₀) ^ 2 ≤ (aopt - a₀) ^ 2 := by
  have e1 := hopt ahat h₂
  have e2 := hhat aopt h₁
  unfold J at e2
  have : c * (ahat - a₀) ^ 2 ≤ c * (aopt - a₀) ^ 2 := by linarith
  exact le_of_mul_le_mul_left this hc

theorem closer_to_default_abs {c aopt ahat : ℝ} (hc : 0 < c)
    (hopt : ∀ a ∈ S, f a ≤ f aopt) (hhat : ∀ a ∈ S, J f a₀ c a ≤ J f a₀ c ahat)
    (h₁ : aopt ∈ S) (h₂ : ahat ∈ S) :
    |ahat - a₀| ≤ |aopt - a₀| :=
  sq_le_sq.mp (closer_to_default hc hopt hhat h₁ h₂)

theorem cost_mono {c₁ c₂ a₁ a₂ : ℝ} (hc : c₁ < c₂)
    (h₁ : a₁ ∈ S) (h₂ : a₂ ∈ S)
    (o₁ : ∀ a ∈ S, J f a₀ c₁ a ≤ J f a₀ c₁ a₁) (o₂ : ∀ a ∈ S, J f a₀ c₂ a ≤ J f a₀ c₂ a₂) :
    (a₂ - a₀) ^ 2 ≤ (a₁ - a₀) ^ 2 := by
  have e1 := o₁ a₂ h₂
  have e2 := o₂ a₁ h₁
  unfold J at e1 e2
  have key : (c₂ - c₁) * ((a₂ - a₀) ^ 2 - (a₁ - a₀) ^ 2) ≤ 0 := by nlinarith
  by_contra hlt
  push Not at hlt
  have : 0 < (c₂ - c₁) * ((a₂ - a₀) ^ 2 - (a₁ - a₀) ^ 2) :=
    mul_pos (sub_pos.mpr hc) (sub_pos.mpr hlt)
  linarith

/-- Unimodality of `f` on `S` with peak `aopt`. -/
def Unimodal (S : Set ℝ) (f : ℝ → ℝ) (aopt : ℝ) : Prop :=
  (∀ x ∈ S, ∀ y ∈ S, x ≤ y → y ≤ aopt → f x ≤ f y) ∧
  (∀ x ∈ S, ∀ y ∈ S, aopt ≤ x → x ≤ y → f y ≤ f x)

/-- If the default is below the optimum, the choice lies in `[a₀, aopt]` (needs both in `S`). -/
theorem between_up {c aopt ahat : ℝ} (hc : 0 < c) (hu : Unimodal S f aopt) (hle : a₀ ≤ aopt)
    (h0 : a₀ ∈ S) (h₁ : aopt ∈ S) (h₂ : ahat ∈ S)
    (hhat : ∀ a ∈ S, J f a₀ c a ≤ J f a₀ c ahat) : a₀ ≤ ahat ∧ ahat ≤ aopt := by
  constructor
  · by_contra hlt
    push Not at hlt
    have hf : f ahat ≤ f a₀ := hu.1 ahat h₂ a₀ h0 hlt.le hle
    have e := hhat a₀ h0
    unfold J at e
    have : 0 < c * (ahat - a₀) ^ 2 := mul_pos hc (by nlinarith)
    nlinarith
  · by_contra hlt
    push Not at hlt
    have hf : f ahat ≤ f aopt := hu.2 aopt h₁ ahat h₂ le_rfl hlt.le
    have e := hhat aopt h₁
    unfold J at e
    have hd : (aopt - a₀) ^ 2 < (ahat - a₀) ^ 2 := by nlinarith
    nlinarith [mul_lt_mul_of_pos_left hd hc]

/-- If the default is above the optimum, the choice lies in `[aopt, a₀]`. -/
theorem between_down {c aopt ahat : ℝ} (hc : 0 < c) (hu : Unimodal S f aopt) (hle : aopt ≤ a₀)
    (h0 : a₀ ∈ S) (h₁ : aopt ∈ S) (h₂ : ahat ∈ S)
    (hhat : ∀ a ∈ S, J f a₀ c a ≤ J f a₀ c ahat) : aopt ≤ ahat ∧ ahat ≤ a₀ := by
  constructor
  · by_contra hlt
    push Not at hlt
    have hf : f ahat ≤ f aopt := hu.1 ahat h₂ aopt h₁ hlt.le le_rfl
    have e := hhat aopt h₁
    unfold J at e
    have hd : (aopt - a₀) ^ 2 < (ahat - a₀) ^ 2 := by nlinarith
    nlinarith [mul_lt_mul_of_pos_left hd hc]
  · by_contra hlt
    push Not at hlt
    have hf : f ahat ≤ f a₀ := hu.2 a₀ h0 ahat h₂ hle hlt.le
    have e := hhat a₀ h0
    unfold J at e
    have : 0 < c * (ahat - a₀) ^ 2 := mul_pos hc (by nlinarith)
    nlinarith

/-- Low stakes (optimum below the default): the choice is at least the optimum (overcaution). -/
theorem over_low {c aopt ahat : ℝ} (hc : 0 < c) (hu : Unimodal S f aopt) (hle : aopt ≤ a₀)
    (h0 : a₀ ∈ S) (h₁ : aopt ∈ S) (h₂ : ahat ∈ S)
    (hhat : ∀ a ∈ S, J f a₀ c a ≤ J f a₀ c ahat) : aopt ≤ ahat :=
  (between_down hc hu hle h0 h₁ h₂ hhat).1

/-- High stakes (optimum above the default): the choice is at most the optimum (undercaution). -/
theorem under_high {c aopt ahat : ℝ} (hc : 0 < c) (hu : Unimodal S f aopt) (hle : a₀ ≤ aopt)
    (h0 : a₀ ∈ S) (h₁ : aopt ∈ S) (h₂ : ahat ∈ S)
    (hhat : ∀ a ∈ S, J f a₀ c a ≤ J f a₀ c ahat) : ahat ≤ aopt :=
  (between_up hc hu hle h0 h₁ h₂ hhat).2

/-- Penalty monotonicity survives the adjustment cost: for `f_P = g - P h` with `h` strictly
decreasing on `S`, choices under `P₁ < P₂` satisfy `a₁ ≤ a₂`. -/
theorem penalty_mono {g h : ℝ → ℝ} {c P₁ P₂ a₁ a₂ : ℝ} (hh : StrictAntiOn h S) (hP : P₁ < P₂)
    (h₁ : a₁ ∈ S) (h₂ : a₂ ∈ S)
    (o₁ : ∀ a ∈ S, J (fun x => g x - P₁ * h x) a₀ c a ≤ J (fun x => g x - P₁ * h x) a₀ c a₁)
    (o₂ : ∀ a ∈ S, J (fun x => g x - P₂ * h x) a₀ c a ≤ J (fun x => g x - P₂ * h x) a₀ c a₂) :
    a₁ ≤ a₂ := by
  by_contra hlt
  push Not at hlt
  have hlt' : h a₁ < h a₂ := hh h₂ h₁ hlt
  have e1 := o₁ a₂ h₂
  have e2 := o₂ a₁ h₁
  unfold J at e1 e2
  nlinarith [mul_pos (sub_pos.mpr hP) (sub_pos.mpr hlt')]

#print axioms closer_to_default_abs
#print axioms cost_mono
#print axioms between_up
#print axioms between_down
#print axioms penalty_mono

end RRat
