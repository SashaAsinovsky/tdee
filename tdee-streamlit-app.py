import streamlit as st
import numpy as np
import pandas as pd

# --- Constants and Coefficients ---
KCAL_PER_KG_TISSUE = 7700
SURPLUS_EFFICIENCY = 0.85
DEFICIT_EFFICIENCY = 1.00
TAU_BMR_ADAPTATION = 10.0 
TAU_NEAT_ADAPTATION = 2.5 
KCAL_PER_STEP_BASE_FACTOR = 0.00062 
SEDENTARY_WELLFED_RMR_MULTIPLIER = 1.6 # For "non-locomotor NEAT & upregulation"
CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR = 1.2 # BMR*1.2 for (BMR+NEAT) at very low intake

# --- Helper Functions ---
def kg_to_lbs(kg): return kg * 2.20462
def lbs_to_kg(lbs): return lbs / 2.20462
def ft_in_to_cm(ft, inch): return (ft * 30.48) + (inch * 2.54)

# --- Core Calculation Functions ---
def calculate_ffm_fm(weight_kg, body_fat_percentage):
    """Calculates Fat-Free Mass (FFM) and Fat Mass (FM) in kg."""
    fm_kg = weight_kg * (body_fat_percentage / 100.0)
    ffm_kg = weight_kg - fm_kg
    return ffm_kg, fm_kg

def calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg):
    """
    Calculates RMR using Pontzer's FFM-based equation:
    exp(-0.954 + 0.707 * Ln(FFM in kg) + 0.019 * Ln(FM in kg)) * 238.853 (to convert MJ to kcal)
    """
    if ffm_kg <= 0: return 0
    try:
        # Use a tiny value for fm_kg if it's zero to avoid log(0) error,
        # as some fat mass is usually present.
        fm_kg_adjusted_for_log = fm_kg if fm_kg > 0.001 else 0.001
        
        term_ffm = 0.707 * np.log(ffm_kg)
        term_fm = 0.019 * np.log(fm_kg_adjusted_for_log)
        
        # The original Pontzer formula gives result in MJ/day
        rmr_mj_day = np.exp(-0.954 + term_ffm + term_fm)
        rmr_kcal_day = rmr_mj_day * 238.8529999 # Conversion factor from MJ to kcal
        return rmr_kcal_day
    except Exception:
        # Return 0 or handle error appropriately if calculation fails
        return 0

def calculate_mifflin_st_jeor_rmr(weight_kg, height_cm, age_years, sex): # Kept for fallback
    """Calculates RMR using Mifflin-St Jeor equation."""
    if sex == "Male":
        rmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) + 5
    else:  # Female
        rmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) - 161
    return rmr

def calculate_dlw_tdee(ffm_kg, fm_kg): # Pontzer FFM-based TDEE
    """
    Calculates TDEE using Pontzer's FFM-based DLW equation:
    exp(-1.102 + 0.916 * Ln(FFM in kg) - 0.030 * Ln(FM in kg)) * 238.83 (to convert MJ to kcal)
    """
    if ffm_kg <= 0: return 0
    try:
        fm_kg_adjusted_for_log = fm_kg if fm_kg > 0.001 else 0.001

        term_ffm = 0.916 * np.log(ffm_kg)
        term_fm = -0.030 * np.log(fm_kg_adjusted_for_log) # Note the negative coefficient for FM
        
        tdee_mj_day = np.exp(-1.102 + term_ffm + term_fm)
        tdee_kcal_day = tdee_mj_day * 238.83 # Conversion factor from MJ to kcal
        return tdee_kcal_day
    except Exception:
        return 0

def get_pal_multiplier_for_heuristic(activity_steps): # Used for UAB fallback
    """Maps step counts to a Physical Activity Level multiplier category."""
    if activity_steps < 5000: return 1.3 
    elif activity_steps < 7500: return 1.45
    elif activity_steps < 10000: return 1.6
    elif activity_steps < 12500: return 1.75
    else: return 1.95 # Corresponds to very active / athlete
    
def adjust_reported_intake(reported_intake_kcal, weight_trend, weight_change_rate_kg_wk):
    """Adjusts reported intake based on weight trend to estimate true intake."""
    kcal_discrepancy_per_day = (weight_change_rate_kg_wk * KCAL_PER_KG_TISSUE) / 7.0
    adjusted_intake = reported_intake_kcal + kcal_discrepancy_per_day
    return adjusted_intake

def calculate_tef(intake_kcal, protein_g):
    """Calculates Thermic Effect of Food."""
    protein_kcal = protein_g * 4.0
    protein_percentage = (protein_kcal / intake_kcal) * 100.0 if intake_kcal > 0 else 0.0
    tef_factor = 0.10 # Default for mixed diet
    if protein_percentage > 25.0: tef_factor = 0.15 # Higher for high protein
    elif protein_percentage > 15.0: tef_factor = 0.12 # Moderate protein
    return intake_kcal * tef_factor

def calculate_cold_thermogenesis(typical_indoor_temp_f, minutes_cold_exposure_daily):
    """Estimates calories burned due to cold exposure."""
    k_c_per_degree_f_day = 4.0; comfort_temp_f = 68.0; cold_kcal_indoor = 0.0
    if typical_indoor_temp_f < comfort_temp_f:
        temp_diff_f_indoor = comfort_temp_f - typical_indoor_temp_f
        # Prorate indoor cold effect by the proportion of the day spent indoors
        cold_kcal_indoor = temp_diff_f_indoor * k_c_per_degree_f_day * ((24.0*60.0 - minutes_cold_exposure_daily) / (24.0*60.0))
    # Simplified outdoor cold exposure effect
    cold_kcal_outdoor = (minutes_cold_exposure_daily / 60.0) * 30.0 # Assume 30 kcal/hr of mild cold exposure
    return max(0, cold_kcal_indoor) + max(0, cold_kcal_outdoor)

def calculate_immune_fever_effect(has_fever_illness, peak_fever_temp_f, current_bmr_adaptive):
    """Estimates extra calories burned due to fever."""
    if not has_fever_illness or peak_fever_temp_f <= 99.0: return 0.0
    temp_diff_f = peak_fever_temp_f - 98.6 # Normal body temp
    if temp_diff_f <= 0: return 0.0 # No fever if not above normal
    # Approx 6.5% RMR increase per 1°F of fever
    return current_bmr_adaptive * temp_diff_f * 0.065

def calculate_ffmi_fmi(ffm_kg, fm_kg, height_m):
    """Calculates Fat-Free Mass Index (FFMI) and Fat Mass Index (FMI)."""
    ffmi = ffm_kg / (height_m**2) if height_m > 0 and ffm_kg >=0 else 0
    fmi = fm_kg / (height_m**2) if height_m > 0 and fm_kg >=0 else 0
    return ffmi, fmi

def calculate_implied_activity_breakdown(tdee_dlw, rmr_pontzer_ffm, weight_kg):
    """Derives implied activity components from FFM-based TDEE (DLW)."""
    if tdee_dlw <= 0 or rmr_pontzer_ffm <= 0 or weight_kg <= 0:
        return 0, 0, 0, 0 

    # TDEE representing a sedentary but well-fed state (RMR * 1.6)
    tdee_sedentary_wellfed_floor = rmr_pontzer_ffm * SEDENTARY_WELLFED_RMR_MULTIPLIER
    
    # TEF at this sedentary well-fed floor (assuming intake matches this TDEE)
    tef_at_sedentary_floor = tdee_sedentary_wellfed_floor * 0.10 # General 10% TEF
    
    # Energy for non-locomotor NEAT, immune, repair, etc., to reach the 1.6x RMR TDEE
    energy_non_locomotor_upregulation = tdee_sedentary_wellfed_floor - rmr_pontzer_ffm - tef_at_sedentary_floor
    energy_non_locomotor_upregulation = max(0, energy_non_locomotor_upregulation)

    # Additional energy from DLW TDEE (if any) beyond this well-fed sedentary floor is for locomotion/exercise
    energy_for_locomotion = tdee_dlw - tdee_sedentary_wellfed_floor
    energy_for_locomotion = max(0, energy_for_locomotion) 

    kcal_per_step = KCAL_PER_STEP_BASE_FACTOR * weight_kg
    implied_locomotor_steps = energy_for_locomotion / kcal_per_step if kcal_per_step > 0 else 0
    
    return energy_non_locomotor_upregulation, energy_for_locomotion, implied_locomotor_steps, tdee_sedentary_wellfed_floor

# --- Main Simulation Logic ---
def simulate_tdee_adaptation(inputs, num_days_to_simulate=14):
    ffm_kg, fm_kg = calculate_ffm_fm(inputs['weight_kg'], inputs['body_fat_percentage'])
    initial_bmr_baseline = calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg)
    if initial_bmr_baseline == 0: 
        initial_bmr_baseline = calculate_mifflin_st_jeor_rmr(inputs['weight_kg'], inputs['height_cm'], inputs['age_years'], inputs['sex'])
        if inputs.get('streamlit_object'): inputs['streamlit_object'].warning("Pontzer FFM-RMR failed. Using Mifflin as initial BMR.")
    
    LAB = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR 
    UAB = calculate_dlw_tdee(ffm_kg, fm_kg)
    pal_for_uab_heuristic = get_pal_multiplier_for_heuristic(inputs['avg_daily_steps'])
    if UAB == 0 or UAB < LAB * 1.05: 
        UAB = initial_bmr_baseline * pal_for_uab_heuristic * 1.05 
        if inputs.get('streamlit_object'): inputs['streamlit_object'].warning(f"DLW-TDEE for UAB low. UAB set heuristically to {UAB:,.0f} kcal.")

    target_true_intake_for_sim = inputs['adjusted_intake']
    kcal_per_step = KCAL_PER_STEP_BASE_FACTOR * inputs['weight_kg']
    eat_kcal_steps = inputs['avg_daily_steps'] * kcal_per_step
    eat_kcal_additional_exercise = inputs['other_exercise_kcal_per_day']
    eat_kcal_fixed_for_sim = eat_kcal_steps + eat_kcal_additional_exercise

    day0_bmr = initial_bmr_baseline
    day0_tef = calculate_tef(target_true_intake_for_sim, inputs['protein_g_per_day'])
    cold_kcal_fixed = calculate_cold_thermogenesis(inputs['typical_indoor_temp_f'], inputs['minutes_cold_exposure_daily'])
    day0_fever = calculate_immune_fever_effect(inputs['has_fever_illness'], inputs['peak_fever_temp_f'], day0_bmr)
    day0_neat_adaptive = 0.0
    TDEE_sim_start = day0_bmr + day0_tef + eat_kcal_fixed_for_sim + day0_neat_adaptive + cold_kcal_fixed + day0_fever

    current_bmr_adaptive = day0_bmr
    current_neat_adaptive_component = day0_neat_adaptive

    CRITICAL_LOW_INTAKE_THRESHOLD = initial_bmr_baseline * 1.35
    is_critically_low_intake = target_true_intake_for_sim < CRITICAL_LOW_INTAKE_THRESHOLD
    
    min_bmr_limit_factor = 0.80; max_bmr_limit_factor = 1.15
    min_bmr_target_limit = initial_bmr_baseline * min_bmr_limit_factor
    max_bmr_target_limit = initial_bmr_baseline * max_bmr_limit_factor
    min_neat_limit = -450; max_neat_limit = 700

    intake_in_adaptive_range = LAB <= target_true_intake_for_sim <= UAB
    daily_log = []

    for day in range(num_days_to_simulate):
        current_eat_kcal = eat_kcal_fixed_for_sim 
        tef_kcal = calculate_tef(target_true_intake_for_sim, inputs['protein_g_per_day'])
        fever_kcal = calculate_immune_fever_effect(inputs['has_fever_illness'], inputs['peak_fever_temp_f'], current_bmr_adaptive)

        day_target_bmr = current_bmr_adaptive 
        day_target_neat = current_neat_adaptive_component 

        if is_critically_low_intake:
            day_target_bmr = min_bmr_target_limit 
            target_bmr_plus_neat_floor = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
            day_target_neat = target_bmr_plus_neat_floor - day_target_bmr
            day_target_neat = np.clip(day_target_neat, min_neat_limit, 0) 
        elif intake_in_adaptive_range:
            total_adaptation_gap = target_true_intake_for_sim - TDEE_sim_start
            neat_share = 0.60 if total_adaptation_gap > 0 else 0.40
            bmr_share = 1.0 - neat_share
            target_total_neat_change = total_adaptation_gap * neat_share
            target_total_bmr_change = total_adaptation_gap * bmr_share
            day_target_neat = np.clip(day0_neat_adaptive + target_total_neat_change, min_neat_limit, max_neat_limit)
            day_target_bmr = np.clip(day0_bmr + target_total_bmr_change, min_bmr_target_limit, max_bmr_target_limit)
        else: 
            current_expenditure_for_balance = current_bmr_adaptive + tef_kcal + current_eat_kcal + current_neat_adaptive_component + cold_kcal_fixed + fever_kcal
            energy_balance = target_true_intake_for_sim - current_expenditure_for_balance
            bmr_target_change_factor_ext = 0.0
            if energy_balance > 250: bmr_target_change_factor_ext = 0.05 
            elif energy_balance < -250: bmr_target_change_factor_ext = -0.10
            day_target_bmr = initial_bmr_baseline * (1 + bmr_target_change_factor_ext)
            day_target_bmr = np.clip(day_target_bmr, min_bmr_target_limit, max_bmr_target_limit)
            neat_responsiveness = 0.30
            if inputs['avg_daily_steps'] > 10000: neat_responsiveness += 0.10
            if inputs['avg_daily_steps'] < 5000: neat_responsiveness -= 0.10
            if inputs['avg_sleep_hours'] < 6.5: neat_responsiveness -= 0.05
            if inputs['uses_caffeine']: neat_responsiveness += 0.05
            neat_responsiveness = np.clip(neat_responsiveness, 0.1, 0.5)
            day_target_neat = energy_balance * neat_responsiveness
            day_target_neat = np.clip(day_target_neat, min_neat_limit, max_neat_limit)

        front_load_factor = 0.0
        current_tau_bmr, current_tau_neat = TAU_BMR_ADAPTATION, TAU_NEAT_ADAPTATION
        if day == 0:
            if is_critically_low_intake:
                front_load_factor = 0.60; current_tau_bmr *= 0.5; current_tau_neat *= 0.5
            elif intake_in_adaptive_range and (target_true_intake_for_sim - TDEE_sim_start) != 0:
                front_load_factor = 0.40
        
        if front_load_factor > 0:
            bmr_change_to_apply_day0 = (day_target_bmr - day0_bmr) * front_load_factor
            neat_change_to_apply_day0 = (day_target_neat - day0_neat_adaptive) * front_load_factor
            current_bmr_adaptive = day0_bmr + bmr_change_to_apply_day0
            current_neat_adaptive_component = day0_neat_adaptive + neat_change_to_apply_day0
        else:
            delta_bmr = (day_target_bmr - current_bmr_adaptive) / current_tau_bmr
            current_bmr_adaptive += delta_bmr
            delta_neat = (day_target_neat - current_neat_adaptive_component) / current_tau_neat
            current_neat_adaptive_component += delta_neat
        
        # Enforce physiological limits and the BMR*1.2 floor for (BMR+NEAT) in critical low intake
        current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit, max_bmr_target_limit)
        current_neat_adaptive_component = np.clip(current_neat_adaptive_component, min_neat_limit, max_neat_limit)

        if is_critically_low_intake:
            bmr_plus_neat_target_floor = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
            # Ensure BMR is at least its own absolute minimum
            current_bmr_adaptive = max(current_bmr_adaptive, min_bmr_target_limit)
            # Adjust NEAT so that BMR + NEAT >= floor, respecting NEAT's own floor
            required_neat_for_floor = bmr_plus_neat_target_floor - current_bmr_adaptive
            current_neat_adaptive_component = max(required_neat_for_floor, min_neat_limit)
            # If NEAT hit its floor, BMR might need to be slightly higher to maintain the sum, but not below its own min
            if current_neat_adaptive_component == min_neat_limit:
                current_bmr_adaptive = max(bmr_plus_neat_target_floor - min_neat_limit, min_bmr_target_limit)
            current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit, max_bmr_target_limit) # Final BMR clip


        total_tdee_today = current_bmr_adaptive + tef_kcal + current_eat_kcal + current_neat_adaptive_component + cold_kcal_fixed + fever_kcal
        
        daily_log.append({
            "Day": day + 1, "Target Intake": target_true_intake_for_sim,
            "BMR_Adaptive": current_bmr_adaptive, "TEF": tef_kcal, "EAT": current_eat_kcal,
            "NEAT_Adaptive": current_neat_adaptive_component, "Cold_Thermo": cold_kcal_fixed,
            "Fever_Effect": fever_kcal, "Total_Dynamic_TDEE": total_tdee_today,
            "Energy_Balance_vs_TDEE": target_true_intake_for_sim - total_tdee_today,
            "Target_BMR_DailyStep": day_target_bmr, "Target_NEAT_DailyStep": day_target_neat
        })

    final_tdee_val = daily_log[-1]['Total_Dynamic_TDEE'] if daily_log else TDEE_sim_start
    final_states = {
        "final_bmr_adaptive": current_bmr_adaptive,
        "final_neat_adaptive_component": current_neat_adaptive_component,
        "final_tdee": final_tdee_val,
        "initial_bmr_baseline": initial_bmr_baseline,
        "LAB": LAB, "UAB": UAB, "intake_in_adaptive_range": intake_in_adaptive_range,
        "is_critically_low_intake_scenario": is_critically_low_intake
    }
    return pd.DataFrame(daily_log), final_states

# --- generate_bulk_cut_assessment (with refined FMI/FFMI HR logic) ---
def generate_bulk_cut_assessment(
    adjusted_intake, dynamic_tdee,
    initial_bmr_baseline, 
    ffm_kg, fm_kg, height_m, bmi, sex,
    sim_final_states 
    ):
    ffmi, fmi = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
    LAB = sim_final_states.get('LAB', initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR) 
    UAB = sim_final_states.get('UAB', initial_bmr_baseline * 1.7) # Fallback UAB
    intake_was_in_adaptive_range = sim_final_states.get('intake_in_adaptive_range', False)
    is_critical_low_intake_sim = sim_final_states.get('is_critically_low_intake_scenario', False)

    advice_primary = ""; status_caloric = ""
    daily_surplus_deficit_vs_dynamic_tdee = adjusted_intake - dynamic_tdee

    if daily_surplus_deficit_vs_dynamic_tdee > 25: status_caloric = "Surplus"
    elif daily_surplus_deficit_vs_dynamic_tdee < -25: status_caloric = "Deficit"
    else: status_caloric = "Maintenance"
    
    advice_primary = (f"Your intake ({adjusted_intake:,.0f} kcal) resulted in a caloric {status_caloric.lower()} of "
                      f"**{daily_surplus_deficit_vs_dynamic_tdee:+.0f} kcal/day** vs. your simulated dynamic TDEE "
                      f"({dynamic_tdee:,.0f} kcal) after adaptation. This supports "
                      f"{'weight gain' if status_caloric == 'Surplus' else ('weight loss' if status_caloric == 'Deficit' else 'weight maintenance')}.\n")
    
    if is_critical_low_intake_sim:
        advice_primary += (f"Your intake was critically low. The simulation adapted TDEE (BMR+NEAT) towards a floor of "
                           f"~{initial_bmr_baseline*CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR:,.0f} kcal (Initial RMR * {CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR:.1f}) before fixed EAT & TEF.\n")
    elif intake_was_in_adaptive_range:
        advice_primary += (f"Your intake fell within the estimated primary metabolic adaptive range ({LAB:,.0f} - {UAB:,.0f} kcal). "
                           "TDEE largely adapted to this intake.\n")
    else:
        advice_primary += (f"Your intake was outside the primary adaptive range ({LAB:,.0f} - {UAB:,.0f} kcal). "
                           "This intake level is expected to primarily drive tissue change.\n")

    advice_composition = ""; fmi_hr_status_text = ""; ffmi_hr_status_text = ""; ffmi_direct_bulk_triggered = False
    
    # FMI Hazard Ratio Logic based on user's detailed description
    FMI_SWEET_SPOT_LOWER = 4.0; FMI_SWEET_SPOT_UPPER = 8.0 # HR 0.8-1.0
    FMI_MODERATE_RISK_LOWER = 3.0 # Below 4, HR ~1.2-1.4
    FMI_MODERATE_RISK_UPPER_START = 10.0 # HR starts > 1.0
    FMI_HIGH_RISK_1_5 = 12.0
    FMI_HIGH_RISK_2_0 = 15.0
    FMI_HIGH_RISK_3_0 = 20.0

    if fmi < FMI_MODERATE_RISK_LOWER: fmi_hr_status_text = f"Very Low (FMI < {FMI_MODERATE_RISK_LOWER:.0f}, HR ~1.2-1.4, slightly elevated risk)"
    elif fmi <= FMI_SWEET_SPOT_UPPER: fmi_hr_status_text = f"Optimal (FMI {FMI_SWEET_SPOT_LOWER:.0f}-{FMI_SWEET_SPOT_UPPER:.0f}, HR ~0.8-1.0, minimal risk)"
    elif fmi < FMI_HIGH_RISK_1_5: fmi_hr_status_text = f"Elevated (FMI > {FMI_SWEET_SPOT_UPPER:.0f}, HR > 1.0, increasing risk)"
    elif fmi < FMI_HIGH_RISK_2_0: fmi_hr_status_text = f"High (FMI ~{FMI_HIGH_RISK_1_5:.0f}, HR ~1.5, significantly increased risk)"
    elif fmi < FMI_HIGH_RISK_3_0: fmi_hr_status_text = f"Very High (FMI ~{FMI_HIGH_RISK_2_0:.0f}, HR ~2.0, high risk)"
    else: fmi_hr_status_text = f"Extremely High (FMI >= {FMI_HIGH_RISK_3_0:.0f}, HR > 3.0, very high risk)"
    advice_composition += f"- FMI: {fmi:.1f} kg/m² - Status: {fmi_hr_status_text}\n"

    # FFMI Hazard Ratio Logic
    FFMI_DIRECT_BULK_THRESHOLD = 20.0 # User's primary threshold
    FFMI_HIGH_RISK_THRESHOLD = 15.0 # HR 2+ below this
    FFMI_MODERATE_RISK_THRESHOLD = 18.0 # HR still > 1 below this but improving
    FFMI_SWEET_SPOT_LOWER = 18.0; FFMI_SWEET_SPOT_UPPER = 22.0 # HR 0.8-1.0
    FFMI_SLIGHT_INCREASE_THRESHOLD = 24.0 # HR might creep up past this

    if ffmi < FFMI_DIRECT_BULK_THRESHOLD: ffmi_direct_bulk_triggered = True
    
    if ffmi < FFMI_HIGH_RISK_THRESHOLD: ffmi_hr_status_text = f"Very Low (FFMI < {FFMI_HIGH_RISK_THRESHOLD:.0f}, HR ~2+, high risk due to frailty)"
    elif ffmi < FFMI_MODERATE_RISK_THRESHOLD: ffmi_hr_status_text = f"Low (FFMI < {FFMI_MODERATE_RISK_THRESHOLD:.0f}, HR > 1, elevated risk)"
    elif ffmi <= FFMI_SWEET_SPOT_UPPER: ffmi_hr_status_text = f"Optimal (FFMI {FFMI_SWEET_SPOT_LOWER:.0f}-{FFMI_SWEET_SPOT_UPPER:.0f}, HR ~0.8-1.0, protective)"
    elif ffmi <= FFMI_SLIGHT_INCREASE_THRESHOLD: ffmi_hr_status_text = f"High/Sufficient (FFMI > {FFMI_SWEET_SPOT_UPPER:.0f}, HR ~1.0, no further longevity benefit)"
    else: ffmi_hr_status_text = f"Very High (FFMI > {FFMI_SLIGHT_INCREASE_THRESHOLD:.0f}, HR may slightly increase, monitor)"
    advice_composition += f"- FFMI: {ffmi:.1f} kg/m² - Status: {ffmi_hr_status_text}\n"

    final_recommendation = ""; overall_status_message = f"Caloric: {status_caloric} | FMI Risk: {fmi_hr_status_text} | FFMI Risk: {ffmi_hr_status_text}"

    # Dominance logic: High FMI risk prompts cut, Low FFMI risk prompts bulk.
    fmi_suggests_cut = fmi > FMI_SWEET_SPOT_UPPER # Simplified: if FMI is above sweet spot, cutting is generally better for HR
    ffmi_suggests_bulk_hr = ffmi < FFMI_MODERATE_RISK_THRESHOLD # If FFMI is in a >1 HR zone

    if ffmi_direct_bulk_triggered: # User rule: FFMI < 20
        if fmi_suggests_cut and fmi > FMI_HIGH_RISK_HR_THRESHOLD: # If FMI is truly high risk
            final_recommendation = "REC: Complex - Body Recomp/Careful Lean Bulk. FFMI is below 20, but FMI is also high risk. Prioritize resistance training. Slight surplus or maintenance. Consult a professional."
        else:
            final_recommendation = f"REC: BULK. Your FFMI ({ffmi:.1f}) is below the {FFMI_DIRECT_BULK_THRESHOLD:.0f} kg/m² target. Focus on a caloric surplus to build lean mass."
            if status_caloric == "Deficit": final_recommendation += " Current state is deficit; increase intake."
            elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; a surplus is needed."
    elif fmi_suggests_cut and fmi > FMI_HIGH_RISK_HR_THRESHOLD: # High FMI risk takes precedence if FFMI is not < 20
        final_recommendation = f"REC: CUT. Your FMI ({fmi:.1f}) is in a high-risk zone. Focus on a caloric deficit."
        if status_caloric == "Surplus": final_recommendation += " Current state is surplus; decrease intake."
        elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; a deficit is needed."
    elif ffmi_suggests_bulk_hr: # FFMI is not < 20, but still in a >1 HR zone
        final_recommendation = f"REC: Consider Lean Bulk. Your FFMI ({ffmi:.1f}) is in a zone with elevated health risk (HR > 1). A caloric surplus for lean mass could be beneficial."
        if status_caloric == "Deficit": final_recommendation += " Current state is deficit; consider adjusting to maintenance or slight surplus."
        elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; a slight surplus might be considered if FMI allows."
    # Default to caloric status if no strong HR-based driver from above
    elif status_caloric == "Surplus": final_recommendation = "REC: SURPLUS. If bulking, this aligns. Monitor body composition."
    elif status_caloric == "Deficit": final_recommendation = "REC: DEFICIT. If cutting, this aligns. Ensure adequate protein and training."
    elif status_caloric == "Maintenance":
        if "Optimal" in fmi_hr_status_text and ("Optimal" in ffmi_hr_status_text or "Sufficient/High" in ffmi_hr_status_text) :
             final_recommendation = "REC: MAINTAIN or Optimize. Your body composition appears to be in a healthy range. Adjust for specific goals."
        else: final_recommendation = "REC: MAINTAIN & RE-EVALUATE. Intake supports maintenance. Review FMI/FFMI status for next phase."
    else: final_recommendation = "Review goals with FMI/FFMI context. Adjust intake."

    full_advice = (f"{advice_primary}\n\n**Body Composition Insights (FMI/FFMI & Hazard Ratio Interpretation):**\n{advice_composition}\n**Overall Strategy Guidance:**\n{final_recommendation}")
    return full_advice, overall_status_message, daily_surplus_deficit_vs_dynamic_tdee

# --- project_weight_change_scenarios (no changes needed) ---
def project_weight_change_scenarios(current_dynamic_tdee, weight_kg):
    scenarios = []
    kcal_per_lb_wk_as_daily = 3500.0 / 7.0
    intake_targets_desc = {
        "Aggressive Cut (~ -1.0 lbs/wk)": current_dynamic_tdee - (1.0 * kcal_per_lb_wk_as_daily),
        "Moderate Cut (~ -0.5 lbs/wk)": current_dynamic_tdee - (0.5 * kcal_per_lb_wk_as_daily),
        "Maintenance (at current Dynamic TDEE)": current_dynamic_tdee,
        "Lean Bulk (~ +0.25 lbs/wk)": current_dynamic_tdee + (0.25 * kcal_per_lb_wk_as_daily),
        "Moderate Bulk (~ +0.5 lbs/wk)": current_dynamic_tdee + (0.5 * kcal_per_lb_wk_as_daily),
    }
    for desc, intake_kcal in intake_targets_desc.items():
        daily_delta_vs_tdee = intake_kcal - current_dynamic_tdee
        eff_factor = SURPLUS_EFFICIENCY if daily_delta_vs_tdee > 0 else (DEFICIT_EFFICIENCY if daily_delta_vs_tdee < 0 else 1.0)
        effective_daily_stored_kcal = daily_delta_vs_tdee * eff_factor
        weekly_kg_change = (effective_daily_stored_kcal * 7) / KCAL_PER_KG_TISSUE
        weekly_lbs_change = kg_to_lbs(weekly_kg_change)
        scenarios.append({
            "Scenario": desc,
            "Target Daily Intake (kcal)": f"{intake_kcal:,.0f}",
            "Est. Effective Daily Surplus/Deficit (kcal)": f"{effective_daily_stored_kcal:+.0f}",
            "Predicted Weekly Weight Change": f"{weekly_lbs_change:+.2f} lbs ({weekly_kg_change:+.3f} kg)"
        })
    return pd.DataFrame(scenarios)

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Advanced TDEE & Metabolic Modeler")

INFO_ICON = "❓"
TOOLTIPS = {
    "EAT": "Exercise Activity Thermogenesis: Calories burned from deliberate, structured physical exercise (e.g., steps, dedicated cardio, resistance training).",
    "NEAT": "Non-Exercise Activity Thermogenesis: Calories burned from all other physical activities excluding sleep, eating, and formal EAT (e.g., fidgeting, posture, spontaneous movements). This model component adapts to energy balance and other factors.",
    "TEF": "Thermic Effect of Food: Calories burned during the digestion, absorption, and processing of food. Varies by macronutrient composition.",
    "LAB": f"Lower Adaptive Bound: The lower end of TDEE (approx. Initial RMR * {CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR:.1f} before deliberate exercise) where your body strongly resists further TDEE reduction and primarily loses tissue mass.",
    "UAB": "Upper Adaptive Bound: The upper end of TDEE (estimated by FFM-based DLW equations) where your body can adapt metabolism to match intake. Intake above this more strongly drives tissue gain.",
    "FMI": "Fat Mass Index: Fat Mass (kg) / Height (m)^2. An indicator of relative body fatness.",
    "FFMI": "Fat-Free Mass Index: Fat-Free Mass (kg) / Height (m)^2. An indicator of relative muscularity.",
    "other_exercise_kcal": "Estimate daily calories burned from exercises NOT well captured by general daily step counts (e.g., cycling, swimming, intense weightlifting). Use other calculators or fitness trackers for this estimation. This is added directly to step-based EAT.",
    "weight_change_rate": "Your average weekly weight change (e.g., -0.5 if losing 0.5 units/week, +0.25 if gaining 0.25 units/week) over the past 2-4 stable weeks. This helps calibrate your true current caloric intake from your reported intake."
}

def init_session_state():
    defaults = {
        "weight_unit": "lbs", "height_unit": "ft/in",
        "weight_input_val": 150.0, 
        "body_fat_percentage": 15.0, "sex": "Male", "age_years": 25,
        "feet": 5, "inches": 10, "height_cm_input": 178.0,
        "avg_daily_steps": 7500, "other_exercise_kcal_per_day": 0,
        "avg_daily_kcal_intake_reported": 2500, "protein_g_per_day": 150.0,
        "weight_trend": "Steady", 
        "weight_change_rate_input_val_lbs": 0.5, 
        "weight_change_rate_input_val_kg": 0.23,
        "typical_indoor_temp_f": 70, "minutes_cold_exposure_daily": 0,
        "avg_sleep_hours": 7.5, "uses_caffeine": True, "has_fever_illness": False,
        "peak_fever_temp_f_input": 98.6, "num_days_to_simulate": 14,
        "weight_input_val_initialized" : False
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if not st.session_state.weight_input_val_initialized:
        if st.session_state.weight_unit == "kg": st.session_state.weight_input_val = 68.0
        else: st.session_state.weight_input_val = 150.0
        st.session_state.weight_input_val_initialized = True
init_session_state()

def display_sidebar_inputs():
    st.sidebar.header("📝 User Inputs")
    st.sidebar.markdown("Inputs are saved for your current browser session.")
    unit_cols = st.sidebar.columns(2)
    unit_cols[0].radio("Weight unit:", ("kg", "lbs"), key="weight_unit", 
                       index=(["kg", "lbs"].index(st.session_state.weight_unit) if st.session_state.weight_unit in ["kg", "lbs"] else 0))
    unit_cols[1].radio("Height unit:", ("cm", "ft/in"), key="height_unit",
                       index=(["cm","ft/in"].index(st.session_state.height_unit) if st.session_state.height_unit in ["cm","ft/in"] else 0))
    st.sidebar.subheader("👤 Body & Demographics")
    st.sidebar.number_input(f"Current Body Weight ({st.session_state.weight_unit}):", 
                            min_value=(50.0 if st.session_state.weight_unit == "lbs" else 20.0), 
                            max_value=(700.0 if st.session_state.weight_unit == "lbs" else 300.0), 
                            step=0.1, format="%.1f", key="weight_input_val")
    st.sidebar.slider("Estimated Body Fat Percentage (%):", min_value=3.0, max_value=60.0, step=0.5, format="%.1f", key="body_fat_percentage")
    st.sidebar.selectbox("Sex:", ("Male", "Female"), key="sex")
    st.sidebar.number_input("Age (years):", min_value=13, max_value=100, step=1, key="age_years")
    if st.session_state.height_unit == "ft/in":
        h_col1, h_col2 = st.sidebar.columns(2)
        h_col1.number_input("Height (feet):", min_value=3, max_value=8, step=1, key="feet")
        h_col2.number_input("Height (inches):", min_value=0, max_value=11, step=1, key="inches")
    else: 
        st.sidebar.number_input("Height (cm):", min_value=100.0, max_value=250.0, step=0.1, format="%.1f", key="height_cm_input")
    st.sidebar.subheader(f"🏃‍♀️ Activity Profile (Step-Based) {INFO_ICON}", help="Define your typical daily physical activity.")
    st.sidebar.number_input("Average Total Daily Steps:", min_value=0, max_value=50000, step=100, key="avg_daily_steps", help="Your typical daily step count from a pedometer or fitness tracker. This is the main input for step-based EAT.")
    st.sidebar.number_input("Other Daily Exercise (non-step, kcal):",min_value=0, max_value=2000, step=25, key="other_exercise_kcal_per_day", help=TOOLTIPS["other_exercise_kcal"])
    st.sidebar.subheader(f"🍽️ Diet (Target or Current Average) {INFO_ICON}", help="Your average daily food intake.")
    st.sidebar.number_input("Reported Average Daily Caloric Intake (kcal):", min_value=500, max_value=10000, step=50, key="avg_daily_kcal_intake_reported")
    st.sidebar.number_input("Protein Intake (grams per day):", min_value=0.0, max_value=500.0, step=1.0, format="%.1f", key="protein_g_per_day")
    st.sidebar.subheader(f"⚖️ Observed Weight Trend {INFO_ICON}", help=TOOLTIPS["weight_change_rate"])
    st.sidebar.caption("Used to calibrate true intake from reported intake.")
    st.sidebar.selectbox("Recent Body Weight Trend:", ("Steady", "Gaining", "Losing"), key="weight_trend")
    rate_help_text = "Average weekly change over last 2-4 weeks. E.g., 0.5 for gaining 0.5 units/wk, or 0.25 for losing 0.25 units/wk."
    if st.session_state.weight_trend != "Steady":
        if st.session_state.weight_unit == "lbs":
            st.sidebar.number_input(f"Rate of {st.session_state.weight_trend.lower()} (lbs/week):", min_value=0.01, max_value=5.0, step=0.05, format="%.2f", key="weight_change_rate_input_val_lbs", help=rate_help_text)
        else: 
             st.sidebar.number_input(f"Rate of {st.session_state.weight_trend.lower()} (kg/week):", min_value=0.01, max_value=2.5, step=0.01, format="%.2f", key="weight_change_rate_input_val_kg", help=rate_help_text)
    st.sidebar.subheader("🌡️ Environment & Physiology")
    st.sidebar.slider("Typical Indoor Temperature (°F):", min_value=50, max_value=90, step=1, key="typical_indoor_temp_f")
    st.sidebar.number_input("Avg. Min/Day Outdoors <60°F/15°C:", min_value=0, max_value=1440, step=15, key="minutes_cold_exposure_daily")
    st.sidebar.slider("Habitual Nightly Sleep (hours):", min_value=4.0, max_value=12.0, step=0.1, format="%.1f", key="avg_sleep_hours")
    st.sidebar.checkbox("Regular Caffeine User?", key="uses_caffeine")
    st.sidebar.checkbox("Current Fever or Acute Illness?", key="has_fever_illness")
    if st.session_state.has_fever_illness:
        st.sidebar.number_input("Peak Fever Temp (°F):", min_value=98.6, max_value=106.0, step=0.1, format="%.1f", key="peak_fever_temp_f_input")
    st.sidebar.slider("Simulation Duration (days for TDEE graph):", 7, 90, 7, key="num_days_to_simulate")
display_sidebar_inputs()

st.title("💪 Advanced Dynamic TDEE & Metabolic Modeler ⚙️")
st.markdown("""...""") # Main intro markdown
st.header("📊 Results & Analysis")

if st.sidebar.button("🚀 Calculate & Simulate TDEE", type="primary", use_container_width=True):
    s_weight_unit = st.session_state.weight_unit
    s_height_unit = st.session_state.height_unit
    s_weight_input_val = st.session_state.weight_input_val 
    if s_weight_unit == "lbs": s_weight_kg = lbs_to_kg(s_weight_input_val)
    else: s_weight_kg = s_weight_input_val
    s_body_fat_percentage = st.session_state.body_fat_percentage
    s_sex = st.session_state.sex
    s_age_years = st.session_state.age_years
    if s_height_unit == "ft/in": s_height_cm = ft_in_to_cm(st.session_state.feet, st.session_state.inches)
    else: s_height_cm = st.session_state.height_cm_input
    s_avg_daily_steps = st.session_state.avg_daily_steps
    s_other_exercise_kcal_per_day = st.session_state.other_exercise_kcal_per_day
    s_avg_daily_kcal_intake_reported = st.session_state.avg_daily_kcal_intake_reported
    s_protein_g_per_day = st.session_state.protein_g_per_day
    s_weight_trend = st.session_state.weight_trend
    s_weight_change_rate_display_val = 0.0 
    s_weight_change_rate_kg_wk = 0.0
    if s_weight_trend != "Steady":
        if s_weight_unit == "lbs":
            s_weight_change_rate_display_val = st.session_state.get("weight_change_rate_input_val_lbs",0.0)
            s_weight_change_rate_kg_wk = lbs_to_kg(s_weight_change_rate_display_val)
        else: 
            s_weight_change_rate_display_val = st.session_state.get("weight_change_rate_input_val_kg",0.0)
            s_weight_change_rate_kg_wk = s_weight_change_rate_display_val
    s_typical_indoor_temp_f = st.session_state.typical_indoor_temp_f
    s_minutes_cold_exposure_daily = st.session_state.minutes_cold_exposure_daily
    s_avg_sleep_hours = st.session_state.avg_sleep_hours
    s_uses_caffeine = st.session_state.uses_caffeine
    s_has_fever_illness = st.session_state.has_fever_illness
    s_peak_fever_temp_f_input = st.session_state.get("peak_fever_temp_f_input", 98.6) if s_has_fever_illness else 98.6
    s_num_days_to_simulate = st.session_state.num_days_to_simulate

    if s_height_cm <= 0:
        st.error("Height must be a positive value. Please check your inputs.")
    else:
        ffm_kg, fm_kg = calculate_ffm_fm(s_weight_kg, s_body_fat_percentage)
        height_m = s_height_cm / 100.0
        bmi = s_weight_kg / (height_m**2) if height_m > 0 else 0
        ffmi_check, fmi_check = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
        if (s_sex == "Male" and ffmi_check > 35) or (s_sex == "Female" and ffmi_check > 30) or \
           (s_sex == "Male" and s_body_fat_percentage < 3.0 and s_weight_kg > 50) or \
           (s_sex == "Female" and s_body_fat_percentage < 10.0 and s_weight_kg > 40) or \
           s_body_fat_percentage > 60 or s_body_fat_percentage < 2.0 :
            st.warning("⚠️ Your input body composition may be physiologically extreme. Results may be less reliable.")

        st.subheader("📋 Initial Body Composition & Indices")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Weight", f"{s_weight_kg:.1f} kg / {kg_to_lbs(s_weight_kg):.1f} lbs")
        res_col2.metric("FFM", f"{ffm_kg:.1f} kg"); res_col3.metric("FM", f"{fm_kg:.1f} kg ({s_body_fat_percentage:.1f}%)"); res_col4.metric("BMI", f"{bmi:.1f}")
        current_ffmi, current_fmi = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
        idx_col1, idx_col2 = st.columns(2)
        idx_col1.metric("FFMI", f"{current_ffmi:.1f} kg/m²", help=TOOLTIPS["FFMI"]); idx_col2.metric("FMI", f"{current_fmi:.1f} kg/m²", help=TOOLTIPS["FMI"])
        
        initial_bmr_ref = calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg)
        if initial_bmr_ref == 0: initial_bmr_ref = calculate_mifflin_st_jeor_rmr(s_weight_kg, s_height_cm, s_age_years, s_sex)
        
        lower_bound_tdee_static_display = initial_bmr_ref * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
        upper_bound_tdee_static_display_calc = calculate_dlw_tdee(ffm_kg, fm_kg)
        pal_for_UAB_display_heuristic = get_pal_multiplier_for_heuristic(s_avg_daily_steps)
        if upper_bound_tdee_static_display_calc <= lower_bound_tdee_static_display :
            upper_bound_tdee_static_display = initial_bmr_ref * pal_for_UAB_display_heuristic * 1.05
            if upper_bound_tdee_static_display_calc == 0: st.warning("FFM-based TDEE (DLW) was zero. Using heuristic for static upper bound.")
        else: upper_bound_tdee_static_display = upper_bound_tdee_static_display_calc
        
        adjusted_true_intake = adjust_reported_intake(s_avg_daily_kcal_intake_reported, s_weight_trend, s_weight_change_rate_kg_wk)
        if adjusted_true_intake < initial_bmr_ref * 0.9: st.warning(f"⚠️ Calibrated intake ({adjusted_true_intake:,.0f} kcal) is very low vs RMR ({initial_bmr_ref:,.0f} kcal).")
        elif upper_bound_tdee_static_display > 0 and adjusted_true_intake > upper_bound_tdee_static_display * 1.75 : st.warning(f"⚠️ Calibrated intake ({adjusted_true_intake:,.0f} kcal) is very high vs upper TDEE ({upper_bound_tdee_static_display:,.0f} kcal).")

        with st.expander("Advanced Metabolic Insights & Benchmarks", expanded=False):
            st.markdown(f"#### ↔️ Estimated Static Metabolic Range {INFO_ICON}", help="General reference for TDEE boundaries. Simulation models dynamic adaptations within a similar range.")
            st.markdown(f"""
            - **Static Lower Adaptive Bound (approx. RMR + minimal NEAT): `{lower_bound_tdee_static_display:,.0f} kcal/day`** (Initial RMR * {CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR:.1f}). <span title='{TOOLTIPS["LAB"]}'>{INFO_ICON}</span>
            - **Static Upper Adaptive Bound (Typical Free-living TDEE): `{upper_bound_tdee_static_display:,.0f} kcal/day`** (FFM-based DLW formula). <span title='{TOOLTIPS["UAB"]}'>{INFO_ICON}</span>
            """, unsafe_allow_html=True)
            
            energy_non_loco, energy_loco, implied_steps, _ = calculate_implied_activity_breakdown(upper_bound_tdee_static_display, initial_bmr_ref, s_weight_kg)
            st.markdown(f"#### 🚶‍♂️Implied Activity for FFM-Based TDEE (DLW) of `{upper_bound_tdee_static_display:,.0f}` kcal")
            st.markdown(f"""
            This TDEE for your body composition inherently includes typical activity. Breakdown:
            - RMR (Pontzer FFM or fallback): `{initial_bmr_ref:,.0f} kcal/day`
            - TEF (~10% at this TDEE): `{upper_bound_tdee_static_display * 0.10:,.0f} kcal/day` <span title='{TOOLTIPS["TEF"]}'>{INFO_ICON}</span>
            - Non-Locomotor Upregulation & Base NEAT (to reach ~RMR*{SEDENTARY_WELLFED_RMR_MULTIPLIER:.1f} before significant steps): **`{energy_non_loco:,.0f} kcal/day`** <span title='{TOOLTIPS["NEAT"]}'>{INFO_ICON}</span>
            - Remaining Energy for Deliberate Locomotion (steps/cardio beyond RMR*{SEDENTARY_WELLFED_RMR_MULTIPLIER:.1f}): **`{energy_loco:,.0f} kcal/day`** <span title='{TOOLTIPS["EAT"]}'>{INFO_ICON}</span>
            - This locomotor portion is roughly equivalent to: **`{implied_steps:,.0f} steps/day`**.
            
            Compare this to your input steps (`{s_avg_daily_steps:,.0f}`) & other exercise (`{s_other_exercise_kcal_per_day:,.0f} kcal`).
            """, unsafe_allow_html=True)

        st.metric("Reported Avg. Daily Intake", f"{s_avg_daily_kcal_intake_reported:,.0f} kcal")
        if abs(adjusted_true_intake - s_avg_daily_kcal_intake_reported) > 20:
            unit_for_trend_cap = s_weight_unit 
            display_rate_val_for_cap = s_weight_change_rate_display_val if s_weight_trend != "Steady" else 0.0
            if s_weight_trend != "Steady" :
                 st.metric("Calibrated True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal", help="Estimated from reported intake & weight trend; used as target for simulation.")
                 # Ensure display_rate_val_for_cap uses the correct unit based on s_weight_unit
                 if s_weight_unit == "kg":
                     rate_to_display = st.session_state.get("weight_change_rate_input_val_kg",0.0)
                 else: #lbs
                     rate_to_display = st.session_state.get("weight_change_rate_input_val_lbs",0.0)
                 st.caption(f"Adjusted based on reported weight trend of {rate_to_display:.2f} {unit_for_trend_cap}/week ({s_weight_trend}).")
            else: st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")
        else: st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")

        simulation_inputs = {
            "weight_kg": s_weight_kg, "body_fat_percentage": s_body_fat_percentage,
            "sex": s_sex, "age_years": s_age_years, "height_cm": s_height_cm,
            "avg_daily_steps": s_avg_daily_steps, 
            "other_exercise_kcal_per_day": s_other_exercise_kcal_per_day,
            "adjusted_intake": adjusted_true_intake,
            "protein_g_per_day": s_protein_g_per_day,
            "typical_indoor_temp_f": s_typical_indoor_temp_f,
            "minutes_cold_exposure_daily": s_minutes_cold_exposure_daily,
            "avg_sleep_hours": s_avg_sleep_hours, "uses_caffeine": s_uses_caffeine,
            "has_fever_illness": s_has_fever_illness, "peak_fever_temp_f": s_peak_fever_temp_f_input,
            "streamlit_object": st 
        }

        st.subheader(f"⏳ Simulated TDEE Adaptation Over {s_num_days_to_simulate} Days")
        st.caption(f"Based on maintaining a calibrated true intake of **{adjusted_true_intake:,.0f} kcal/day**.")
        
        daily_tdee_log_df, final_tdee_states = simulate_tdee_adaptation(simulation_inputs, s_num_days_to_simulate)

        if not daily_tdee_log_df.empty:
            current_dynamic_tdee = final_tdee_states['final_tdee']
            LAB_sim = final_tdee_states['LAB']; UAB_sim = final_tdee_states['UAB']
            crit_low_active = final_tdee_states.get('is_critically_low_intake_scenario', False)
            crit_low_msg = f"<span style='color:orange; font-weight:bold;'> (Critically Low Intake Mode {'Activated' if crit_low_active else 'Inactive'})</span>"
            st.metric(f"Simulated Dynamic TDEE (at Day {s_num_days_to_simulate})", f"{current_dynamic_tdee:,.0f} kcal/day")
            st.markdown(f"Primary Metabolic Adaptive Range used: `{LAB_sim:,.0f} - {UAB_sim:,.0f} kcal/day`. {crit_low_msg}", unsafe_allow_html=True)

            with st.expander("Show Detailed Daily Simulation Log & Component Breakdown", expanded=False):
                cols_to_display = [col for col in ["Day", "Target Intake", "BMR_Adaptive", "TEF", "EAT", "NEAT_Adaptive", "Cold_Thermo", "Fever_Effect", "Total_Dynamic_TDEE", "Energy_Balance_vs_TDEE", "Target_BMR_DailyStep", "Target_NEAT_DailyStep"] if col in daily_tdee_log_df.columns]
                st.dataframe(daily_tdee_log_df[cols_to_display].style.format("{:,.0f}", na_rep="-", subset=pd.IndexSlice[:, [c for c in cols_to_display if c != 'Day']]))
            
            chart_cols_to_plot = ['Total_Dynamic_TDEE', 'BMR_Adaptive', 'NEAT_Adaptive', 'TEF', 'EAT']
            valid_chart_cols = [col for col in chart_cols_to_plot if col in daily_tdee_log_df.columns and daily_tdee_log_df[col].abs().sum() > 0.01]
            if 'Day' in daily_tdee_log_df.columns and len(valid_chart_cols) > 0:
                st.line_chart(daily_tdee_log_df[['Day'] + valid_chart_cols].set_index('Day'))
                st.caption("Chart shows key TDEE components adapting over the simulation period.")
        else:
            st.error("Simulation failed to produce results."); current_dynamic_tdee = (lower_bound_tdee_static_display + upper_bound_tdee_static_display) / 2.0

        st.subheader(f"🎯 Nutritional Strategy Assessment {INFO_ICON}", help="Guidance based on simulated TDEE, body composition, and FMI/FFMI risk profiles.")
        advice, overall_status_msg, _ = generate_bulk_cut_assessment(
            adjusted_true_intake, current_dynamic_tdee,
            final_tdee_states.get('initial_bmr_baseline', initial_bmr_ref), 
            ffm_kg, fm_kg, height_m, bmi, s_sex, final_tdee_states
        )
        
        if "SURPLUS" in overall_status_msg.upper() and "CUT" not in advice.upper(): st.success(f"{overall_status_msg}")
        elif "DEFICIT" in overall_status_msg.upper() and "BULK" not in advice.upper(): st.error(f"{overall_status_msg}")
        elif "COMPLEX" in advice.upper() or "CONSULT" in advice.upper() or "CAREFUL" in advice.upper() : st.warning(f"{overall_status_msg}")
        else: st.info(f"{overall_status_msg}")
        st.markdown(advice)

        st.subheader(f"📅 Future Intake Scenarios & Projected Weight Change {INFO_ICON}", help="Estimates based on maintaining different intake levels after your body has adapted as per the simulation.")
        st.caption(f"Based on simulated dynamic TDEE of **{current_dynamic_tdee:,.0f} kcal/day** (after {s_num_days_to_simulate}-day adaptation). 'Est. Effective Surplus/Deficit' accounts for metabolic efficiencies (e.g., ~85% of surplus contributes to tissue gain).")
        df_weight_scenarios = project_weight_change_scenarios(current_dynamic_tdee, s_weight_kg)
        st.dataframe(df_weight_scenarios, hide_index=True)
        
        st.success("✅ Analysis Complete!")
        st.info("Note: Metabolic adaptation is complex. This model provides estimates. Real-world results can vary, and plateaus are common with prolonged dietary changes.")

else:
    st.info("👈 Please fill in your details in the sidebar and click 'Calculate & Simulate TDEE'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Model based on user-provided research & insights.")
st.markdown("---")
st.markdown(f"""
    **Disclaimer & Model Limitations:** This tool provides estimates based on scientific literature and mathematical modeling.
    Individual metabolic responses vary. Results are for informational/educational purposes only, not medical/nutritional advice.
    Consult qualified professionals before changing diet/exercise. Accuracy depends on inputs and model simplifications.
    Metabolic adaptation is dynamic; plateaus can occur. FMI/FFMI HR interpretations are visual approximations from a chart.
    "Implied Activity" is a theoretical estimate based on population averages in FFM-based TDEE formulas.

    **Glossary of Terms:**
    - **TDEE:** <span title="Total Daily Energy Expenditure: Total calories your body burns in a day.">{INFO_ICON}</span>
    - **RMR:** <span title="Resting Metabolic Rate: Calories burned at complete rest. Pontzer FFM-RMR is used as a base for BMR_Adaptive here.">{INFO_ICON}</span>
    - **BMR_Adaptive:** <span title="The component of TDEE representing RMR that adapts over time in response to energy balance and other factors.">{INFO_ICON}</span>
    - **EAT:** <span title='{TOOLTIPS["EAT"]}'>{INFO_ICON}</span>
    - **NEAT_Adaptive:** <span title='{TOOLTIPS["NEAT"]}'>{INFO_ICON}</span>
    - **TEF:** <span title='{TOOLTIPS["TEF"]}'>{INFO_ICON}</span>
    - **LAB (Lower Adaptive Bound):** <span title='{TOOLTIPS["LAB"]}'>{INFO_ICON}</span>
    - **UAB (Upper Adaptive Bound):** <span title='{TOOLTIPS["UAB"]}'>{INFO_ICON}</span>
    - **FFMI (Fat-Free Mass Index):** <span title='{TOOLTIPS["FFMI"]}'>{INFO_ICON}</span>
    - **FMI (Fat Mass Index):** <span title='{TOOLTIPS["FMI"]}'>{INFO_ICON}</span>
    """, unsafe_allow_html=True)

