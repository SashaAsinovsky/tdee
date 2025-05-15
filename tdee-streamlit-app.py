import streamlit as st
import numpy as np
import pandas as pd
# import json # No longer needed for profile saving
# from io import StringIO # No longer needed for profile saving

# --- Constants and Coefficients ---
KCAL_PER_KG_TISSUE = 7700  # Approx. kcal per kg of body tissue change
SURPLUS_EFFICIENCY = 0.85  # % of surplus calories stored (15% lost to adaptive thermogenesis)
DEFICIT_EFFICIENCY = 1.00  # % of deficit calories leading to tissue loss

TAU_BMR_ADAPTATION = 10.0 
TAU_NEAT_ADAPTATION = 2.5 # Slightly increased for more gradual general adaptation

KCAL_PER_STEP_BASE_FACTOR = 0.00062 # kcal per step per kg of bodyweight

SEDENTARY_WELLFED_RMR_MULTIPLIER = 1.6 # For "non-locomotor NEAT & upregulation" implies TDEE = RMR * 1.6 for sedentary well-fed
CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR = 1.2 # BMR*1.2 for (BMR_adaptive + NEAT_adaptive) at very low intake

# Glycogen related constants
GLYCOGEN_G_PER_KG_FFM_MUSCLE = 15.0 # g of glycogen per kg of FFM for muscle stores
LIVER_GLYCOGEN_CAPACITY_G = 100.0 # g for liver
WATER_G_PER_G_GLYCOGEN = 3.5 # g of water stored with each g of glycogen
KCAL_PER_G_CARB_FOR_GLYCOGEN = 4.0 # Approx. kcal to store 1g of glycogen from carbs

# Muscle gain rates (lbs/month) - will be converted to kg/day
MUSCLE_GAIN_NOVICE_LBS_MONTH = 2.0
MUSCLE_GAIN_INTERMEDIATE_LBS_MONTH = 1.0
MUSCLE_GAIN_ADVANCED_LBS_MONTH = 0.5

# --- Helper Functions ---
def kg_to_lbs(kg): return kg * 2.20462
def lbs_to_kg(lbs): return lbs / 2.20462
def ft_in_to_cm(ft, inch): return (ft * 30.48) + (inch * 2.54)

# --- Core Calculation Functions (Stage 1 & General) ---
def calculate_ffm_fm(weight_kg, body_fat_percentage):
    """Calculates Fat-Free Mass (FFM) and Fat Mass (FM) in kg."""
    if weight_kg is None or body_fat_percentage is None: return 0,0
    fm_kg = weight_kg * (body_fat_percentage / 100.0)
    ffm_kg = weight_kg - fm_kg
    return ffm_kg, fm_kg

def calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg):
    """
    Calculates RMR using Pontzer's FFM-based equation:
    exp(-0.954 + 0.707 * Ln(FFM in kg) + 0.019 * Ln(FM in kg)) * 238.853 (to convert MJ to kcal)
    """
    if ffm_kg is None or fm_kg is None or ffm_kg <= 0: return 0
    try:
        fm_kg_adjusted_for_log = fm_kg if fm_kg > 0.001 else 0.001
        
        term_ffm = 0.707 * np.log(ffm_kg)
        term_fm = 0.019 * np.log(fm_kg_adjusted_for_log)
        
        rmr_mj_day = np.exp(-0.954 + term_ffm + term_fm) # Result in MJ/day
        rmr_kcal_day = rmr_mj_day * 238.8529999 # Conversion factor from MJ to kcal
        return rmr_kcal_day
    except Exception:
        return 0

def calculate_mifflin_st_jeor_rmr(weight_kg, height_cm, age_years, sex): # Kept for fallback
    """Calculates RMR using Mifflin-St Jeor equation."""
    if weight_kg is None or height_cm is None or age_years is None or sex is None: return 0
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
    if ffm_kg is None or fm_kg is None or ffm_kg <= 0: return 0
    try:
        fm_kg_adjusted_for_log = fm_kg if fm_kg > 0.001 else 0.001

        term_ffm = 0.916 * np.log(ffm_kg)
        term_fm = -0.030 * np.log(fm_kg_adjusted_for_log) 
        
        tdee_mj_day = np.exp(-1.102 + term_ffm + term_fm)
        tdee_kcal_day = tdee_mj_day * 238.83 
        return tdee_kcal_day
    except Exception:
        return 0

def get_pal_multiplier_for_heuristic(activity_steps): # Used for UAB fallback
    """Maps step counts to a Physical Activity Level multiplier category."""
    if activity_steps is None: activity_steps = 7500 # Default if None
    if activity_steps < 5000: return 1.3 
    elif activity_steps < 7500: return 1.45
    elif activity_steps < 10000: return 1.6
    elif activity_steps < 12500: return 1.75
    else: return 1.95 
    
def adjust_reported_intake(reported_intake_kcal, weight_trend, weight_change_rate_kg_wk):
    """Adjusts reported intake based on weight trend to estimate true intake."""
    if reported_intake_kcal is None or weight_trend is None: return reported_intake_kcal if reported_intake_kcal is not None else 0
    if weight_trend == "Steady" or weight_change_rate_kg_wk is None: return reported_intake_kcal

    kcal_discrepancy_per_day = (weight_change_rate_kg_wk * KCAL_PER_KG_TISSUE) / 7.0
    adjusted_intake = reported_intake_kcal + kcal_discrepancy_per_day
    return adjusted_intake

def calculate_tef(intake_kcal, protein_g):
    """Calculates Thermic Effect of Food."""
    if intake_kcal is None or protein_g is None or intake_kcal == 0: return 0
    protein_kcal = protein_g * 4.0
    protein_percentage = (protein_kcal / intake_kcal) * 100.0
    tef_factor = 0.10 
    if protein_percentage > 25.0: tef_factor = 0.15 
    elif protein_percentage > 15.0: tef_factor = 0.12 
    return intake_kcal * tef_factor

def calculate_cold_thermogenesis(typical_indoor_temp_f, minutes_cold_exposure_daily):
    """Estimates calories burned due to cold exposure."""
    if typical_indoor_temp_f is None or minutes_cold_exposure_daily is None: return 0
    k_c_per_degree_f_day = 4.0; comfort_temp_f = 68.0; cold_kcal_indoor = 0.0
    if typical_indoor_temp_f < comfort_temp_f:
        temp_diff_f_indoor = comfort_temp_f - typical_indoor_temp_f
        cold_kcal_indoor = temp_diff_f_indoor * k_c_per_degree_f_day * ((24.0*60.0 - minutes_cold_exposure_daily) / (24.0*60.0))
    cold_kcal_outdoor = (minutes_cold_exposure_daily / 60.0) * 30.0
    return max(0, cold_kcal_indoor) + max(0, cold_kcal_outdoor)

def calculate_immune_fever_effect(has_fever_illness, peak_fever_temp_f, current_bmr_adaptive):
    """Estimates extra calories burned due to fever."""
    if not has_fever_illness or peak_fever_temp_f is None or peak_fever_temp_f <= 99.0 or current_bmr_adaptive is None: return 0.0
    temp_diff_f = peak_fever_temp_f - 98.6 
    if temp_diff_f <= 0: return 0.0 
    return current_bmr_adaptive * temp_diff_f * 0.065

def calculate_ffmi_fmi(ffm_kg, fm_kg, height_m):
    """Calculates Fat-Free Mass Index (FFMI) and Fat Mass Index (FMI)."""
    if ffm_kg is None or fm_kg is None or height_m is None or height_m <= 0: return 0,0
    ffmi = ffm_kg / (height_m**2) if ffm_kg >=0 else 0
    fmi = fm_kg / (height_m**2) if fm_kg >=0 else 0
    return ffmi, fmi

def calculate_implied_activity_breakdown(tdee_dlw, rmr_pontzer_ffm, weight_kg):
    """Derives implied activity components from FFM-based TDEE (DLW)."""
    if tdee_dlw is None or rmr_pontzer_ffm is None or weight_kg is None or \
       tdee_dlw <= 0 or rmr_pontzer_ffm <= 0 or weight_kg <= 0:
        return 0, 0, 0, 0 

    tdee_sedentary_wellfed_floor = rmr_pontzer_ffm * SEDENTARY_WELLFED_RMR_MULTIPLIER
    tef_at_sedentary_floor = tdee_sedentary_wellfed_floor * 0.10
    energy_non_locomotor_upregulation = tdee_sedentary_wellfed_floor - rmr_pontzer_ffm - tef_at_sedentary_floor
    energy_non_locomotor_upregulation = max(0, energy_non_locomotor_upregulation)

    energy_for_locomotion = tdee_dlw - tdee_sedentary_wellfed_floor
    energy_for_locomotion = max(0, energy_for_locomotion) 

    kcal_per_step = KCAL_PER_STEP_BASE_FACTOR * weight_kg
    implied_locomotor_steps = energy_for_locomotion / kcal_per_step if kcal_per_step > 0 else 0
    
    return energy_non_locomotor_upregulation, energy_for_locomotion, implied_locomotor_steps, tdee_sedentary_wellfed_floor
# --- Main Simulation Logic (Stage 1: TDEE Adaptation) ---
def simulate_tdee_adaptation(inputs, num_days_to_simulate=14):
    """
    Simulates TDEE adaptation over a number of days based on sustained intake and activity.
    """
    ffm_kg, fm_kg = calculate_ffm_fm(inputs['weight_kg'], inputs['body_fat_percentage'])
    
    # Use Pontzer FFM-based RMR as the primary initial BMR baseline
    initial_bmr_baseline = calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg)
    if initial_bmr_baseline <= 0: # Fallback if Pontzer RMR calculation fails
        initial_bmr_baseline = calculate_mifflin_st_jeor_rmr(inputs['weight_kg'], inputs['height_cm'], inputs['age_years'], inputs['sex'])
        if inputs.get('streamlit_object') and initial_bmr_baseline > 0: 
            inputs['streamlit_object'].warning("Pontzer FFM-RMR calculation failed or returned zero. Using Mifflin-St Jeor as initial BMR baseline for simulation.")
        elif inputs.get('streamlit_object') and initial_bmr_baseline <= 0:
            inputs['streamlit_object'].error("Critical error: Baseline RMR could not be calculated. Check inputs.")
            return pd.DataFrame(), {} # Return empty if BMR is still zero


    # Define Adaptive Bounds based on the reliable initial_bmr_baseline
    LAB = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR 
    UAB = calculate_dlw_tdee(ffm_kg, fm_kg) # FFM-based TDEE for upper bound
    
    # Heuristic for UAB if DLW TDEE is problematic
    pal_for_uab_heuristic = get_pal_multiplier_for_heuristic(inputs['avg_daily_steps'])
    min_plausible_UAB = initial_bmr_baseline * pal_for_uab_heuristic # Must be at least RMR * some PAL
    if UAB == 0 or UAB < LAB * 1.05 or UAB < min_plausible_UAB : 
        UAB = initial_bmr_baseline * pal_for_uab_heuristic * 1.05 # Heuristic: RMR * PAL + 5%
        if inputs.get('streamlit_object'): 
            inputs['streamlit_object'].warning(f"FFM-based DLW TDEE for UAB was low/zero or inconsistent ({calculate_dlw_tdee(ffm_kg, fm_kg):.0f} vs LAB {LAB:.0f}). Upper Adaptive Bound (UAB) set heuristically to {UAB:,.0f} kcal for simulation.")

    target_true_intake_for_sim = inputs['adjusted_intake']

    # EAT Calculation (fixed for sim period based on initial inputs)
    kcal_per_step = KCAL_PER_STEP_BASE_FACTOR * inputs['weight_kg']
    eat_kcal_steps = inputs['avg_daily_steps'] * kcal_per_step
    eat_kcal_additional_exercise = inputs['other_exercise_kcal_per_day']
    eat_kcal_fixed_for_sim = eat_kcal_steps + eat_kcal_additional_exercise

    # Initial state (Day 0 components)
    day0_bmr = initial_bmr_baseline
    day0_tef = calculate_tef(target_true_intake_for_sim, inputs['protein_g_per_day'])
    cold_kcal_fixed = calculate_cold_thermogenesis(inputs['typical_indoor_temp_f'], inputs['minutes_cold_exposure_daily'])
    day0_fever = calculate_immune_fever_effect(inputs['has_fever_illness'], inputs['peak_fever_temp_f'], day0_bmr)
    day0_neat_adaptive = 0.0 # Adaptive NEAT starts at zero offset
    
    # TDEE at the very start of the simulation, with new intake but unadapted BMR/NEAT
    TDEE_sim_start = day0_bmr + day0_tef + eat_kcal_fixed_for_sim + day0_neat_adaptive + cold_kcal_fixed + day0_fever

    # Initialize adaptive components
    current_bmr_adaptive = day0_bmr
    current_neat_adaptive_component = day0_neat_adaptive

    # Critical Low Intake Check
    CRITICAL_LOW_INTAKE_THRESHOLD = initial_bmr_baseline * 1.35 # e.g., if intake < RMR * 1.35
    is_critically_low_intake = target_true_intake_for_sim < CRITICAL_LOW_INTAKE_THRESHOLD
    
    # Physiological limits for BMR and NEAT adaptation
    min_bmr_limit_factor = 0.80; max_bmr_limit_factor = 1.15 
    min_bmr_target_limit = initial_bmr_baseline * min_bmr_limit_factor
    max_bmr_target_limit = initial_bmr_baseline * max_bmr_limit_factor
    min_neat_limit = -500; max_neat_limit = 750 # Adjusted NEAT limits slightly

    intake_in_adaptive_range = LAB <= target_true_intake_for_sim <= UAB
    daily_log = []

    for day in range(num_days_to_simulate):
        # Calculate daily non-adaptive components based on current state or fixed inputs
        current_eat_kcal = eat_kcal_fixed_for_sim 
        tef_kcal = calculate_tef(target_true_intake_for_sim, inputs['protein_g_per_day']) # Based on consistent target intake
        fever_kcal = calculate_immune_fever_effect(inputs['has_fever_illness'], inputs['peak_fever_temp_f'], current_bmr_adaptive)

        # Determine daily targets for BMR and NEAT based on intake scenario
        day_target_bmr = current_bmr_adaptive # Default to no change if no specific rule hit
        day_target_neat = current_neat_adaptive_component

        if is_critically_low_intake:
            # Rule: TDEE (BMR + NEAT portion) aims for initial_bmr_baseline * 1.2
            day_target_bmr = min_bmr_target_limit # BMR aims for its physiological floor
            target_bmr_plus_neat_floor = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
            day_target_neat = target_bmr_plus_neat_floor - day_target_bmr 
            day_target_neat = np.clip(day_target_neat, min_neat_limit, 0) # NEAT is suppressed, can be negative
        
        elif intake_in_adaptive_range:
            # TDEE (BMR + NEAT part) tries to adapt to make Total TDEE match intake
            total_adaptation_gap = target_true_intake_for_sim - TDEE_sim_start # Gap from Day 0 TDEE
            
            neat_share = 0.60 if total_adaptation_gap > 0 else 0.40 # NEAT more responsive to surplus
            bmr_share = 1.0 - neat_share # BMR takes the rest
            
            target_total_neat_change = total_adaptation_gap * neat_share
            target_total_bmr_change = total_adaptation_gap * bmr_share
            
            # Targets are relative to day0 adaptive states
            day_target_neat = np.clip(day0_neat_adaptive + target_total_neat_change, min_neat_limit, max_neat_limit)
            day_target_bmr = np.clip(day0_bmr + target_total_bmr_change, min_bmr_target_limit, max_bmr_target_limit)
        
        else: # Not critically low, AND outside primary adaptive range (driving tissue change primarily)
            # BMR and NEAT still adapt based on overall energy balance
            current_expenditure_for_balance = current_bmr_adaptive + tef_kcal + current_eat_kcal + current_neat_adaptive_component + cold_kcal_fixed + fever_kcal
            energy_balance = target_true_intake_for_sim - current_expenditure_for_balance # Positive if surplus

            bmr_target_change_factor_ext = 0.0
            if energy_balance > 250: bmr_target_change_factor_ext = 0.05 
            elif energy_balance < -250: bmr_target_change_factor_ext = -0.10
            day_target_bmr = initial_bmr_baseline * (1 + bmr_target_change_factor_ext) # Target relative to initial baseline BMR
            day_target_bmr = np.clip(day_target_bmr, min_bmr_target_limit, max_bmr_target_limit)

            neat_responsiveness = 0.30 # Base factor
            if inputs['avg_daily_steps'] > 10000: neat_responsiveness += 0.10
            if inputs['avg_daily_steps'] < 5000: neat_responsiveness -= 0.10
            if inputs['avg_sleep_hours'] < 6.5: neat_responsiveness -= 0.05
            if inputs['uses_caffeine']: neat_responsiveness += 0.05
            neat_responsiveness = np.clip(neat_responsiveness, 0.1, 0.5) # Bound responsiveness
            
            # NEAT target is proportional to the energy balance (increases in surplus, decreases in deficit)
            day_target_neat = energy_balance * neat_responsiveness 
            day_target_neat = np.clip(day_target_neat, min_neat_limit, max_neat_limit)

        # Adaptation Step (Euler method with front-loading for Day 1)
        front_load_factor = 0.0
        current_tau_bmr, current_tau_neat = TAU_BMR_ADAPTATION, TAU_NEAT_ADAPTATION

        if day == 0: # Apply front-loading on day 1
            if is_critically_low_intake:
                front_load_factor = 0.60 # More aggressive for critical low intake TDEE drop
                current_tau_bmr *= 0.5 # Faster adaptation to floor
                current_tau_neat *= 0.5
            elif intake_in_adaptive_range and (target_true_intake_for_sim - TDEE_sim_start) != 0: # If adaptation is needed
                front_load_factor = 0.40 # Standard front-load for adaptive range adjustments
        
        if front_load_factor > 0: # Apply front-loaded change from Day 0 state
            bmr_change_to_apply_day0 = (day_target_bmr - day0_bmr) * front_load_factor
            neat_change_to_apply_day0 = (day_target_neat - day0_neat_adaptive) * front_load_factor
            current_bmr_adaptive = day0_bmr + bmr_change_to_apply_day0
            current_neat_adaptive_component = day0_neat_adaptive + neat_change_to_apply_day0
        else: # Standard Euler step for subsequent days or if no front-loading
            delta_bmr = (day_target_bmr - current_bmr_adaptive) / current_tau_bmr
            current_bmr_adaptive += delta_bmr
            delta_neat = (day_target_neat - current_neat_adaptive_component) / current_tau_neat
            current_neat_adaptive_component += delta_neat
        
        # Enforce physiological limits for BMR and NEAT components individually
        current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit, max_bmr_target_limit)
        current_neat_adaptive_component = np.clip(current_neat_adaptive_component, min_neat_limit, max_neat_limit)

        # Enforce the BMR*1.2 floor for the SUM of (BMR_Adaptive + NEAT_Adaptive) if critically_low_intake
        if is_critically_low_intake:
            bmr_plus_neat_target_floor = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
            current_sum_bmr_neat = current_bmr_adaptive + current_neat_adaptive_component
            if current_sum_bmr_neat < bmr_plus_neat_target_floor:
                # BMR should not go below its own min_bmr_target_limit
                current_bmr_adaptive = max(current_bmr_adaptive, min_bmr_target_limit)
                # NEAT makes up the difference to reach the floor, but not below its own min_neat_limit
                required_neat_for_floor = bmr_plus_neat_target_floor - current_bmr_adaptive
                current_neat_adaptive_component = max(required_neat_for_floor, min_neat_limit)
                # If NEAT hit its floor, BMR might need to be slightly above its target to maintain the sum floor
                if current_neat_adaptive_component == min_neat_limit:
                    current_bmr_adaptive = max(bmr_plus_neat_target_floor - min_neat_limit, min_bmr_target_limit)
                # Final clip for BMR after adjustment
                current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit, max_bmr_target_limit)


        total_tdee_today = current_bmr_adaptive + tef_kcal + current_eat_kcal + current_neat_adaptive_component + cold_kcal_fixed + fever_kcal
        
        daily_log.append({
            "Day": day + 1, "Target Intake": target_true_intake_for_sim,
            "BMR_Adaptive": current_bmr_adaptive, "TEF": tef_kcal, "EAT": current_eat_kcal,
            "NEAT_Adaptive": current_neat_adaptive_component, "Cold_Thermo": cold_kcal_fixed,
            "Fever_Effect": fever_kcal, "Total_Dynamic_TDEE": total_tdee_today,
            "Energy_Balance_vs_TDEE": target_true_intake_for_sim - total_tdee_today,
            "Target_BMR_DailyStep": day_target_bmr, "Target_NEAT_DailyStep": day_target_neat,
            "Is_Critically_Low_Intake_Mode": is_critically_low_intake,
            "Is_In_Adaptive_Range_Mode": intake_in_adaptive_range and not is_critically_low_intake
        })

    final_tdee_val = daily_log[-1]['Total_Dynamic_TDEE'] if daily_log else TDEE_sim_start
    final_states = {
        "final_bmr_adaptive": current_bmr_adaptive,
        "final_neat_adaptive_component": current_neat_adaptive_component,
        "final_tdee": final_tdee_val,
        "initial_bmr_baseline": initial_bmr_baseline, # Pass out the actual baseline RMR used
        "LAB": LAB, "UAB": UAB, "intake_in_adaptive_range": intake_in_adaptive_range,
        "is_critically_low_intake_scenario": is_critically_low_intake
    }
    return pd.DataFrame(daily_log), final_states

# --- generate_bulk_cut_assessment (incorporates new FMI/FFMI HR logic) ---
def generate_bulk_cut_assessment(
    adjusted_intake, dynamic_tdee,
    initial_bmr_baseline, 
    ffm_kg, fm_kg, height_m, bmi, sex,
    sim_final_states 
    ):
    ffmi, fmi = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
    LAB = sim_final_states.get('LAB', initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR) 
    UAB = sim_final_states.get('UAB', initial_bmr_baseline * 1.7) # Fallback UAB if not in states
    intake_was_in_adaptive_range = sim_final_states.get('intake_in_adaptive_range', False)
    is_critical_low_intake_sim = sim_final_states.get('is_critically_low_intake_scenario', False)

    advice_primary = ""; status_caloric = ""
    daily_surplus_deficit_vs_dynamic_tdee = adjusted_intake - dynamic_tdee

    # Determine Caloric Status
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
    else: # Outside adaptive range but not critically low
        advice_primary += (f"Your intake was outside the primary metabolic adaptive range ({LAB:,.0f} - {UAB:,.0f} kcal). "
                           "This intake level is expected to primarily drive tissue change.\n")

    # Refined FMI/FFMI Hazard Ratio Logic
    advice_composition = ""; fmi_hr_category = ""; ffmi_hr_category = ""; ffmi_direct_bulk_triggered = False
    
    # FMI Categories based on user-provided text
    if fmi < 4: fmi_hr_category = "Very Low (HR ~1.2-1.4, under-fat risk)"
    elif fmi <= 8: fmi_hr_category = "Optimal (HR ~0.8-1.0, sweet spot)"
    elif fmi < 12: fmi_hr_category = "Slightly Elevated (HR >1.0-1.5, increasing risk)"
    elif fmi < 15: fmi_hr_category = "High (HR ~1.5, significant risk)"
    elif fmi < 20: fmi_hr_category = "Very High (HR ~2.0, very high risk)"
    else: fmi_hr_category = "Extremely High (HR >3.0, extremely high risk)"
    advice_composition += f"- **Fat Mass Index (FMI: {fmi:.1f} kg/m²):** {fmi_hr_category}\n"

    # FFMI Categories
    FFMI_DIRECT_BULK_THRESHOLD = 20.0
    if ffmi < FFMI_DIRECT_BULK_THRESHOLD: ffmi_direct_bulk_triggered = True
    
    if ffmi < 15: ffmi_hr_category = f"Critically Low (FFMI < 15, HR ~2+, high frailty risk)"
    elif ffmi < 18: ffmi_hr_category = f"Low (FFMI 15-17.9, HR > 1, elevated risk)"
    elif ffmi <= 22: ffmi_hr_category = f"Optimal (FFMI 18-22, HR ~0.8-1.0, protective)"
    elif ffmi <= 24: ffmi_hr_category = f"High/Sufficient (FFMI 22.1-24, HR ~1.0, no extra longevity benefit)"
    else: ffmi_hr_category = f"Very High (FFMI > 24, HR may slightly increase, monitor)"
    advice_composition += f"- **Fat-Free Mass Index (FFMI: {ffmi:.1f} kg/m²):** {ffmi_hr_category}"
    if ffmi_direct_bulk_triggered and not ffmi_hr_category.startswith("Critically Low") and not ffmi_hr_category.startswith("Low"):
        advice_composition += f" (Below your {FFMI_DIRECT_BULK_THRESHOLD:.0f} target, suggesting bulk)\n"
    else:
        advice_composition += "\n"


    final_recommendation = ""; overall_status_message = f"Caloric: {status_caloric} | FMI: {fmi_hr_category} | FFMI: {ffmi_hr_category}"

    # Dominance logic refined
    fmi_high_risk_flag = fmi > 8 # Simplified: FMI above sweet spot indicates need to be cautious or cut
    ffmi_low_risk_flag = ffmi < 18 # FFMI below optimal/sweet spot indicates need to bulk for HR benefit

    if ffmi_direct_bulk_triggered: # User rule: FFMI < 20
        if fmi > FMI_HIGH_RISK_HR_THRESHOLD: # If FMI is very high risk
            final_recommendation = "REC: Complex - Body Recomp or Prioritize Fat Loss then Lean Bulk. FFMI is below 20, but FMI is also very high risk. Consult a professional."
        else:
            final_recommendation = f"REC: BULK. Your FFMI ({ffmi:.1f}) is below the {FFMI_DIRECT_BULK_THRESHOLD:.0f} kg/m² target. Focus on a caloric surplus."
            if status_caloric == "Deficit": final_recommendation += " Current state is deficit; increase intake."
            elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; surplus needed."
    elif fmi > FMI_HIGH_RISK_HR_THRESHOLD: # High FMI risk takes precedence if FFMI is not < 20
        final_recommendation = f"REC: CUT. Your FMI ({fmi:.1f}) is in a high-risk zone ({fmi_hr_category}). Focus on a caloric deficit."
        if status_caloric == "Surplus": final_recommendation += " Current state is surplus; decrease intake."
        elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; a deficit is needed."
    elif ffmi_low_risk_flag: # FFMI is not < 20, but still in a >1 HR zone (e.g. 15-17.9)
        final_recommendation = f"REC: Consider Lean Bulk. Your FFMI ({ffmi:.1f}) is in a zone with elevated health risk ({ffmi_hr_category})."
        if status_caloric == "Deficit": final_recommendation += " Current state is deficit; adjust."
        elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; slight surplus may be beneficial if FMI allows."
    
    # Fallback to caloric status if no strong HR-based driver from above
    elif status_caloric == "Surplus": final_recommendation = "REC: SURPLUS. If bulking, this aligns. Monitor body composition."
    elif status_caloric == "Deficit": final_recommendation = "REC: DEFICIT. If cutting, this aligns. Ensure adequate protein and training."
    elif status_caloric == "Maintenance":
        if "Optimal" in fmi_hr_category and ("Optimal" in ffmi_hr_category or "Sufficient/High" in ffmi_hr_category or "Very High" in ffmi_hr_category) :
             final_recommendation = "REC: MAINTAIN or Optimize. Your body composition appears to be in a healthy range. Adjust for specific goals."
        else: final_recommendation = "REC: MAINTAIN & RE-EVALUATE. Intake supports maintenance. Review FMI/FFMI status for next phase."
    
    if not final_recommendation: # Catch-all if no specific recommendation was triggered yet
        final_recommendation = "Review your caloric intake in context of your FMI/FFMI status and personal goals. Adjust as needed."


    full_advice = (f"{advice_primary}\n\n**Body Composition Insights (FMI/FFMI & Hazard Ratio Interpretation):**\n{advice_composition}\n**Overall Strategy Guidance:**\n{final_recommendation}")
    return full_advice, overall_status_message, daily_surplus_deficit_vs_dynamic_tdee
# --- project_weight_change_scenarios function ---
def project_weight_change_scenarios(current_dynamic_tdee, weight_kg):
    """
    Generates a table of potential weight change rates at different intake levels,
    based on the provided current dynamic TDEE.
    """
    scenarios = []
    kcal_per_lb_wk_as_daily = 3500.0 / 7.0 # kcal adjustment per day to change 1 lb per week

    # Define intake levels relative to the current dynamic TDEE for different goals
    intake_targets_desc = {
        "Aggressive Cut (~ -1.0 lbs/wk)": current_dynamic_tdee - (1.0 * kcal_per_lb_wk_as_daily),
        "Moderate Cut (~ -0.5 lbs/wk)": current_dynamic_tdee - (0.5 * kcal_per_lb_wk_as_daily),
        "Maintenance (at current Dynamic TDEE)": current_dynamic_tdee,
        "Lean Bulk (~ +0.25 lbs/wk)": current_dynamic_tdee + (0.25 * kcal_per_lb_wk_as_daily),
        "Moderate Bulk (~ +0.5 lbs/wk)": current_dynamic_tdee + (0.5 * kcal_per_lb_wk_as_daily),
    }

    for desc, intake_kcal in intake_targets_desc.items():
        daily_delta_vs_tdee = intake_kcal - current_dynamic_tdee # Raw caloric difference from current TDEE
        
        # Determine efficiency factor based on surplus or deficit
        if daily_delta_vs_tdee > 0: # Surplus
            eff_factor = SURPLUS_EFFICIENCY 
        elif daily_delta_vs_tdee < 0: # Deficit
            eff_factor = DEFICIT_EFFICIENCY
        else: # Maintenance
            eff_factor = 1.0 
            
        effective_daily_stored_or_lost_kcal = daily_delta_vs_tdee * eff_factor
        
        weekly_kg_change = (effective_daily_stored_or_lost_kcal * 7) / KCAL_PER_KG_TISSUE
        weekly_lbs_change = kg_to_lbs(weekly_kg_change)
        
        scenarios.append({
            "Scenario": desc,
            "Target Daily Intake (kcal)": f"{intake_kcal:,.0f}",
            "Est. Effective Daily Surplus/Deficit (kcal)": f"{effective_daily_stored_or_lost_kcal:+.0f}",
            "Predicted Weekly Weight Change": f"{weekly_lbs_change:+.2f} lbs ({weekly_kg_change:+.3f} kg)"
        })
    return pd.DataFrame(scenarios)

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Advanced TDEE & Metabolic Modeler")

INFO_ICON = "❓" # Helper for tooltips
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

# Initialize session state for all inputs if not already present
def init_session_state():
    """Initializes session state with default values for all input widgets."""
    defaults = {
        # Stage 1 Keys
        "weight_unit": "lbs", 
        "height_unit": "ft/in",
        "weight_input_val": 150.0, # This will be adjusted based on unit if not initialized
        "body_fat_percentage": 15.0, 
        "sex": "Male", 
        "age_years": 25,
        "feet": 5, "inches": 10, "height_cm_input": 178.0,
        "avg_daily_steps": 7500, 
        "other_exercise_kcal_per_day": 0,
        "avg_daily_kcal_intake_reported": 2500, 
        "protein_g_per_day": 150.0,
        "weight_trend": "Steady", 
        "weight_change_rate_input_val_lbs": 0.5, # Specific key for lbs rate input
        "weight_change_rate_input_val_kg": 0.23, # Specific key for kg rate input
        "typical_indoor_temp_f": 70, 
        "minutes_cold_exposure_daily": 0,
        "avg_sleep_hours": 7.5, 
        "uses_caffeine": True, 
        "has_fever_illness": False,
        "peak_fever_temp_f_input": 98.6, 
        "num_days_to_simulate_s1": 14, # Renamed for clarity for Stage 1 simulation
        "weight_input_val_initialized_flag" : False, # Flag to set initial weight based on unit

        # Stage 2 Keys (will be set by Stage 2 inputs later)
        "stage2_goal": "Bulk", # Default, user will choose
        "stage2_rate_lbs_wk": 0.5, # Default rate
        "stage2_duration_weeks": 8, # Default duration
        "stage2_target_protein_g": 150.0,
        "stage2_target_carb_g": 300.0,
        "stage2_target_fat_g": 70.0,
        "stage2_training_status": "Novice",
        "stage2_weightlifting": True,
        "stage1_results_calculated": False # Flag to control Stage 2 UI visibility
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Adjust initial weight_input_val based on selected unit if not already done
    if not st.session_state.weight_input_val_initialized_flag:
        if st.session_state.weight_unit == "kg":
            st.session_state.weight_input_val = 68.0
        else: # lbs
            st.session_state.weight_input_val = 150.0
        st.session_state.weight_input_val_initialized_flag = True

init_session_state() # Initialize session state at the start of the script

# Sidebar inputs will now directly use st.session_state via their `key` parameter
def display_sidebar_stage1_inputs():
    """Defines and displays Stage 1 input widgets in the sidebar."""
    st.sidebar.header("📝 User Inputs (Stage 1 Analysis)")
    st.sidebar.markdown("Inputs are saved for your current browser session.")

    unit_cols = st.sidebar.columns(2)
    # The radio calls directly update st.session_state[key]
    unit_cols[0].radio("Weight unit:", ("kg", "lbs"), 
                       key="weight_unit", 
                       index=(["kg", "lbs"].index(st.session_state.weight_unit) if st.session_state.weight_unit in ["kg", "lbs"] else 0))
    unit_cols[1].radio("Height unit:", ("cm", "ft/in"), 
                       key="height_unit",
                       index=(["cm","ft/in"].index(st.session_state.height_unit) if st.session_state.height_unit in ["cm","ft/in"] else 0))

    st.sidebar.subheader("👤 Body & Demographics")
    
    # The number_input will use the value from st.session_state.weight_input_val and update it
    st.sidebar.number_input(f"Current Body Weight ({st.session_state.weight_unit}):", 
                            min_value=(50.0 if st.session_state.weight_unit == "lbs" else 20.0), 
                            max_value=(700.0 if st.session_state.weight_unit == "lbs" else 300.0), 
                            step=0.1, format="%.1f", key="weight_input_val")

    st.sidebar.slider("Estimated Body Fat Percentage (%):", min_value=3.0, max_value=60.0, step=0.5, format="%.1f", key="body_fat_percentage")
    st.sidebar.selectbox("Sex:", ("Male", "Female"), key="sex") # Default/index handled by init
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
    
    st.sidebar.subheader(f"🍽️ Diet (Target or Current Average) {INFO_ICON}", help="Your average daily food intake for the period being analyzed or as a target for simulation.")
    st.sidebar.number_input("Reported Average Daily Caloric Intake (kcal):", min_value=500, max_value=10000, step=50, key="avg_daily_kcal_intake_reported")
    st.sidebar.number_input("Protein Intake (grams per day):", min_value=0.0, max_value=500.0, step=1.0, format="%.1f", key="protein_g_per_day")
    
    st.sidebar.subheader(f"⚖️ Observed Weight Trend {INFO_ICON}", help=TOOLTIPS["weight_change_rate"])
    st.sidebar.caption("Used to calibrate true intake from reported intake.")
    st.sidebar.selectbox("Recent Body Weight Trend:", ("Steady", "Gaining", "Losing"), key="weight_trend")
    
    rate_help_text = "Average weekly change over last 2-4 weeks. E.g., 0.5 for gaining 0.5 units/wk, or 0.25 for losing 0.25 units/wk."
    if st.session_state.weight_trend != "Steady":
        if st.session_state.weight_unit == "lbs":
            # This widget will update st.session_state.weight_change_rate_input_val_lbs
            st.sidebar.number_input(f"Rate of {st.session_state.weight_trend.lower()} (lbs/week):", min_value=0.01, max_value=5.0, step=0.05, format="%.2f", key="weight_change_rate_input_val_lbs", help=rate_help_text)
        else: # kg
             # This widget will update st.session_state.weight_change_rate_input_val_kg
             st.sidebar.number_input(f"Rate of {st.session_state.weight_trend.lower()} (kg/week):", min_value=0.01, max_value=2.5, step=0.01, format="%.2f", key="weight_change_rate_input_val_kg", help=rate_help_text)
    
    st.sidebar.subheader("🌡️ Environment & Physiology")
    st.sidebar.slider("Typical Indoor Temperature (°F):", min_value=50, max_value=90, step=1, key="typical_indoor_temp_f")
    st.sidebar.number_input("Avg. Min/Day Outdoors <60°F/15°C:", min_value=0, max_value=1440, step=15, key="minutes_cold_exposure_daily")
    st.sidebar.slider("Habitual Nightly Sleep (hours):", min_value=4.0, max_value=12.0, step=0.1, format="%.1f", key="avg_sleep_hours")
    st.sidebar.checkbox("Regular Caffeine User?", key="uses_caffeine")
    st.sidebar.checkbox("Current Fever or Acute Illness?", key="has_fever_illness")
    if st.session_state.has_fever_illness: # Only show if checked
        st.sidebar.number_input("Peak Fever Temp (°F):", min_value=98.6, max_value=106.0, step=0.1, format="%.1f", key="peak_fever_temp_f_input")
    
    st.sidebar.slider("Stage 1 Simulation Duration (days for TDEE adaptation graph):", 
                      min_value=7, max_value=90, value=st.session_state.num_days_to_simulate_s1, step=7, key="num_days_to_simulate_s1")

display_sidebar_stage1_inputs() # Call function to display sidebar and manage state via keys

# --- Main App Title (already called once, but for context here)---
st.title("💪 Advanced Dynamic TDEE & Metabolic Modeler ⚙️")
# ... Main intro markdown ...

# --- Stage 1 Analysis Area ---
st.header("📊 Stage 1: Current TDEE Analysis & Metabolic Snapshot")
# --- Streamlit App UI Main Display Area (Continued from Chunk 3) ---

st.title("💪 Advanced Dynamic TDEE & Metabolic Modeler ⚙️")
st.markdown("""
This tool simulates Total Daily Energy Expenditure (TDEE) by modeling metabolic adaptations.
It incorporates body composition health risk profiles (FMI/FFMI) for nuanced nutritional strategy insights.
Inputs should reflect **current, stable conditions** for initial assessment, or **target conditions** for simulation.
""")
st.header("📊 Stage 1: Current TDEE Analysis & Metabolic Snapshot")

if st.sidebar.button("🚀 Calculate & Simulate TDEE (Stage 1)", type="primary", use_container_width=True, key="calculate_stage1_button"):
    # Retrieve ALL values from st.session_state using their keys
    # These keys were defined in display_sidebar_stage1_inputs()
    s_weight_unit = st.session_state.get("weight_unit", "lbs")
    s_height_unit = st.session_state.get("height_unit", "ft/in")
    s_weight_input_val = st.session_state.get("weight_input_val", 150.0 if s_weight_unit == "lbs" else 68.0)
    
    if s_weight_unit == "lbs": 
        s_weight_kg = lbs_to_kg(s_weight_input_val)
    else: 
        s_weight_kg = s_weight_input_val

    s_body_fat_percentage = st.session_state.get("body_fat_percentage", 15.0)
    s_sex = st.session_state.get("sex", "Male")
    s_age_years = st.session_state.get("age_years", 25)

    if s_height_unit == "ft/in": 
        s_height_cm = ft_in_to_cm(st.session_state.get("feet", 5), st.session_state.get("inches", 10))
    else: 
        s_height_cm = st.session_state.get("height_cm_input", 178.0)
    
    s_avg_daily_steps = st.session_state.get("avg_daily_steps", 7500)
    s_other_exercise_kcal_per_day = st.session_state.get("other_exercise_kcal_per_day", 0)
    s_avg_daily_kcal_intake_reported = st.session_state.get("avg_daily_kcal_intake_reported", 2500)
    s_protein_g_per_day = st.session_state.get("protein_g_per_day", 150.0)
    s_weight_trend = st.session_state.get("weight_trend", "Steady")
    
    s_weight_change_rate_display_val = 0.0 
    s_weight_change_rate_kg_wk = 0.0
    if s_weight_trend != "Steady":
        if s_weight_unit == "lbs":
            s_weight_change_rate_display_val = st.session_state.get("weight_change_rate_input_val_lbs", 0.0)
            s_weight_change_rate_kg_wk = lbs_to_kg(s_weight_change_rate_display_val)
        else: # kg
            s_weight_change_rate_display_val = st.session_state.get("weight_change_rate_input_val_kg", 0.0)
            s_weight_change_rate_kg_wk = s_weight_change_rate_display_val
    
    s_typical_indoor_temp_f = st.session_state.get("typical_indoor_temp_f", 70)
    s_minutes_cold_exposure_daily = st.session_state.get("minutes_cold_exposure_daily", 0)
    s_avg_sleep_hours = st.session_state.get("avg_sleep_hours", 7.5)
    s_uses_caffeine = st.session_state.get("uses_caffeine", True)
    s_has_fever_illness = st.session_state.get("has_fever_illness", False)
    s_peak_fever_temp_f_input = st.session_state.get("peak_fever_temp_f_input", 98.6) if s_has_fever_illness else 98.6
    s_num_days_to_simulate_s1 = st.session_state.get("num_days_to_simulate_s1", 14)


    if s_height_cm <= 0:
        st.error("Height must be a positive value. Please check your inputs.")
    else:
        ffm_kg, fm_kg = calculate_ffm_fm(s_weight_kg, s_body_fat_percentage)
        height_m = s_height_cm / 100.0
        bmi = s_weight_kg / (height_m**2) if height_m > 0 else 0

        # Edge Case: Body Composition Check
        ffmi_check, fmi_check = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
        # More lenient checks for extreme values, focusing on very improbable scenarios
        if (s_sex == "Male" and ffmi_check > 40) or \
           (s_sex == "Female" and ffmi_check > 32) or \
           (s_body_fat_percentage < 2.0 and s_sex == "Male") or \
           (s_body_fat_percentage < 8.0 and s_sex == "Female") or \
           s_body_fat_percentage > 65:
            st.warning("⚠️ Your input body composition results in an FFMI or body fat level that is physiologically extreme or highly unlikely. Results may be less reliable. Please double-check your inputs.")

        st.subheader("📋 Initial Body Composition & Indices")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Weight", f"{s_weight_kg:.1f} kg / {kg_to_lbs(s_weight_kg):.1f} lbs")
        res_col2.metric("FFM", f"{ffm_kg:.1f} kg")
        res_col3.metric("FM", f"{fm_kg:.1f} kg ({s_body_fat_percentage:.1f}%)")
        res_col4.metric("BMI", f"{bmi:.1f}")

        current_ffmi, current_fmi = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
        idx_col1, idx_col2 = st.columns(2)
        idx_col1.metric(f"FFMI {INFO_ICON}", f"{current_ffmi:.1f} kg/m²", help=TOOLTIPS["FFMI"])
        idx_col2.metric(f"FMI {INFO_ICON}", f"{current_fmi:.1f} kg/m²", help=TOOLTIPS["FMI"])
        
        # Calculate initial RMR using Pontzer FFM-based, fallback to Mifflin
        initial_bmr_ref = calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg)
        if initial_bmr_ref <= 0: # Check if Pontzer RMR failed
            initial_bmr_ref = calculate_mifflin_st_jeor_rmr(s_weight_kg, s_height_cm, s_age_years, s_sex)
            st.caption("Using Mifflin-St Jeor RMR as fallback for baseline.")
        if initial_bmr_ref <= 0: # If Mifflin also fails (e.g. bad inputs for it too)
            st.error("Could not calculate a valid baseline RMR. Please check all body inputs.")
            st.stop() # Halt further execution if no valid RMR

        # Static Metabolic Range Display
        # Lower bound is RMR * 1.2 (tanked BMR+NEAT floor)
        lower_bound_tdee_static_display = initial_bmr_ref * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
        # Upper bound is DLW TDEE (free-living)
        upper_bound_tdee_static_display_calc = calculate_dlw_tdee(ffm_kg, fm_kg)
        
        # Heuristic for UAB if DLW fails or is too low
        pal_for_UAB_display_heuristic = get_pal_multiplier_for_heuristic(s_avg_daily_steps)
        min_plausible_UAB_display = initial_bmr_ref * pal_for_UAB_display_heuristic # At least RMR * some activity
        if upper_bound_tdee_static_display_calc <= lower_bound_tdee_static_display or upper_bound_tdee_static_display_calc < min_plausible_UAB_display:
            upper_bound_tdee_static_display = initial_bmr_ref * pal_for_UAB_display_heuristic * 1.05 # Add a small margin
            if upper_bound_tdee_static_display_calc == 0: 
                 st.caption("Note: FFM-based TDEE (DLW) was zero. Using PAL-based heuristic for static upper bound display.")
        else:
            upper_bound_tdee_static_display = upper_bound_tdee_static_display_calc
        
        # Adjusted True Intake Calculation
        adjusted_true_intake = adjust_reported_intake(s_avg_daily_kcal_intake_reported, s_weight_trend, s_weight_change_rate_kg_wk)
        
        # Edge Case: Intake Check (moved after initial_bmr_ref is confirmed)
        if adjusted_true_intake < initial_bmr_ref * 0.9 and initial_bmr_ref > 0: 
            st.warning(f"⚠️ Calibrated intake ({adjusted_true_intake:,.0f} kcal) is very low vs RMR ({initial_bmr_ref:,.0f} kcal). This may be unsustainable, carry health risks, and lead to significant metabolic slowdown. The simulation will activate 'Critically Low Intake Mode'.")
        elif upper_bound_tdee_static_display > 0 and adjusted_true_intake > upper_bound_tdee_static_display * 1.75 : 
            st.warning(f"⚠️ Calibrated intake ({adjusted_true_intake:,.0f} kcal) is very high vs upper TDEE ({upper_bound_tdee_static_display:,.0f} kcal). This suggests a very large surplus and rapid weight gain, potentially with unfavorable body composition changes.")

        with st.expander("Advanced Metabolic Insights & Benchmarks", expanded=False):
            st.markdown(f"#### ↔️ Estimated Static Metabolic Range {INFO_ICON}", help="General reference for TDEE boundaries. The simulation below models dynamic adaptations within a similar, internally calculated range (LAB & UAB).")
            st.markdown(f"""
            - **Static Lower Adaptive Bound (approx. RMR + minimal NEAT): `{lower_bound_tdee_static_display:,.0f} kcal/day`** (Initial RMR * {CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR:.1f}). <span title='{TOOLTIPS["LAB"]}'>{INFO_ICON}</span>
            - **Static Upper Adaptive Bound (Typical Free-living TDEE): `{upper_bound_tdee_static_display:,.0f} kcal/day`** (FFM-based DLW formula or heuristic). <span title='{TOOLTIPS["UAB"]}'>{INFO_ICON}</span>
            """, unsafe_allow_html=True)
            
            # Calculate and display implied activity breakdown
            energy_non_loco, energy_loco, implied_steps, tdee_sed_floor_calc = calculate_implied_activity_breakdown(
                upper_bound_tdee_static_display, # Use the displayed UAB for this calculation
                initial_bmr_ref,      
                s_weight_kg
            )
            st.markdown(f"#### 🚶‍♂️Implied Activity for FFM-Based TDEE (DLW) of `{upper_bound_tdee_static_display:,.0f}` kcal")
            st.markdown(f"""
            The FFM-Based TDEE (DLW) for your body composition inherently includes an activity level typical for maintaining that physique. A potential breakdown:
            - RMR (Pontzer FFM-based or fallback): `{initial_bmr_ref:,.0f} kcal/day`
            - TEF (~10% at this TDEE level): `{upper_bound_tdee_static_display * 0.10:,.0f} kcal/day` <span title='{TOOLTIPS["TEF"]}'>{INFO_ICON}</span>
            - Non-Locomotor Upregulation & Base NEAT (to reach ~RMR*{SEDENTARY_WELLFED_RMR_MULTIPLIER:.1f} before significant steps, estimated at `{tdee_sed_floor_calc:,.0f}` kcal TDEE): **`{energy_non_loco:,.0f} kcal/day`** <span title='{TOOLTIPS["NEAT"]}'>{INFO_ICON}</span>
            - Remaining Energy for Deliberate Locomotion (steps/cardio to reach full DLW TDEE): **`{energy_loco:,.0f} kcal/day`** <span title='{TOOLTIPS["EAT"]}'>{INFO_ICON}</span>
            - This locomotor portion is roughly equivalent to: **`{implied_steps:,.0f} steps/day`**.
            
            Compare this implied locomotor energy/steps to your inputted daily steps (`{s_avg_daily_steps:,.0f}`) & other exercise (`{s_other_exercise_kcal_per_day:,.0f} kcal`).
            """, unsafe_allow_html=True)

        st.metric("Reported Avg. Daily Intake", f"{s_avg_daily_kcal_intake_reported:,.0f} kcal")
        if abs(adjusted_true_intake - s_avg_daily_kcal_intake_reported) > 20: # Only show if adjustment is notable
            unit_for_trend_cap = s_weight_unit 
            display_rate_val_for_cap = s_weight_change_rate_display_val 
            if s_weight_trend != "Steady" :
                 st.metric("Calibrated True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal", help="Estimated from reported intake & recent weight trend; used as target for simulation.")
                 # Determine the rate value as input by the user for the caption
                 if s_weight_unit == "kg":
                     rate_val_for_caption_display = st.session_state.get("weight_change_rate_input_val_kg",0.0)
                 else: # lbs
                     rate_val_for_caption_display = st.session_state.get("weight_change_rate_input_val_lbs",0.0)
                 st.caption(f"Adjusted based on reported weight trend of {rate_val_for_caption_display:.2f} {unit_for_trend_cap}/week ({s_weight_trend}).")
            # else: # If steady, no need for "Calibrated" if not different, but main metric is already shown
            #      st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")
        else: # No significant adjustment or trend was steady
            st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")

        # Prepare inputs for the simulation function
        simulation_inputs = {
            "weight_kg": s_weight_kg, "body_fat_percentage": s_body_fat_percentage,
            "sex": s_sex, "age_years": s_age_years, "height_cm": s_height_cm,
            "avg_daily_steps": s_avg_daily_steps, 
            "other_exercise_kcal_per_day": s_other_exercise_kcal_per_day, # Correct key used
            "adjusted_intake": adjusted_true_intake,
            "protein_g_per_day": s_protein_g_per_day,
            "typical_indoor_temp_f": s_typical_indoor_temp_f,
            "minutes_cold_exposure_daily": s_minutes_cold_exposure_daily,
            "avg_sleep_hours": s_avg_sleep_hours, 
            "uses_caffeine": s_uses_caffeine, # Already boolean
            "has_fever_illness": s_has_fever_illness, # Already boolean
            "peak_fever_temp_f": s_peak_fever_temp_f_input,
            "streamlit_object": st # Pass Streamlit object for warnings inside functions
        }
        st.session_state.stage1_results_calculated = True # Flag to show Stage 2 inputs
        st.session_state.stage1_simulation_inputs = simulation_inputs # Store for Stage 2
        st.session_state.stage1_ffm_kg = ffm_kg
        st.session_state.stage1_fm_kg = fm_kg
        st.session_state.stage1_initial_bmr = initial_bmr_ref
        st.session_state.stage1_adjusted_intake = adjusted_true_intake
        # --- (Continuing within the `if st.sidebar.button("🚀 Calculate & Simulate TDEE (Stage 1)", ...):` block from Chunk 4) ---
# --- (Assuming all variables s_weight_unit, s_height_unit, s_weight_kg, etc. are defined as in Chunk 4) ---
# --- (And ffm_kg, fm_kg, height_m, bmi, initial_bmr_ref, lower_bound_tdee_static_display, 
#      upper_bound_tdee_static_display, adjusted_true_intake, simulation_inputs are all calculated)

        # --- Stage 1 TDEE Adaptation Simulation ---
        st.subheader(f"⏳ Simulated TDEE Adaptation Over {s_num_days_to_simulate_s1} Days")
        st.caption(f"Based on maintaining a calibrated true intake of **{adjusted_true_intake:,.0f} kcal/day** and other lifestyle factors.")
        
        daily_tdee_log_df, final_tdee_states = simulate_tdee_adaptation(simulation_inputs, s_num_days_to_simulate_s1)

        current_dynamic_tdee = 0.0 # Initialize
        if not daily_tdee_log_df.empty and final_tdee_states:
            current_dynamic_tdee = final_tdee_states.get('final_tdee', 0)
            LAB_sim = final_tdee_states.get('LAB', 0)
            UAB_sim = final_tdee_states.get('UAB', 0)
            crit_low_active = final_tdee_states.get('is_critically_low_intake_scenario', False)
            
            crit_low_msg = f"<span style='color:orange; font-weight:bold;'> (Critically Low Intake Mode {'Activated' if crit_low_active else 'Inactive'})</span>"
            
            st.metric(f"Simulated Dynamic TDEE (at Day {s_num_days_to_simulate_s1})", f"{current_dynamic_tdee:,.0f} kcal/day",
                      help="This is the model's estimate of your TDEE after metabolic adaptation to the specified intake and activity over the simulation period.")
            st.markdown(f"Primary Metabolic Adaptive Range used in simulation: `{LAB_sim:,.0f} - {UAB_sim:,.0f} kcal/day`. {crit_low_msg}", unsafe_allow_html=True)

            with st.expander("Show Detailed Daily Simulation Log & Component Breakdown (Stage 1)", expanded=False):
                cols_to_display = [col for col in ["Day", "Target Intake", "BMR_Adaptive", "TEF", "EAT", "NEAT_Adaptive", "Cold_Thermo", "Fever_Effect", "Total_Dynamic_TDEE", "Energy_Balance_vs_TDEE", "Target_BMR_DailyStep", "Target_NEAT_DailyStep"] if col in daily_tdee_log_df.columns]
                display_df_log = daily_tdee_log_df[cols_to_display]
                st.dataframe(display_df_log.style.format("{:,.0f}", na_rep="-", subset=pd.IndexSlice[:, [c for c in cols_to_display if c != 'Day']]))
            
            chart_cols_to_plot = ['Total_Dynamic_TDEE', 'BMR_Adaptive', 'NEAT_Adaptive', 'TEF', 'EAT']
            valid_chart_cols = [col for col in chart_cols_to_plot if col in daily_tdee_log_df.columns and daily_tdee_log_df[col].abs().sum() > 0.01] # Plot if not all zero
            if 'Day' in daily_tdee_log_df.columns and len(valid_chart_cols) > 0:
                chart_data = daily_tdee_log_df[['Day'] + valid_chart_cols].copy()
                chart_data.set_index('Day', inplace=True)
                st.line_chart(chart_data)
                st.caption("Chart shows key TDEE components adapting over the simulation period.")
        else:
            st.error("Stage 1 TDEE Adaptation Simulation failed to produce results.")
            # Fallback TDEE if simulation fails
            current_dynamic_tdee = (lower_bound_tdee_static_display + upper_bound_tdee_static_display) / 2.0 
            final_tdee_states = {} # Ensure final_tdee_states is a dict

        # --- Stage 1 Nutritional Strategy Assessment ---
        st.subheader(f"🎯 Nutritional Strategy Assessment (Stage 1) {INFO_ICON}", help="Guidance based on simulated TDEE, body composition, and FMI/FFMI risk profiles.")
        
        # Ensure initial_bmr_ref is passed correctly; it's calculated before simulation_inputs
        # and should be the same as final_tdee_states.get('initial_bmr_baseline', initial_bmr_ref)
        # The 'initial_bmr_baseline' in final_tdee_states is the one used *in the simulation*.
        
        advice_s1, overall_status_msg_s1, daily_surplus_deficit_val_s1 = generate_bulk_cut_assessment(
            adjusted_true_intake, 
            current_dynamic_tdee,
            final_tdee_states.get('initial_bmr_baseline', initial_bmr_ref), # Pass the actual baseline RMR used in sim
            ffm_kg, 
            fm_kg, 
            height_m, 
            bmi, 
            s_sex,
            final_tdee_states # Pass all simulation results, including LAB, UAB, crit_low flags
        )
        
        # Color coding the status message for Stage 1
        if "SURPLUS" in overall_status_msg_s1.upper() and "CUT" not in advice_s1.upper(): st.success(f"{overall_status_msg_s1}")
        elif "DEFICIT" in overall_status_msg_s1.upper() and "BULK" not in advice_s1.upper(): st.error(f"{overall_status_msg_s1}")
        elif "COMPLEX" in advice_s1.upper() or "CONSULT" in advice_s1.upper() or "CAREFUL" in advice_s1.upper() : st.warning(f"{overall_status_msg_s1}")
        else: st.info(f"{overall_status_msg_s1}")
        st.markdown(advice_s1)

        # --- Stage 1 Future Intake Scenarios ---
        st.subheader(f"📅 Future Intake Scenarios & Projected Weight Change (Based on Stage 1 TDEE) {INFO_ICON}", 
                     help="Estimates based on maintaining different intake levels if your TDEE stabilized as per the Stage 1 simulation.")
        st.caption(f"Based on your simulated dynamic TDEE of **{current_dynamic_tdee:,.0f} kcal/day** (after {s_num_days_to_simulate_s1}-day adaptation). The 'Est. Effective Surplus/Deficit' accounts for metabolic efficiencies (e.g., ~85% of a surplus contributes to tissue gain).")
        df_weight_scenarios_s1 = project_weight_change_scenarios(current_dynamic_tdee, s_weight_kg)
        st.dataframe(df_weight_scenarios_s1, hide_index=True)
        
        st.success("✅ Stage 1 Analysis Complete!")
        st.info("Note: Metabolic adaptation is complex. This model provides estimates. Real-world results can vary, and plateaus are common with prolonged dietary changes.")

        # --- Store results for Stage 2 ---
        st.session_state.stage1_results_calculated = True
        st.session_state.stage1_final_tdee = current_dynamic_tdee
        st.session_state.stage1_ffm_kg = ffm_kg
        st.session_state.stage1_fm_kg = fm_kg
        st.session_state.stage1_weight_kg = s_weight_kg # Store starting weight for Stage 2
        st.session_state.stage1_height_m = height_m
        st.session_state.stage1_initial_bmr_baseline = final_tdee_states.get('initial_bmr_baseline', initial_bmr_ref)
        st.session_state.stage1_adjusted_intake = adjusted_true_intake # Intake that led to this TDEE
        st.session_state.stage1_protein_g_day = s_protein_g_per_day # Protein from stage 1 diet
        st.session_state.stage1_avg_daily_steps = s_avg_daily_steps # Activity from stage 1
        st.session_state.stage1_other_exercise_kcal = s_other_exercise_kcal_per_day # Activity from stage 1
        st.session_state.stage1_sex = s_sex
        st.session_state.stage1_age_years = s_age_years
        st.session_state.stage1_height_cm = s_height_cm
        st.session_state.stage1_is_critically_low_intake = final_tdee_states.get('is_critically_low_intake_scenario', False)
        # Pass other relevant inputs if needed by stage 2 glycogen estimation or TDEE adaptation
        st.session_state.stage1_typical_indoor_temp_f = s_typical_indoor_temp_f
        st.session_state.stage1_minutes_cold_exposure_daily = s_minutes_cold_exposure_daily
        st.session_state.stage1_avg_sleep_hours = s_avg_sleep_hours
        st.session_state.stage1_uses_caffeine = s_uses_caffeine
        st.session_state.stage1_has_fever_illness = s_has_fever_illness
        st.session_state.stage1_peak_fever_temp_f_input = s_peak_fever_temp_f_input

# else: # This was for the main button click, handled inside the button logic now
#     st.info("👈 Please fill in your details in the sidebar and click 'Calculate & Simulate TDEE (Stage 1)'.")

# --- Placeholder for Stage 2 UI and Logic (to be built in subsequent chunks) ---
# This section will only appear if Stage 1 results have been calculated.
# if st.session_state.get('stage1_results_calculated', False):
#    st.header("🚀 Stage 2: Bulk/Cut Forecast & Simulation")
#    # UI for Stage 2 inputs will go here
#    # Call to Stage 2 simulation function
#    # Display Stage 2 results (plots, etc.)
# --- (Continuing from Chunk 5, after the Stage 1 "Analysis Complete!" message and disclaimers) ---

# --- Stage 2: Bulk/Cut Forecast UI ---
if st.session_state.get('stage1_results_calculated', False):
    st.markdown("---") # Visual separator
    st.header("🚀 Stage 2: Bulk/Cut Forecast & Simulation")
    st.markdown("Based on your Stage 1 analysis, plan your next phase.")

    # Retrieve necessary values from Stage 1 stored in session_state
    s1_final_tdee = st.session_state.get('stage1_final_tdee', 2500)
    s1_ffm_kg = st.session_state.get('stage1_ffm_kg', 60)
    s1_fm_kg = st.session_state.get('stage1_fm_kg', 15)
    s1_weight_kg = st.session_state.get('stage1_weight_kg', 75)
    s1_initial_bmr_baseline = st.session_state.get('stage1_initial_bmr_baseline', 1600)
    s1_UAB = st.session_state.get('stage1_UAB', 2800) # Upper Adaptive Bound from Stage 1 (DLW TDEE)
    s1_is_critically_low_intake = st.session_state.get('stage1_is_critically_low_intake', False)
    s1_sex = st.session_state.get('stage1_sex', "Male")
    s1_height_cm = st.session_state.get('stage1_height_cm', 178.0)
    s1_age_years = st.session_state.get('stage1_age_years', 25)
    
    # Current activity levels from Stage 1 (to carry over as default for Stage 2 simulation if not re-inputted)
    s1_avg_daily_steps = st.session_state.get("stage1_avg_daily_steps", 7500)
    s1_other_exercise_kcal = st.session_state.get("stage1_other_exercise_kcal", 0)


    st.subheader("🎯 Set Your Goal for Stage 2")

    s2_goal_options = ["Bulk (Gain Weight/Muscle)", "Cut (Lose Weight/Fat)", "Maintain (Re-evaluate Current TDEE)"]
    # Initialize stage2_goal if it's not set or if stage1 results were just recalculated
    if 'stage2_goal' not in st.session_state or st.session_state.stage1_results_calculated: # Reset if stage 1 re-run
        st.session_state.stage2_goal = "Bulk (Gain Weight/Muscle)" if s1_final_tdee < UAB * 0.95 else "Cut (Lose Weight/Fat)"


    st.session_state.stage2_goal = st.radio(
        "What is your primary goal for this phase?",
        s2_goal_options,
        key="stage2_goal_radio", # Using a distinct key from the session state variable for radio
        index=s2_goal_options.index(st.session_state.stage2_goal)
    )

    # Inputs for Rate and Duration
    rate_col, duration_col = st.columns(2)
    s2_rate_unit = st.session_state.get("weight_unit", "lbs") # Use same unit as Stage 1 for consistency

    with rate_col:
        if st.session_state.stage2_goal.startswith("Bulk"):
            default_rate = 0.5 if s2_rate_unit == "lbs" else 0.23
            min_rate, max_rate, step_rate = (0.1, 2.0, 0.1) if s2_rate_unit == "lbs" else (0.05, 1.0, 0.05)
            help_text_rate = f"Recommended safe lean bulking rates are typically 0.25-0.5 {s2_rate_unit}/week for most. Higher rates risk more fat gain."
        elif st.session_state.stage2_goal.startswith("Cut"):
            default_rate = 1.0 if s2_rate_unit == "lbs" else 0.45
            min_rate, max_rate, step_rate = (0.25, 3.0, 0.05) if s2_rate_unit == "lbs" else (0.1, 1.5, 0.025) # Allow more aggressive for cut
            help_text_rate = f"Sustainable fat loss rates are often 0.5-2.0 {s2_rate_unit}/week. Faster rates (>2 {s2_rate_unit}/wk or >1% bodyweight/wk) increase risk of muscle loss."
        else: # Maintain
            default_rate = 0.0
            min_rate, max_rate, step_rate = (0.0, 0.0, 0.01) # Rate is effectively zero
            help_text_rate = "For maintenance, the goal is to keep weight stable (rate of change = 0)."

        st.session_state.stage2_rate_val = st.number_input(
            f"Desired Rate of Weight Change ({s2_rate_unit}/week):",
            min_value=min_rate, max_value=max_rate, 
            value=st.session_state.get("stage2_rate_val", default_rate), # Use .get for robustness
            step=step_rate, format="%.2f", 
            key="stage2_rate_val_key", # Distinct key for widget
            help=help_text_rate,
            disabled=(st.session_state.stage2_goal.startswith("Maintain"))
        )
        # Ensure rate is positive for bulk/cut for calculations, user inputs absolute rate
        s2_rate_for_calc = abs(st.session_state.stage2_rate_val) if not st.session_state.stage2_goal.startswith("Maintain") else 0.0


    with duration_col:
        st.session_state.stage2_duration_weeks = st.number_input(
            "Duration of this Phase (weeks):",
            min_value=1, max_value=52, 
            value=st.session_state.get("stage2_duration_weeks", 8), 
            step=1, key="stage2_duration_weeks_key",
            help="How long do you plan to adhere to this bulk/cut phase?",
            disabled=(st.session_state.stage2_goal.startswith("Maintain"))
        )

    # Calculate Target Daily Calories for Stage 2
    s2_target_daily_kcal = s1_final_tdee # Default to current TDEE (maintenance)
    kcal_change_for_rate = (s2_rate_for_calc * (3500 if s2_rate_unit == "lbs" else KCAL_PER_KG_TISSUE)) / 7.0

    if st.session_state.stage2_goal.startswith("Bulk"):
        # User wants to bulk from UAB (DLW TDEE) as per earlier request
        # However, UAB itself is an *output* of maintaining a composition.
        # More practically, bulking surplus is added to current adapted TDEE or a target like UAB.
        # Let's use the definition: Target Intake = UAB_from_Stage1_composition + Surplus_for_Gain
        # UAB_sim was calculated in Stage 1 based on ffm_kg, fm_kg from user inputs.
        target_maintenance_tdee_for_bulk = st.session_state.get("stage1_UAB", s1_final_tdee) # Use UAB if available and reasonable
        if target_maintenance_tdee_for_bulk < s1_final_tdee: # If UAB is lower than current adapted TDEE, use current TDEE
            target_maintenance_tdee_for_bulk = s1_final_tdee
        s2_target_daily_kcal = target_maintenance_tdee_for_bulk + kcal_change_for_rate
        st.sidebar.caption(f"Bulking target: adding {kcal_change_for_rate:,.0f} kcal/day to an estimated upper maintenance TDEE of {target_maintenance_tdee_for_bulk:,.0f} kcal.")

    elif st.session_state.stage2_goal.startswith("Cut"):
        # User wants to cut from "Lowest End TDEE" which is RMR*1.2 + EAT + TEF.
        # The simulation will drive TDEE down. We set the intake.
        # TDEE floor for BMR+NEAT is initial_bmr_baseline * 1.2.
        # The EAT and TEF will depend on the *new cutting intake and activity*.
        # So, the target intake is s1_final_tdee - kcal_change_for_rate, and TDEE will adapt.
        s2_target_daily_kcal = s1_final_tdee - kcal_change_for_rate
        st.sidebar.caption(f"Cutting target: subtracting {kcal_change_for_rate:,.0f} kcal/day from current adapted TDEE of {s1_final_tdee:,.0f} kcal.")
    
    st.session_state.s2_target_daily_kcal = s2_target_daily_kcal # Store for macro sliders

    st.subheader("🥗 Target Macronutrient Split for Stage 2 Diet")
    st.markdown(f"Your target daily intake for Stage 2 is estimated at **`{s2_target_daily_kcal:,.0f} kcal/day`**. "
                "Please set your target macronutrient breakdown in grams. The tool will help you align this with your total calorie target.")

    # Initialize macro session state keys if they don't exist
    if 's2_protein_g' not in st.session_state: st.session_state.s2_protein_g = s1_protein_g_per_day # Default to Stage 1 protein
    if 's2_carb_g' not in st.session_state: st.session_state.s2_carb_g = (s2_target_daily_kcal * 0.45) / 4 # Default 45% Carbs
    if 's2_fat_g' not in st.session_state: st.session_state.s2_fat_g = (s2_target_daily_kcal * 0.25) / 9 # Default 25% Fat
    
    # Ensure defaults align somewhat with new s2_target_daily_kcal if it's first time or goal changed
    if st.session_state.get("stage2_macros_need_reinit", True) or st.session_state.stage2_goal != st.session_state.get("prev_stage2_goal_for_macros",""):
        # A simple re-distribution if target kcal changes significantly or goal changes
        protein_kcal_s2 = st.session_state.s2_protein_g * 4
        remaining_kcal_s2 = s2_target_daily_kcal - protein_kcal_s2
        if remaining_kcal_s2 < 0: remaining_kcal_s2 = 0 # Avoid negative

        st.session_state.s2_carb_g = (remaining_kcal_s2 * 0.65) / 4 # e.g. 65% of remaining for carbs
        st.session_state.s2_fat_g = (remaining_kcal_s2 * 0.35) / 9  # e.g. 35% for fat
        st.session_state.stage2_macros_need_reinit = False
        st.session_state.prev_stage2_goal_for_macros = st.session_state.stage2_goal


    max_protein_g = s_weight_kg * 3.0 # Max reasonable protein
    max_carb_g = s2_target_daily_kcal / 4 # Theoretical max if all cals were carbs
    max_fat_g = s2_target_daily_kcal / 9  # Theoretical max if all cals were fat

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.session_state.s2_protein_g = st.slider("Target Protein (g/day):", 0.0, max_protein_g, st.session_state.s2_protein_g, 1.0, key="s2_protein_g_slider")
    with m_col2:
        st.session_state.s2_carb_g = st.slider("Target Carbohydrates (g/day):", 0.0, max_carb_g, st.session_state.s2_carb_g, 1.0, key="s2_carb_g_slider")
    with m_col3:
        st.session_state.s2_fat_g = st.slider("Target Fat (g/day):", 0.0, max_fat_g, st.session_state.s2_fat_g, 1.0, key="s2_fat_g_slider")

    # Display current macro calorie total and difference from target
    current_macro_kcal = (st.session_state.s2_protein_g * 4) + \
                         (st.session_state.s2_carb_g * 4) + \
                         (st.session_state.s2_fat_g * 9)
    kcal_difference = current_macro_kcal - s2_target_daily_kcal
    
    st.info(f"Calories from Macros: `{current_macro_kcal:,.0f} kcal` | Target: `{s2_target_daily_kcal:,.0f} kcal` | Difference: `{kcal_difference:+.0f} kcal`")
    if abs(kcal_difference) > 50:
        st.warning("Adjust macronutrient sliders so their total calories closely match your target daily intake.")

    # Inputs for Bulking Phase
    if st.session_state.stage2_goal.startswith("Bulk"):
        st.subheader(f"🏋️‍♂️ Bulking Phase Considerations {INFO_ICON}", help="These factors influence how gained weight is partitioned into muscle vs. fat.")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.session_state.stage2_training_status = b_col1.selectbox(
                "Your Resistance Training Status:",
                ("Novice (Less than 1 year consistent training)", 
                 "Intermediate (1-3 years consistent training)", 
                 "Advanced (3+ years consistent training)"),
                key="stage2_training_status_key",
                index=["Novice", "Intermediate", "Advanced"].index(st.session_state.get("stage2_training_status", "Novice"))
            )
        with b_col2:
            st.session_state.stage2_weightlifting = b_col2.checkbox(
                "Actively Weightlifting/Resistance Training during this Bulk?", 
                value=st.session_state.get("stage2_weightlifting", True), 
                key="stage2_weightlifting_key"
            )
# --- (Continuing from previous chunks. Assume all helper and Stage 1 functions are defined above) ---

# --- Stage 2: Bulk/Cut Forecast Simulation Function ---
def simulate_bulk_cut_forecast(stage1_data, stage2_inputs, num_forecast_days):
    """
    Simulates weight, body composition, and TDEE changes during a bulk or cut phase.
    """
    # Unpack relevant data from Stage 1
    current_weight_kg = stage1_data['weight_kg']
    current_ffm_kg = stage1_data['ffm_kg']
    current_fm_kg = stage1_data['fm_kg']
    initial_bmr_baseline = stage1_data['initial_bmr_baseline'] # Pontzer FFM-RMR from Stage 1
    # Use Stage 1's adapted TDEE components as starting points for Stage 2 adaptation if available
    # For simplicity, we'll re-initialize adaptive BMR and NEAT based on the new intake context for Stage 2.
    # However, the initial_bmr_baseline is crucial.
    s1_is_critically_low_intake = stage1_data.get('is_critically_low_intake_scenario', False)
    s1_intake = stage1_data.get('adjusted_intake', initial_bmr_baseline * 1.5) # Fallback for s1 intake

    # Carry over environmental/physiological factors from Stage 1 inputs, assuming they persist
    # unless we add UI for user to change them for Stage 2
    sim_inputs_for_components = {
        "weight_kg": current_weight_kg, # This will update daily
        "body_fat_percentage": (current_fm_kg / current_weight_kg) * 100 if current_weight_kg > 0 else 0, # Updates daily
        "sex": stage1_data['sex'], 
        "age_years": stage1_data['age_years'], 
        "height_cm": stage1_data['height_cm'],
        "avg_daily_steps": stage1_data.get('avg_daily_steps', 7500), # Assume stage 1 activity persists
        "other_exercise_kcal_per_day": stage1_data.get('other_exercise_kcal', 0),
        "typical_indoor_temp_f": stage1_data.get('typical_indoor_temp_f', 70),
        "minutes_cold_exposure_daily": stage1_data.get('minutes_cold_exposure_daily',0),
        "avg_sleep_hours": stage1_data.get('avg_sleep_hours', 7.5),
        "uses_caffeine": stage1_data.get('uses_caffeine', True),
        "has_fever_illness": stage1_data.get('has_fever_illness', False), # Assume no fever for forecast period
        "peak_fever_temp_f": stage1_data.get('peak_fever_temp_f_input', 98.6),
        "streamlit_object": stage1_data.get("streamlit_object", None) # For warnings
    }

    # Unpack Stage 2 goal parameters
    goal = stage2_inputs['goal'] # "Bulk", "Cut", "Maintain"
    target_intake_s2 = stage2_inputs['target_daily_kcal']
    target_protein_g_s2 = stage2_inputs['protein_g']
    target_carb_g_s2 = stage2_inputs['carb_g']
    target_fat_g_s2 = stage2_inputs['fat_g']
    
    # EAT calculation for Stage 2 (based on Stage 1 activity inputs, as Stage 2 activity inputs are not yet defined)
    # This assumes activity level (steps, other exercise) remains consistent from Stage 1.
    # If you want to allow users to change activity for Stage 2, add those inputs.
    kcal_per_step_s2 = KCAL_PER_STEP_BASE_FACTOR * current_weight_kg # Initial, will update with weight
    eat_kcal_steps_s2 = sim_inputs_for_components['avg_daily_steps'] * kcal_per_step_s2
    eat_kcal_additional_exercise_s2 = sim_inputs_for_components['other_exercise_kcal_per_day']
    current_eat_kcal = eat_kcal_steps_s2 + eat_kcal_additional_exercise_s2


    # --- Glycogen Modeling Initialization ---
    max_glycogen_capacity_g = (current_ffm_kg * GLYCOGEN_G_PER_KG_FFM_MUSCLE) + LIVER_GLYCOGEN_CAPACITY_G
    
    # Infer initial glycogen state based on Stage 1
    # If critically low intake in S1 OR S1 intake was significantly below S1 TDEE (e.g. >500kcal deficit)
    s1_final_tdee = stage1_data.get('final_tdee', initial_bmr_baseline * 1.5) # Get S1 final TDEE
    if s1_is_critically_low_intake or (s1_intake < s1_final_tdee - 400): # Significant deficit
        current_glycogen_g = max_glycogen_capacity_g * 0.25 # Start at 25% if depleted
    elif s1_intake > s1_final_tdee + 400: # Significant surplus in S1
        current_glycogen_g = max_glycogen_capacity_g * 0.85 # Start at 85% if was in surplus
    else: # Near maintenance or moderate deficit/surplus
        current_glycogen_g = max_glycogen_capacity_g * 0.60 # Start at 60%

    current_glycogen_g = np.clip(current_glycogen_g, 0, max_glycogen_capacity_g)


    # --- Initialize TDEE components for Stage 2 adaptation ---
    # Start with BMR from end of Stage 1 or re-evaluate based on initial_bmr_baseline
    # For Stage 2, let BMR and NEAT re-adapt from baseline based on the new sustained intake.
    current_bmr_adaptive = initial_bmr_baseline # Start from the true baseline RMR
    current_neat_adaptive_component = 0.0 # Reset adaptive NEAT for new phase

    # Physiological limits for BMR adaptation (relative to the *initial* baseline RMR of the person)
    min_bmr_limit_factor = 0.80; max_bmr_limit_factor = 1.15 
    min_bmr_target_limit = initial_bmr_baseline * min_bmr_limit_factor
    max_bmr_target_limit = initial_bmr_baseline * max_bmr_limit_factor
    min_neat_limit = -500; max_neat_limit = 750 # NEAT component limits (can be more aggressive for NEAT)

    # Check if new Stage 2 intake is critically low
    CRITICAL_LOW_INTAKE_THRESHOLD_S2 = initial_bmr_baseline * 1.35
    is_critically_low_intake_s2 = target_intake_s2 < CRITICAL_LOW_INTAKE_THRESHOLD_S2
    
    # Define adaptive bounds for Stage 2 based on *current* (start of Stage 2) body composition
    s2_LAB = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
    s2_UAB = calculate_dlw_tdee(current_ffm_kg, current_fm_kg) # UAB can change as FFM/FM change
    pal_for_s2_uab_heuristic = get_pal_multiplier_for_heuristic(sim_inputs_for_components['avg_daily_steps'])
    if s2_UAB == 0 or s2_UAB < s2_LAB * 1.05:
        s2_UAB = initial_bmr_baseline * pal_for_s2_uab_heuristic * 1.05

    s2_intake_in_adaptive_range = s2_LAB <= target_intake_s2 <= s2_UAB
    
    # Day 0 TDEE for Stage 2 (before adaptation to Stage 2 intake begins for BMR/NEAT)
    # Use current_bmr_adaptive and current_neat_adaptive_component as they are at start of Stage 2
    day0_s2_tef = calculate_tef(target_intake_s2, target_protein_g_s2)
    day0_s2_cold = calculate_cold_thermogenesis(sim_inputs_for_components['typical_indoor_temp_f'], sim_inputs_for_components['minutes_cold_exposure_daily'])
    day0_s2_fever = calculate_immune_fever_effect(sim_inputs_for_components['has_fever_illness'], sim_inputs_for_components['peak_fever_temp_f'], current_bmr_adaptive)
    TDEE_s2_sim_start = current_bmr_adaptive + day0_s2_tef + current_eat_kcal + current_neat_adaptive_component + day0_s2_cold + day0_s2_fever


    s2_daily_log = []
    current_weight_for_sim = current_weight_kg # Track weight changes during simulation

    # Muscle gain rates (kg/day)
    lbs_to_kg_conversion = 0.453592
    days_in_month = 30.4375
    muscle_gain_rate_kg_day = {
        "Novice": (MUSCLE_GAIN_NOVICE_LBS_MONTH * lbs_to_kg_conversion) / days_in_month,
        "Intermediate": (MUSCLE_GAIN_INTERMEDIATE_LBS_MONTH * lbs_to_kg_conversion) / days_in_month,
        "Advanced": (MUSCLE_GAIN_ADVANCED_LBS_MONTH * lbs_to_kg_conversion) / days_in_month,
    }
    
    # Carb oxidation estimation (simplified)
    # Base + activity-driven. Activity here is EAT.
    # For simplicity, assume a % of non-protein, non-fat energy is from carbs, or a fixed amount.
    # This needs more nuance if we want to be precise.
    # Let's assume ~50g baseline + EAT / 10 (rough estimate: 10kcal/g carb for intense exercise)
    # Or, more simply, a % of (TDEE - protein_kcal - fat_kcal)
    # For now, let's directly use user's target_carb_g_s2 for daily carb intake.

    # Store initial values for plots
    s2_daily_log.append({
            "Day": 0, "Weight_kg": current_weight_kg, "FFM_kg": current_ffm_kg, "FM_kg": current_fm_kg,
            "Glycogen_g": current_glycogen_g, "Max_Glycogen_g": max_glycogen_capacity_g,
            "BMR_Adaptive": current_bmr_adaptive, "NEAT_Adaptive": current_neat_adaptive_component,
            "EAT": current_eat_kcal, "TEF": day0_s2_tef, "Total_Dynamic_TDEE": TDEE_s2_sim_start,
            "Target_Intake_s2": target_intake_s2,
            "Energy_Balance_Daily": target_intake_s2 - TDEE_s2_sim_start, # Initial balance
            "Delta_FFM_kg": 0, "Delta_FM_kg": 0, "Delta_Glycogen_g":0
    })
    
    # Simulation Loop for Stage 2
    for day in range(int(num_forecast_days)): # num_forecast_days is duration_weeks * 7
        # Update weight-dependent EAT factor (kcal_per_step)
        kcal_per_step_s2 = KCAL_PER_STEP_BASE_FACTOR * current_weight_for_sim
        eat_kcal_steps_s2 = sim_inputs_for_components['avg_daily_steps'] * kcal_per_step_s2
        current_eat_kcal = eat_kcal_steps_s2 + sim_inputs_for_components['other_exercise_kcal_per_day']

        # TEF based on the fixed Stage 2 target intake and protein
        tef_s2 = calculate_tef(target_intake_s2, target_protein_g_s2)
        
        # Other fixed components for the day (cold, fever)
        cold_s2 = cold_kcal_fixed # Assumed constant from S1 inputs for now
        fever_s2 = calculate_immune_fever_effect(sim_inputs_for_components['has_fever_illness'], sim_inputs_for_components['peak_fever_temp_f'], current_bmr_adaptive)


        # --- TDEE Adaptation (BMR & NEAT) for Stage 2 ---
        # This logic mirrors Stage 1's adaptation but driven by Stage 2's target_intake_s2
        # and current body composition state
        day_target_bmr_s2 = current_bmr_adaptive
        day_target_neat_s2 = current_neat_adaptive_component

        # Recalculate LAB/UAB based on potentially changed FFM/FM if doing daily body comp updates
        # For simplicity, let's assume LAB/UAB (and thus intake_in_adaptive_range) are based on start-of-stage2 comp for now.
        # A more advanced model would update LAB/UAB daily.
        # For this iteration, we'll use the initial s2_LAB, s2_UAB, s2_intake_in_adaptive_range for target setting style.
        
        # Re-evaluate if intake is critically low based on current BMR baseline
        current_bmr_baseline_for_crit_check = calculate_pontzer_ffm_based_rmr(current_ffm_kg, current_fm_kg)
        if current_bmr_baseline_for_crit_check == 0: current_bmr_baseline_for_crit_check = initial_bmr_baseline

        is_critically_low_intake_s2_daily = target_intake_s2 < (current_bmr_baseline_for_crit_check * 1.35)
        current_min_bmr_target = current_bmr_baseline_for_crit_check * min_bmr_limit_factor
        current_max_bmr_target = current_bmr_baseline_for_crit_check * max_bmr_limit_factor


        if is_critically_low_intake_s2_daily:
            day_target_bmr_s2 = current_min_bmr_target
            target_bmr_plus_neat_floor_s2 = current_bmr_baseline_for_crit_check * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
            day_target_neat_s2 = target_bmr_plus_neat_floor_s2 - day_target_bmr_s2
            day_target_neat_s2 = np.clip(day_target_neat_s2, min_neat_limit, 0)
        
        elif s2_intake_in_adaptive_range: # Using UAB/LAB from start of Stage 2
            # TDEE (BMR+NEAT) aims to make Total TDEE = Intake
            # Gap is relative to the TDEE at the start of *this specific Stage 2 phase*
            # This needs to be based on a more stable "unadapted" Day 0 state for Stage 2
            s2_day0_bmr = current_bmr_baseline_for_crit_check # BMR if not adapted to S2 intake yet
            s2_day0_neat = 0 # Assume adaptive NEAT is 0 before adapting to S2 intake

            s2_tdee_start_for_adaptation_calc = s2_day0_bmr + tef_s2 + current_eat_kcal + s2_day0_neat + cold_s2 + fever_s2
            s2_total_adaptation_gap = target_intake_s2 - s2_tdee_start_for_adaptation_calc
            
            s2_neat_share = 0.60 if s2_total_adaptation_gap > 0 else 0.40
            s2_bmr_share = 1.0 - s2_neat_share
            s2_target_total_neat_change = s2_total_adaptation_gap * s2_neat_share
            s2_target_total_bmr_change = s2_total_adaptation_gap * s2_bmr_share
            
            day_target_neat_s2 = np.clip(s2_day0_neat + s2_target_total_neat_change, min_neat_limit, max_neat_limit)
            day_target_bmr_s2 = np.clip(s2_day0_bmr + s2_target_total_bmr_change, current_min_bmr_target, current_max_bmr_target)

        else: # Outside adaptive range, not critically low
            # Energy balance drives BMR/NEAT changes
            expenditure_for_balance_s2 = current_bmr_adaptive + tef_s2 + current_eat_kcal + current_neat_adaptive_component + cold_s2 + fever_s2
            energy_balance_s2 = target_intake_s2 - expenditure_for_balance_s2
            
            bmr_target_change_factor_s2 = 0.0
            if energy_balance_s2 > 250: bmr_target_change_factor_s2 = 0.05
            elif energy_balance_s2 < -250: bmr_target_change_factor_s2 = -0.10
            day_target_bmr_s2 = current_bmr_baseline_for_crit_check * (1 + bmr_target_change_factor_s2)
            day_target_bmr_s2 = np.clip(day_target_bmr_s2, current_min_bmr_target, current_max_bmr_target)

            neat_resp_s2 = 0.30 # Base responsiveness
            # (Add factors like steps, sleep, caffeine from sim_inputs_for_components)
            if sim_inputs_for_components['avg_daily_steps'] > 10000: neat_resp_s2 += 0.10
            # ... etc ...
            neat_resp_s2 = np.clip(neat_resp_s2, 0.1, 0.5)
            day_target_neat_s2 = energy_balance_s2 * neat_resp_s2
            day_target_neat_s2 = np.clip(day_target_neat_s2, min_neat_limit, max_neat_limit)

        # Adaptation step for BMR and NEAT for Stage 2
        # Simplified front-loading for Stage 2: Apply a faster adaptation for the first few days
        s2_current_tau_bmr = TAU_BMR_ADAPTATION * (0.7 if day < 7 else 1.0) # Faster for first week
        s2_current_tau_neat = TAU_NEAT_ADAPTATION * (0.7 if day < 7 else 1.0)

        delta_bmr_s2 = (day_target_bmr_s2 - current_bmr_adaptive) / s2_current_tau_bmr
        current_bmr_adaptive += delta_bmr_s2
        delta_neat_s2 = (day_target_neat_s2 - current_neat_adaptive_component) / s2_current_tau_neat
        current_neat_adaptive_component += delta_neat_s2
        
        current_bmr_adaptive = np.clip(current_bmr_adaptive, current_min_bmr_target, current_max_bmr_target)
        current_neat_adaptive_component = np.clip(current_neat_adaptive_component, min_neat_limit, max_neat_limit)
        
        # Enforce BMR*1.2 floor for BMR+NEAT if critically low intake
        if is_critically_low_intake_s2_daily:
            bmr_plus_neat_target_floor_s2 = current_bmr_baseline_for_crit_check * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR
            current_bmr_adaptive = max(current_bmr_adaptive, current_min_bmr_target)
            required_neat_for_floor_s2 = bmr_plus_neat_target_floor_s2 - current_bmr_adaptive
            current_neat_adaptive_component = max(required_neat_for_floor_s2, min_neat_limit)
            if current_neat_adaptive_component == min_neat_limit: # If NEAT floored out
                current_bmr_adaptive = max(bmr_plus_neat_target_floor_s2 - min_neat_limit, current_min_bmr_target)
            current_bmr_adaptive = np.clip(current_bmr_adaptive, current_min_bmr_target, current_max_bmr_target)

        # Recalculate daily TDEE with adapted BMR/NEAT
        total_tdee_today_s2 = current_bmr_adaptive + tef_s2 + current_eat_kcal + current_neat_adaptive_component + cold_s2 + fever_s2
        
        # Energy balance for tissue change (after TDEE adaptation for the day)
        # This is (Intake - TDEE_final_for_day)
        daily_net_energy_balance = target_intake_s2 - total_tdee_today_s2f
        # --- (Continuing inside the `for day in range(int(num_forecast_days)):` loop in `simulate_bulk_cut_forecast`) ---
# --- (Assume: current_eat_kcal, tef_s2, fever_s2, cold_s2 are calculated for the day)
# --- (Assume: current_bmr_adaptive, current_neat_adaptive_component are updated for the day)
# --- (Assume: total_tdee_today_s2 is calculated for the day)

        # --- Daily Glycogen Dynamics ---
        delta_glycogen_g_today = 0.0
        kcal_for_glycogen_change_today = 0.0

        # Estimate daily carbohydrate needs/oxidation
        # Simplified: base (e.g., brain) + % of non-protein energy expenditure
        # A more direct approach: use target_carb_g_s2 from user input
        carb_intake_g_s2 = stage2_inputs['carb_g'] 
        
        # Very simplified carb oxidation model:
        # Assume a baseline oxidation + some for activity (EAT)
        # This is a major simplification without detailed metabolic pathway modeling.
        # Let's say baseline carb oxidation is ~100-150g.
        # And activity burns carbs at roughly 4-5 kcal/g, so EAT/4.5.
        estimated_carb_oxidation_g = 125.0 + (current_eat_kcal / 4.5)
        
        net_carb_for_glycogen_g = carb_intake_g_s2 - estimated_carb_oxidation_g

        max_glycogen_capacity_g = (current_ffm_kg * GLYCOGEN_G_PER_KG_FFM_MUSCLE) + LIVER_GLYCOGEN_CAPACITY_G
        min_glycogen_physio_floor_g = max_glycogen_capacity_g * 0.15 # Don't allow glycogen to go to absolute zero easily

        if net_carb_for_glycogen_g > 0 and current_glycogen_g < max_glycogen_capacity_g: # Surplus carbs and space to store
            # Glycogen repletion
            can_store_g = max_glycogen_capacity_g - current_glycogen_g
            # Rate of repletion: assume a portion of surplus carbs goes to glycogen,
            # or use a max daily repletion rate (e.g., 5-10g/hr, so 120-240g/day is high but possible with effort)
            # For simplicity with daily step, let's assume available net carbs can be stored up to a reasonable daily cap.
            max_daily_glycogen_gain_g = 150 # Max g of glycogen synthesized per day (tuneable)
            delta_glycogen_g_today = min(net_carb_for_glycogen_g, can_store_g, max_daily_glycogen_gain_g)
            current_glycogen_g += delta_glycogen_g_today
            kcal_for_glycogen_change_today = delta_glycogen_g_today * KCAL_PER_G_CARB_FOR_GLYCOGEN # Energy cost to store
        
        elif net_carb_for_glycogen_g < 0 and current_glycogen_g > min_glycogen_physio_floor_g: # Deficit carbs and stores available
            # Glycogen depletion
            # Amount to deplete is driven by the carb deficit from oxidation needs.
            can_lose_g = current_glycogen_g - min_glycogen_physio_floor_g
            # Deplete by the carb deficit, up to what can be lost, or a max daily rate
            max_daily_glycogen_loss_g = 200 # Tuneable
            delta_glycogen_g_today = max(net_carb_for_glycogen_g, -can_lose_g, -max_daily_glycogen_loss_g) # net_carb is negative
            current_glycogen_g += delta_glycogen_g_today # delta is negative
            # kcal_for_glycogen_change_today is 0 as mobilizing glycogen yields energy, not costs it here.
            # The energy from mobilized glycogen contributes to meeting daily_net_energy_balance.

        current_glycogen_g = np.clip(current_glycogen_g, min_glycogen_physio_floor_g, max_glycogen_capacity_g)
        weight_change_from_glycogen_water_g_today = delta_glycogen_g_today * (1 + WATER_G_PER_G_GLYCOGEN) # 1g glycogen + water

        # --- Energy Balance for Tissue Change ---
        # Total energy balance for the day (Intake - TDEE)
        overall_daily_energy_balance = target_intake_s2 - total_tdee_today_s2
        
        # Adjust this balance for energy used in glycogen synthesis (if any)
        # If glycogen was depleted, the energy from it met some of the TDEE, so the "tissue deficit" is less.
        # If glycogen was synthesized, energy was used, so less is available for tissue.
        energy_balance_for_tissue_today = overall_daily_energy_balance - kcal_for_glycogen_change_today


        # --- Tissue Partitioning (Fat/Lean Mass Changes) ---
        delta_ffm_kg_today = 0.0
        delta_fm_kg_today = 0.0

        if goal.startswith("Bulk"):
            if current_glycogen_g >= max_glycogen_capacity_g * 0.90: # Start tissue gain once glycogen is mostly full
                effective_surplus_for_tissue = energy_balance_for_tissue_today * SURPLUS_EFFICIENCY
                if effective_surplus_for_tissue > 0:
                    target_muscle_gain_kg_day = 0
                    if stage2_inputs.get('weightlifting', False): # Check if key exists
                        status = stage2_inputs.get('training_status', "Novice")
                        if status == "Novice": target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Novice"]
                        elif status == "Intermediate": target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Intermediate"]
                        elif status == "Advanced": target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Advanced"]
                    else: # Not weightlifting, minimal muscle gain from surplus
                        target_muscle_gain_kg_day = (muscle_gain_rate_kg_day["Advanced"] * 0.25) # e.g. 25% of advanced rate

                    # Energy cost of FFM gain (protein is ~4kcal/g, but anabolism cost) ~2500-2800 kcal/lb FFM, so ~5500-6200 kcal/kg FFM
                    kcal_cost_per_kg_ffm = 5800 # Average estimate
                    kcal_for_target_ffm_gain = target_muscle_gain_kg_day * kcal_cost_per_kg_ffm
                    
                    actual_kcal_to_ffm = min(kcal_for_target_ffm_gain, effective_surplus_for_tissue * 0.75) # Cap FFM gain by a portion of surplus
                    delta_ffm_kg_today = actual_kcal_to_ffm / kcal_cost_per_kg_ffm
                    
                    remaining_surplus_for_fat = effective_surplus_for_tissue - actual_kcal_to_ffm
                    delta_fm_kg_today = remaining_surplus_for_fat / KCAL_PER_KG_TISSUE # Fat gain from remaining surplus
                    delta_fm_kg_today = max(0, delta_fm_kg_today) # Cannot lose fat in surplus here

        elif goal.startswith("Cut"):
            effective_deficit_for_tissue = energy_balance_for_tissue_today * DEFICIT_EFFICIENCY # Will be negative
            if effective_deficit_for_tissue < 0:
                # Determine current weekly loss rate to choose partition ratio
                # For simplicity, use the target rate from Stage 2 inputs for partition logic.
                # A more dynamic approach would be to track actual loss rate over last 7 days of sim.
                target_weekly_loss_rate_lbs = stage2_inputs['rate_val']
                if stage2_inputs['rate_unit'] == "kg": target_weekly_loss_rate_lbs = kg_to_lbs(target_weekly_loss_rate_lbs)

                if target_weekly_loss_rate_lbs <= 2.0: # Slower loss
                    ffm_loss_percentage = 0.25
                else: # Faster loss
                    ffm_loss_percentage = 0.50
                
                kcal_loss_from_ffm = abs(effective_deficit_for_tissue) * ffm_loss_percentage
                kcal_loss_from_fm = abs(effective_deficit_for_tissue) * (1.0 - ffm_loss_percentage)
                
                # kcal_cost_per_kg_ffm for loss can also be estimated around 1200-1800 kcal/kg FFM (protein content)
                # Or assume similar energy density to fat for simplification in deficit if not specified otherwise.
                # For simplicity of energy accounting for loss from FFM (it's not "burned" the same way fat is)
                # Let's assume protein component of FFM is ~4kcal/g. 1kg FFM has ~200g protein. So ~800 kcal/kg FFM.
                # This is very simplified. A better model uses nitrogen balance.
                # For now, let's use a higher effective energy density for FFM loss to make it "harder" to lose.
                kcal_density_ffm_loss = 1800 # kcal/kg, representing the usable energy if LBM is catabolized.
                # --- (Continuing inside the `simulate_bulk_cut_forecast` function, within the daily loop from Chunk 8) ---
# --- (Assume: all daily calculations up to `energy_balance_for_tissue_today` are complete)
# --- (Assume: current_weight_for_sim, current_ffm_kg, current_fm_kg are being updated)

        # --- Tissue Partitioning (Fat/Lean Mass Changes) ---
        delta_ffm_kg_today = 0.0
        delta_fm_kg_today = 0.0
        kcal_to_ffm_today = 0.0
        kcal_to_fm_today = 0.0

        if goal.startswith("Bulk"):
            # Only attempt tissue gain if glycogen stores are reasonably full (e.g., >80% of current max)
            # and there's an actual energy surplus available for tissue.
            if current_glycogen_g >= (max_glycogen_capacity_g * 0.80) and energy_balance_for_tissue_today > 0:
                effective_surplus_for_tissue_kcal = energy_balance_for_tissue_today * SURPLUS_EFFICIENCY
                
                target_muscle_gain_kg_day = 0.0
                if stage2_inputs.get('weightlifting', False): # Check if weightlifting key exists and is True
                    status = stage2_inputs.get('training_status', "Novice")
                    if status == "Novice (Less than 1 year consistent training)": target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Novice"]
                    elif status == "Intermediate (1-3 years consistent training)": target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Intermediate"]
                    elif status == "Advanced (3+ years consistent training)": target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Advanced"]
                else: # Not weightlifting, assume minimal muscle gain from surplus
                    target_muscle_gain_kg_day = (muscle_gain_rate_kg_day["Advanced"] * 0.20) # e.g. 20% of advanced rate as a low baseline

                # Energy cost of FFM gain (protein is ~4kcal/g, but anabolism cost)
                # Using ~5800 kcal/kg FFM gained (includes energy for protein itself + synthesis cost)
                kcal_cost_per_kg_ffm_gain = 5800 
                kcal_for_target_ffm_gain_today = target_muscle_gain_kg_day * kcal_cost_per_kg_ffm_gain
                
                # Actual FFM gain is limited by both target rate and available effective surplus
                # Let's cap energy going to FFM to, say, 75% of the effective surplus to ensure some fat gain occurs too
                # or that FFM gain doesn't "consume" all surplus if it's small.
                kcal_to_ffm_today = min(kcal_for_target_ffm_gain_today, effective_surplus_for_tissue_kcal * 0.75)
                kcal_to_ffm_today = min(kcal_to_ffm_today, effective_surplus_for_tissue_kcal) # Cannot be more than total effective surplus

                delta_ffm_kg_today = kcal_to_ffm_today / kcal_cost_per_kg_ffm_gain
                
                remaining_surplus_for_fat_kcal = effective_surplus_for_tissue_kcal - kcal_to_ffm_today
                delta_fm_kg_today = remaining_surplus_for_fat_kcal / KCAL_PER_KG_TISSUE 
                delta_fm_kg_today = max(0, delta_fm_kg_today) # Ensure fat mass doesn't decrease in surplus

        elif goal.startswith("Cut"):
            effective_deficit_for_tissue_kcal = energy_balance_for_tissue_today * DEFICIT_EFFICIENCY # Will be negative
            if effective_deficit_for_tissue_kcal < 0: # If there's an actual deficit for tissue
                # Determine current weekly loss rate to choose partition ratio
                # Use the target rate from Stage 2 inputs for partition logic.
                target_weekly_loss_rate_lbs = stage2_inputs['rate_val'] # User inputs absolute rate
                # If rate_unit was kg, it should be converted to lbs for this logic if necessary
                # (Assuming rate_val is already in lbs/wk or converted before calling this function)
                # Or, more simply, use the kcal deficit to define "aggressiveness"
                
                daily_deficit_for_partitioning = abs(effective_deficit_for_tissue_kcal)
                # If daily deficit > 700 kcal (i.e., > 4900/wk, approaching 1.5-2lb/wk loss) -> more aggressive
                if daily_deficit_for_partitioning * 7 > (2.0 * 3500): # Corresponds to > 2 lbs/wk
                    ffm_loss_percentage_of_tissue_deficit = 0.50
                else: 
                    ffm_loss_percentage_of_tissue_deficit = 0.25
                
                kcal_lost_as_ffm = daily_deficit_for_partitioning * ffm_loss_percentage_of_tissue_deficit
                kcal_lost_as_fm = daily_deficit_for_partitioning * (1.0 - ffm_loss_percentage_of_tissue_deficit)
                
                # Energy density of FFM loss (protein + associated water, effectively ~1200-1800 kcal/kg of "dry" FFM loss)
                # To lose 1kg of actual FFM tissue (which is ~20% protein), the caloric equivalent is complex.
                # Let's use a simplified energy equivalent for FFM loss.
                kcal_density_ffm_loss = 1800 # kcal per kg of FFM lost (tuneable heuristic)
                
                delta_ffm_kg_today = -(kcal_lost_as_ffm / kcal_density_ffm_loss) 
                delta_fm_kg_today = -(kcal_lost_as_fm / KCAL_PER_KG_TISSUE)   
        
        elif goal.startswith("Maintain"): # Small adjustments around maintenance
            if abs(energy_balance_for_tissue_today) > 30: # Only if balance is somewhat significant
                 if energy_balance_for_tissue_today > 0: # Slight surplus
                     delta_ffm_kg_today = (energy_balance_for_tissue_today * 0.30 * SURPLUS_EFFICIENCY) / 5800 
                     delta_fm_kg_today = (energy_balance_for_tissue_today * 0.70 * SURPLUS_EFFICIENCY) / KCAL_PER_KG_TISSUE
                 else: # Slight deficit
                     delta_ffm_kg_today = (energy_balance_for_tissue_today * 0.20 * DEFICIT_EFFICIENCY) / 1800 
                     delta_fm_kg_today = (energy_balance_for_tissue_today * 0.80 * DEFICIT_EFFICIENCY) / KCAL_PER_KG_TISSUE

        # --- Update Body Composition and Weight for next day's simulation step ---
        current_ffm_kg += delta_ffm_kg_today
        current_fm_kg += delta_fm_kg_today
        
        # Ensure FM and FFM don't go below physiologically plausible minimums
        min_physio_fm_kg = current_weight_for_sim * (0.03 if sim_inputs_for_components['sex']=="Male" else 0.10) 
        current_fm_kg = max(current_fm_kg, min_physio_fm_kg)
        # Ensure FFM doesn't drop unrealistically low (e.g. < 25% of total body weight for an adult)
        min_physio_ffm_kg = current_weight_for_sim * 0.25 if current_weight_for_sim * 0.25 > 20 else 20 
        current_ffm_kg = max(current_ffm_kg, min_physio_ffm_kg)
        
        # Recalculate weight based on FFM, FM, and current glycogen + water
        # Weight change from tissue already applied to current_ffm_kg, current_fm_kg
        # Glycogen weight includes water: current_glycogen_g * (1 + WATER_G_PER_G_GLYCOGEN) / 1000 to get kg
        glycogen_plus_water_kg_today = current_glycogen_g * (1 + WATER_G_PER_G_GLYCOGEN) / 1000.0
        
        # The weight change is sum of tissue change and glycogen/water change
        # Note: delta_ffm and delta_fm are changes in actual tissue mass.
        # Glycogen is part of FFM, but its water is separate for weight fluctuation.
        # A more precise way: New Weight = New FFM (dry) + New FM + Glycogen + Glycogen Water
        # For simplicity, FFM change here is "dry" FFM. Glycogen is tracked separately.
        
        # Track daily change in actual FFM and FM tissue
        daily_ffm_tissue_change = delta_ffm_kg_today
        daily_fm_tissue_change = delta_fm_kg_today
        
        # Total weight is FFM + FM + (glycogen_g + glycogen_water_g)/1000
        # Since current_ffm_kg already changed by delta_ffm_kg_today, and same for fm:
        # Need to be careful not to double count.
        # Let's track "dry" FFM and FM, and total weight is sum of these + glycogen&water mass
        
        # Previous day's glycogen + water mass
        if day == 0:
            prev_glycogen_plus_water_kg = s2_daily_log[0]["Glycogen_g"] * (1 + WATER_G_PER_G_GLYCOGEN) / 1000.0
        else:
            prev_glycogen_plus_water_kg = s2_daily_log[-1]["Glycogen_g"] * (1 + WATER_G_PER_G_GLYCOGEN) / 1000.0
            
        # Change in weight from glycogen+water shift this day
        delta_weight_from_glycogen_shift = glycogen_plus_water_kg_today - prev_glycogen_plus_water_kg
        
        current_weight_for_sim += delta_ffm_kg_today + delta_fm_kg_today + delta_weight_from_glycogen_shift


        # Update weight in sim_inputs_for_components for next day's EAT kcal_per_step factor
        sim_inputs_for_components['weight_kg'] = current_weight_for_sim
        if current_weight_for_sim > 0:
             sim_inputs_for_components['body_fat_percentage'] = (current_fm_kg / current_weight_for_sim) * 100
        else:
             sim_inputs_for_components['body_fat_percentage'] = 0


        # Log daily values for Stage 2
        s2_daily_log.append({
            "Day": day + 1, 
            "Weight_kg": current_weight_for_sim, 
            "FFM_kg": current_ffm_kg, # This is "dry" FFM excluding glycogen for this log
            "FM_kg": current_fm_kg,
            "Glycogen_g": current_glycogen_g, 
            "Max_Glycogen_g": max_glycogen_capacity_g,
            "BMR_Adaptive": current_bmr_adaptive, 
            "NEAT_Adaptive": current_neat_adaptive_component,
            "EAT": current_eat_kcal, 
            "TEF": tef_s2, 
            "Total_Dynamic_TDEE": total_tdee_today_s2,
            "Target_Intake_s2": target_intake_s2,
            "Energy_Balance_Daily_vs_TDEE": daily_net_energy_balance, # Intake - TDEE
            "Energy_For_Tissue_Change_kcal": energy_balance_for_tissue_today,
            "Delta_FFM_kg_Tissue": daily_ffm_tissue_change, 
            "Delta_FM_kg_Tissue": daily_fm_tissue_change,
            "Delta_Glycogen_g": delta_glycogen_g_today
        })
    # End of daily simulation loop for Stage 2

    s2_final_states = {
        "final_weight_kg": current_weight_for_sim,
        "final_ffm_kg": current_ffm_kg,
        "final_fm_kg": current_fm_kg,
        "final_glycogen_g": current_glycogen_g,
        "final_tdee_s2": s2_daily_log[-1]['Total_Dynamic_TDEE'] if s2_daily_log else TDEE_s2_sim_start,
        "final_bmr_adaptive_s2": current_bmr_adaptive,
        "final_neat_adaptive_s2": current_neat_adaptive_component
    }
    return pd.DataFrame(s2_daily_log), s2_final_states

# --- (UI part for calling Stage 2 simulation will be in the next chunk) ---
# --- (Continuing from Chunk 9 - all previous functions are assumed to be defined above) ---

# --- Streamlit App UI (Main Body - Continuation for Stage 2) ---

# This block is conditional on Stage 1 being completed
if st.session_state.get('stage1_results_calculated', False):
    st.markdown("---") 
    st.header("🚀 Stage 2: Bulk/Cut Forecast & Simulation")
    st.markdown("Set your parameters for the planned bulk or cut phase.")

    # Retrieve necessary values from Stage 1 stored in session_state to pass to Stage 2 simulation
    # These were stored at the end of the Stage 1 calculation block
    stage1_data_for_s2 = {
        "weight_kg": st.session_state.get('stage1_weight_kg', 75.0),
        "ffm_kg": st.session_state.get('stage1_ffm_kg', 60.0),
        "fm_kg": st.session_state.get('stage1_fm_kg', 15.0),
        "initial_bmr_baseline": st.session_state.get('stage1_initial_bmr_baseline', 1600.0),
        "final_tdee": st.session_state.get('stage1_final_tdee', 2500.0),
        "UAB": st.session_state.get('stage1_UAB', 2800.0), # Upper Adaptive Bound from Stage 1
        "is_critically_low_intake_scenario": st.session_state.get('stage1_is_critically_low_intake', False),
        "adjusted_intake": st.session_state.get('stage1_adjusted_intake', 2500.0),
        "protein_g_per_day": st.session_state.get('stage1_protein_g_day',150.0),
        "sex": st.session_state.get("stage1_sex", "Male"),
        "age_years": st.session_state.get("stage1_age_years", 25),
        "height_cm": st.session_state.get("stage1_height_cm", 178.0),
        "avg_daily_steps": st.session_state.get("stage1_avg_daily_steps", 7500),
        "other_exercise_kcal": st.session_state.get("stage1_other_exercise_kcal", 0),
        "typical_indoor_temp_f": st.session_state.get("stage1_typical_indoor_temp_f", 70),
        "minutes_cold_exposure_daily": st.session_state.get("stage1_minutes_cold_exposure_daily",0),
        "avg_sleep_hours": st.session_state.get("stage1_avg_sleep_hours", 7.5),
        "uses_caffeine": st.session_state.get("stage1_uses_caffeine", True),
        "has_fever_illness": False, # Assume no illness for forecast
        "peak_fever_temp_f_input": 98.6, # Assume no illness for forecast
        "streamlit_object": st
    }

    # --- Stage 2 Inputs (from sidebar, now retrieved from session_state) ---
    s2_goal = st.session_state.get("stage2_goal_radio", "Bulk (Gain Weight/Muscle)")
    s2_duration_weeks = st.session_state.get("stage2_duration_weeks_key", 8)
    
    s2_rate_val_display = 0.0
    s2_rate_unit_display = st.session_state.get("weight_unit", "lbs") # Use the global weight unit
    if s2_goal != "Maintain (Re-evaluate Current TDEE)":
        if s2_rate_unit_display == "lbs":
            s2_rate_val_display = st.session_state.get("stage2_rate_val_key", 0.5) # Get the value from the correct keyed widget
        else: # kg
            s2_rate_val_display = st.session_state.get("stage2_rate_val_key", 0.23)


    # Convert display rate to kg/wk for internal calculations if needed, but target kcal uses the direct rate
    s2_rate_for_calc_kg_wk = 0.0
    if s2_goal != "Maintain (Re-evaluate Current TDEE)":
        if s2_rate_unit_display == "lbs":
            s2_rate_for_calc_kg_wk = lbs_to_kg(s2_rate_val_display)
        else:
            s2_rate_for_calc_kg_wk = s2_rate_val_display

    # Target Daily Calories for Stage 2 (retrieved from session_state where it was calculated)
    s2_target_daily_kcal = st.session_state.get("s2_target_daily_kcal", stage1_data_for_s2['final_tdee'])


    s2_target_protein_g = st.session_state.get("s2_protein_g_slider", 150.0)
    s2_target_carb_g = st.session_state.get("s2_carb_g_slider", 300.0)
    s2_target_fat_g = st.session_state.get("s2_fat_g_slider", 70.0)
    
    s2_training_status_for_sim = "Novice" # Default
    s2_weightlifting_for_sim = True # Default
    if s2_goal.startswith("Bulk"):
        s2_training_status_for_sim = st.session_state.get("stage2_training_status_key", "Novice")
        s2_weightlifting_for_sim = st.session_state.get("stage2_weightlifting_key", True)

    # Prepare stage2_inputs dictionary
    stage2_inputs_for_sim = {
        "goal": s2_goal,
        "rate_val": s2_rate_val_display, # The numeric rate value
        "rate_unit": s2_rate_unit_display, # The unit of the rate value
        "duration_weeks": s2_duration_weeks,
        "target_daily_kcal": s2_target_daily_kcal,
        "protein_g": s2_target_protein_g,
        "carb_g": s2_target_carb_g,
        "fat_g": s2_target_fat_g,
        "training_status": s2_training_status_for_sim,
        "weightlifting": s2_weightlifting_for_sim
    }

    # Check if macro calories align with target calories for Stage 2
    current_macro_kcal_s2 = (s2_target_protein_g * 4) + (s2_target_carb_g * 4) + (s2_target_fat_g * 9)
    kcal_difference_s2 = current_macro_kcal_s2 - s2_target_daily_kcal
    if abs(kcal_difference_s2) > 75: # Allow a bit more leeway
        st.warning(f"Macro Calorie Check: Calories from your target macros ({current_macro_kcal_s2:,.0f} kcal) differ by {kcal_difference_s2:+.0f} kcal from the estimated target daily intake for your goal ({s2_target_daily_kcal:,.0f} kcal). Consider adjusting macros in the sidebar for closer alignment before starting the forecast.")

    if st.button(f"🔮 Start Stage 2: Forecast {s2_goal.split(' ')[0]}", type="primary", use_container_width=True, key="start_stage2_button"):
        num_forecast_days = s2_duration_weeks * 7
        
        st.markdown(f"#### Simulating: **{s2_goal}** for **{s2_duration_weeks} weeks** at a target rate of **{s2_rate_val_display:.2f} {s2_rate_unit_display}/week**.")
        st.markdown(f"Targeting **{s2_target_daily_kcal:,.0f} kcal/day** with P: {s2_target_protein_g:.0f}g, C: {s2_target_carb_g:.0f}g, F: {s2_target_fat_g:.0f}g.")
        if s2_goal.startswith("Bulk"):
            st.markdown(f"Training Status: **{s2_training_status_for_sim}**, Weightlifting: **{'Yes' if s2_weightlifting_for_sim else 'No'}**")

        s2_daily_log_df, s2_final_states = simulate_bulk_cut_forecast(
            stage1_data_for_s2, 
            stage2_inputs_for_sim, 
            num_forecast_days
        )

        if not s2_daily_log_df.empty and s2_final_states:
            st.subheader("📈 Stage 2 Forecast Results")

            s2_initial_weight = stage1_data_for_s2['weight_kg']
            s2_final_weight = s2_final_states['final_weight_kg']
            s2_total_weight_change = s2_final_weight - s2_initial_weight
            
            s2_initial_ffm = stage1_data_for_s2['ffm_kg']
            s2_final_ffm = s2_final_states['final_ffm_kg']
            s2_total_ffm_change = s2_final_ffm - s2_initial_ffm

            s2_initial_fm = stage1_data_for_s2['fm_kg']
            s2_final_fm = s2_final_states['final_fm_kg']
            s2_total_fm_change = s2_final_fm - s2_initial_fm

            res_s2_col1, res_s2_col2, res_s2_col3 = st.columns(3)
            res_s2_col1.metric("Projected End Weight", f"{s2_final_weight:.1f} kg ({kg_to_lbs(s2_final_weight):.1f} lbs)", delta=f"{s2_total_weight_change:+.2f} kg ({kg_to_lbs(s2_total_weight_change):+.2f} lbs)")
            res_s2_col2.metric("Projected Change in FFM", f"{s2_total_ffm_change:+.2f} kg ({kg_to_lbs(s2_total_ffm_change):+.2f} lbs)")
            res_s2_col3.metric("Projected Change in Fat Mass", f"{s2_total_fm_change:+.2f} kg ({kg_to_lbs(s2_total_fm_change):+.2f} lbs)")
            
            st.markdown("#### Forecasted Changes Over Time:")
            plot_s2_cols = st.columns(2)
            with plot_s2_cols[0]:
                st.markdown("**Weight (kg)**")
                st.line_chart(s2_daily_log_df.set_index('Day')['Weight_kg'])
                st.markdown("**Fat-Free Mass (FFM, kg)**")
                st.line_chart(s2_daily_log_df.set_index('Day')['FFM_kg'])
            with plot_s2_cols[1]:
                st.markdown(f"**Total Dynamic TDEE (kcal) {INFO_ICON}**", help="Simulated TDEE adapting to your Stage 2 intake and changing body composition.")
                st.line_chart(s2_daily_log_df.set_index('Day')['Total_Dynamic_TDEE'])
                st.markdown("**Fat Mass (FM, kg)**")
                st.line_chart(s2_daily_log_df.set_index('Day')['FM_kg'])
            
            st.markdown("**Glycogen Stores (g)**")
            st.line_chart(s2_daily_log_df.set_index('Day')[['Glycogen_g', 'Max_Glycogen_g']])


            with st.expander("Show Detailed Daily Forecast Log (Stage 2)", expanded=False):
                cols_to_display_s2 = [col for col in ["Day", "Weight_kg", "FFM_kg", "FM_kg", "Glycogen_g", "Total_Dynamic_TDEE", "Target_Intake_s2", "Energy_Balance_Daily_vs_TDEE", "Energy_For_Tissue_Change_kcal", "Delta_FFM_kg_Tissue", "Delta_FM_kg_Tissue", "Delta_Glycogen_g"] if col in s2_daily_log_df.columns]
                display_df_log_s2 = s2_daily_log_df[cols_to_display_s2]
                st.dataframe(display_df_log_s2.style.format("{:,.1f}", na_rep="-", subset=pd.IndexSlice[:, [c for c in cols_to_display_s2 if c not in ['Day', 'Target_Intake_s2']]]).format("{:,.0f}", na_rep="-", subset=pd.IndexSlice[:, ['Target_Intake_s2']]))
        
        else:
            st.error("Stage 2 Forecast Simulation failed to produce results.")
            
# --- Final Disclaimer and Glossary outside of any button condition ---
st.sidebar.markdown("---")
st.sidebar.markdown("Model based on user-provided research & insights.")
st.markdown("---")
# ... (Full Disclaimer and Glossary as in the previous version) ...
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
