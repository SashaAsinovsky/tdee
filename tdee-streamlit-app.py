import streamlit as st
import numpy as np
import pandas as pd
# import json # No longer needed for profile saving
# from io import StringIO # No longer needed for profile saving
#--- Constants and Coefficients ---

KCAL_PER_KG_TISSUE=7700  # Approx. kcal per kg of body tissue change

SURPLUS_EFFICIENCY=0.85  # % of surplus calories stored (15% lost to adaptive thermogenesis)

DEFICIT_EFFICIENCY=1.00  # % of deficit calories leading to tissue loss



TAU_BMR_ADAPTATION=10.0

TAU_NEAT_ADAPTATION=2.5 # Slightly increased for more gradual general adaptation



KCAL_PER_STEP_BASE_FACTOR=0.00062 # kcal per step per kg of bodyweight



SEDENTARY_WELLFED_RMR_MULTIPLIER=1.6 # For "non-locomotor NEAT & upregulation" implies TDEE = RMR * 1.6 for sedentary well-fed

CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR=1.2 # BMR*1.2 for (BMR_adaptive + NEAT_adaptive) at very low intake



# Glycogen related constants

GLYCOGEN_G_PER_KG_FFM_MUSCLE = 15.0 # g of glycogen per kg of FFM for muscle stores

LIVER_GLYCOGEN_CAPACITY_G = 100.0 # g for liver

WATER_G_PER_G_GLYCOGEN = 3.5 # g of water stored with each g of glycogen

KCAL_PER_G_CARB_FOR_GLYCOGEN = 4.0 # Approx. kcal to store 1g of glycogen from carbs



# Muscle gain rates (lbs/month) - will be converted to kg/day

MUSCLE_GAIN_NOVICE_LBS_MONTH = 2.0

MUSCLE_GAIN_INTERMEDIATE_LBS_MONTH = 1.0

MUSCLE_GAIN_ADVANCED_LBS_MONTH = 0.5



# FMI/FFMI Hazard Ratio Threshold

FMI_HIGH_RISK_HR_THRESHOLD = 12.0 # FMI >= 12 considered "High" or "Very High" risk



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

    else:  # Female

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

    protein_percentage = (protein_kcal / intake_kcal) * 100.0 if intake_kcal > 0 else 0

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

    if not has_fever_illness or peak_fever_temp_f is None or peak_fever_temp_f <= 99.0 or current_bmr_adaptive is None or current_bmr_adaptive == 0: return 0.0

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

    fmi_significant_risk_flag = fmi >= FMI_HIGH_RISK_HR_THRESHOLD # FMI >= 12.0

    ffmi_low_for_health_risk_flag = ffmi < 18 # FFMI below optimal/sweet spot indicates need to bulk for HR benefit



    if ffmi_direct_bulk_triggered: # User rule: FFMI < 20

        if fmi_significant_risk_flag: # If FMI is in significant risk zone (>=12)

            final_recommendation = "REC: Complex - Body Recomp or Prioritize Fat Loss then Lean Bulk. FFMI is below 20, but FMI is also in a high-risk zone. Consult a professional."

        else:

            final_recommendation = f"REC: BULK. Your FFMI ({ffmi:.1f}) is below the {FFMI_DIRECT_BULK_THRESHOLD:.0f} kg/m² target. Focus on a caloric surplus."

            if status_caloric == "Deficit": final_recommendation += " Current state is deficit; increase intake."

            elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; surplus needed."

    elif fmi_significant_risk_flag: # High FMI risk takes precedence if FFMI is not < 20 for the direct bulk trigger

        final_recommendation = f"REC: CUT. Your FMI ({fmi:.1f}) is in a high-risk zone ({fmi_hr_category}). Focus on a caloric deficit."

        if status_caloric == "Surplus": final_recommendation += " Current state is surplus; decrease intake."

        elif status_caloric == "Maintenance": final_recommendation += " Current state is maintenance; a deficit is needed."

    elif ffmi_low_for_health_risk_flag: # FFMI is not < 20 (direct bulk), but still in a >1 HR zone (e.g. 15-17.9)

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

        "stage2_goal_radio": "Bulk (Gain Weight/Muscle)", # Default, user will choose via radio

        "stage2_rate_val_key": 0.5, # Default rate, widget specific key

        "stage2_duration_weeks_key": 8, # Default duration, widget specific key

        "s2_protein_g_slider": 150.0, # Macronutrient slider keys

        "s2_carb_g_slider": 300.0,

        "s2_fat_g_slider": 70.0,

        "stage2_training_status_key": "Novice (Less than 1 year consistent training)", # Widget specific key

        "stage2_weightlifting_key": True, # Widget specific key

        "stage1_results_calculated": False, # Flag to control Stage 2 UI visibility

        "stage2_macros_need_reinit": True, # Flag to reinitialize macros for Stage 2

        "prev_stage2_goal_for_macros": "" # To track goal changes for macro reinitialization

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

                            min_value=(20.0 if st.session_state.weight_unit == "kg" else 50.0), 

                            max_value=(300.0 if st.session_state.weight_unit == "kg" else 700.0), 

                            step=0.1, format="%.1f", key="weight_input_val")



    st.sidebar.slider("Estimated Body Fat Percentage (%):", min_value=3.0, max_value=60.0, step=0.5, format="%.1f", key="body_fat_percentage")

    st.sidebar.selectbox("Sex:", ("Male", "Female"), key="sex", index=["Male", "Female"].index(st.session_state.sex))

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

    st.sidebar.selectbox("Recent Body Weight Trend:", ("Steady", "Gaining", "Losing"), key="weight_trend", index=["Steady", "Gaining", "Losing"].index(st.session_state.weight_trend))

    

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

                      min_value=7, max_value=90, step=7, key="num_days_to_simulate_s1")



display_sidebar_stage1_inputs() # Call function to display sidebar and manage state via keys





st.title("💪 Advanced Dynamic TDEE & Metabolic Modeler ⚙️")

st.markdown("""

This tool simulates Total Daily Energy Expenditure (TDEE) by modeling metabolic adaptations.

It incorporates body composition health risk profiles (FMI/FFMI) for nuanced nutritional strategy insights.

Inputs should reflect **current, stable conditions** for initial assessment, or **target conditions** for simulation.

""")

st.header("📊 Stage 1: Current TDEE Analysis & Metabolic Snapshot")



if st.sidebar.button("🚀 Calculate & Simulate TDEE (Stage 1)", type="primary", use_container_width=True, key="calculate_stage1_button"):

    # Retrieve ALL values from st.session_state using their keys

    s_weight_unit = st.session_state.weight_unit

    s_height_unit = st.session_state.height_unit

    s_weight_input_val = st.session_state.weight_input_val

    

    if s_weight_unit == "lbs": 

        s_weight_kg = lbs_to_kg(s_weight_input_val)

    else: 

        s_weight_kg = s_weight_input_val



    s_body_fat_percentage = st.session_state.body_fat_percentage

    s_sex = st.session_state.sex

    s_age_years = st.session_state.age_years



    if s_height_unit == "ft/in": 

        s_height_cm = ft_in_to_cm(st.session_state.feet, st.session_state.inches)

    else: 

        s_height_cm = st.session_state.height_cm_input

    

    s_avg_daily_steps = st.session_state.avg_daily_steps

    s_other_exercise_kcal_per_day = st.session_state.other_exercise_kcal_per_day

    s_avg_daily_kcal_intake_reported = st.session_state.avg_daily_kcal_intake_reported

    s_protein_g_per_day = st.session_state.protein_g_per_day

    s_weight_trend = st.session_state.weight_trend

    

    s_weight_change_rate_display_val = 0.0 

    s_weight_change_rate_kg_wk = 0.0

    if s_weight_trend != "Steady":

        if s_weight_unit == "lbs":

            s_weight_change_rate_display_val = st.session_state.weight_change_rate_input_val_lbs

            s_weight_change_rate_kg_wk = lbs_to_kg(s_weight_change_rate_display_val)

        else: # kg

            s_weight_change_rate_display_val = st.session_state.weight_change_rate_input_val_kg

            s_weight_change_rate_kg_wk = s_weight_change_rate_display_val

    

    s_typical_indoor_temp_f = st.session_state.typical_indoor_temp_f

    s_minutes_cold_exposure_daily = st.session_state.minutes_cold_exposure_daily

    s_avg_sleep_hours = st.session_state.avg_sleep_hours

    s_uses_caffeine = st.session_state.uses_caffeine

    s_has_fever_illness = st.session_state.has_fever_illness

    s_peak_fever_temp_f_input = st.session_state.peak_fever_temp_f_input if s_has_fever_illness else 98.6

    s_num_days_to_simulate_s1 = st.session_state.num_days_to_simulate_s1





    if s_height_cm <= 0:

        st.error("Height must be a positive value. Please check your inputs.")

    else:

        ffm_kg, fm_kg = calculate_ffm_fm(s_weight_kg, s_body_fat_percentage)

        height_m = s_height_cm / 100.0

        bmi = s_weight_kg / (height_m**2) if height_m > 0 else 0



        # Edge Case: Body Composition Check

        ffmi_check, fmi_check = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)

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

        lower_bound_tdee_static_display = initial_bmr_ref * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR

        upper_bound_tdee_static_display_calc = calculate_dlw_tdee(ffm_kg, fm_kg)

        

        pal_for_UAB_display_heuristic = get_pal_multiplier_for_heuristic(s_avg_daily_steps)

        min_plausible_UAB_display = initial_bmr_ref * pal_for_UAB_display_heuristic 

        if upper_bound_tdee_static_display_calc <= lower_bound_tdee_static_display or upper_bound_tdee_static_display_calc < min_plausible_UAB_display:

            upper_bound_tdee_static_display = initial_bmr_ref * pal_for_UAB_display_heuristic * 1.05 

            if upper_bound_tdee_static_display_calc == 0: 

                 st.caption("Note: FFM-based TDEE (DLW) was zero. Using PAL-based heuristic for static upper bound display.")

        else:

            upper_bound_tdee_static_display = upper_bound_tdee_static_display_calc

        

        # Adjusted True Intake Calculation

        adjusted_true_intake = adjust_reported_intake(s_avg_daily_kcal_intake_reported, s_weight_trend, s_weight_change_rate_kg_wk)

        

        # Edge Case: Intake Check (moved after initial_bmr_ref is confirmed)

        if initial_bmr_ref > 0 and adjusted_true_intake < initial_bmr_ref * 0.9 : 

            st.warning(f"⚠️ Calibrated intake ({adjusted_true_intake:,.0f} kcal) is very low vs RMR ({initial_bmr_ref:,.0f} kcal). This may be unsustainable, carry health risks, and lead to significant metabolic slowdown. The simulation will activate 'Critically Low Intake Mode'.")

        elif upper_bound_tdee_static_display > 0 and adjusted_true_intake > upper_bound_tdee_static_display * 1.75 : 

            st.warning(f"⚠️ Calibrated intake ({adjusted_true_intake:,.0f} kcal) is very high vs upper TDEE ({upper_bound_tdee_static_display:,.0f} kcal). This suggests a very large surplus and rapid weight gain, potentially with unfavorable body composition changes.")



        with st.expander("Advanced Metabolic Insights & Benchmarks", expanded=False):

            st.markdown(f"#### ↔️ Estimated Static Metabolic Range {INFO_ICON}", help="General reference for TDEE boundaries. The simulation below models dynamic adaptations within a similar, internally calculated range (LAB & UAB).")

            st.markdown(f"""

            - **Static Lower Adaptive Bound (approx. RMR + minimal NEAT): `{lower_bound_tdee_static_display:,.0f} kcal/day`** (Initial RMR * {CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR:.1f}). <span title='{TOOLTIPS["LAB"]}'>{INFO_ICON}</span>

            - **Static Upper Adaptive Bound (Typical Free-living TDEE): `{upper_bound_tdee_static_display:,.0f} kcal/day`** (FFM-based DLW formula or heuristic). <span title='{TOOLTIPS["UAB"]}'>{INFO_ICON}</span>

            """, unsafe_allow_html=True)

            

            energy_non_loco, energy_loco, implied_steps, tdee_sed_floor_calc = calculate_implied_activity_breakdown(

                upper_bound_tdee_static_display, 

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

        if abs(adjusted_true_intake - s_avg_daily_kcal_intake_reported) > 20: 

            unit_for_trend_cap = s_weight_unit 

            display_rate_val_for_cap = s_weight_change_rate_display_val 

            if s_weight_trend != "Steady" :

                 st.metric("Calibrated True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal", help="Estimated from reported intake & recent weight trend; used as target for simulation.")

                 # Determine the rate value as input by the user for the caption

                 if s_weight_unit == "kg":

                     rate_val_for_caption_display = st.session_state.weight_change_rate_input_val_kg

                 else: # lbs

                     rate_val_for_caption_display = st.session_state.weight_change_rate_input_val_lbs

                 st.caption(f"Adjusted based on reported weight trend of {rate_val_for_caption_display:.2f} {unit_for_trend_cap}/week ({s_weight_trend}).")

        else: 

            st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")



        # Prepare inputs for the simulation function

        simulation_inputs = {

            "weight_kg": s_weight_kg, "body_fat_percentage": s_body_fat_percentage,

            "sex": s_sex, "age_years": s_age_years, "height_cm": s_height_cm,

            "avg_daily_steps": s_avg_daily_steps, 

            "other_exercise_kcal_per_day": s_other_exercise_kcal_per_day, 

            "adjusted_intake": adjusted_true_intake,

            "protein_g_per_day": s_protein_g_per_day,

            "typical_indoor_temp_f": s_typical_indoor_temp_f,

            "minutes_cold_exposure_daily": s_minutes_cold_exposure_daily,

            "avg_sleep_hours": s_avg_sleep_hours, 

            "uses_caffeine": s_uses_caffeine, 

            "has_fever_illness": s_has_fever_illness, 

            "peak_fever_temp_f": s_peak_fever_temp_f_input,

            "streamlit_object": st 

        }

        

        # --- Stage 1 TDEE Adaptation Simulation ---

        st.subheader(f"⏳ Simulated TDEE Adaptation Over {s_num_days_to_simulate_s1} Days")

        st.caption(f"Based on maintaining a calibrated true intake of **{adjusted_true_intake:,.0f} kcal/day** and other lifestyle factors.")

        

        daily_tdee_log_df, final_tdee_states = simulate_tdee_adaptation(simulation_inputs, s_num_days_to_simulate_s1)



        current_dynamic_tdee = 0.0 

        if not daily_tdee_log_df.empty and final_tdee_states:

            current_dynamic_tdee = final_tdee_states.get('final_tdee', 0)

            LAB_sim = final_tdee_states.get('LAB', 0)

            UAB_sim = final_tdee_states.get('UAB', 0)

            st.session_state.stage1_UAB = UAB_sim # Store for Stage 2 target calculation

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

            valid_chart_cols = [col for col in chart_cols_to_plot if col in daily_tdee_log_df.columns and daily_tdee_log_df[col].abs().sum() > 0.01] 

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

        

        advice_s1, overall_status_msg_s1, daily_surplus_deficit_val_s1 = generate_bulk_cut_assessment(

            adjusted_true_intake, 

            current_dynamic_tdee,

            final_tdee_states.get('initial_bmr_baseline', initial_bmr_ref), 

            ffm_kg, 

            fm_kg, 

            height_m, 

            bmi, 

            s_sex,

            final_tdee_states 

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

        st.session_state.stage1_weight_kg = s_weight_kg 

        st.session_state.stage1_height_m = height_m

        st.session_state.stage1_initial_bmr_baseline = final_tdee_states.get('initial_bmr_baseline', initial_bmr_ref)

        st.session_state.stage1_adjusted_intake = adjusted_true_intake 

        st.session_state.stage1_protein_g_day = s_protein_g_per_day 

        st.session_state.stage1_avg_daily_steps = s_avg_daily_steps 

        st.session_state.stage1_other_exercise_kcal = s_other_exercise_kcal_per_day 

        st.session_state.stage1_sex = s_sex

        st.session_state.stage1_age_years = s_age_years

        st.session_state.stage1_height_cm = s_height_cm

        st.session_state.stage1_is_critically_low_intake = final_tdee_states.get('is_critically_low_intake_scenario', False)

        # (stage1_UAB was stored earlier when UAB_sim was defined)

        st.session_state.stage1_typical_indoor_temp_f = s_typical_indoor_temp_f

        st.session_state.stage1_minutes_cold_exposure_daily = s_minutes_cold_exposure_daily

        st.session_state.stage1_avg_sleep_hours = s_avg_sleep_hours

        st.session_state.stage1_uses_caffeine = s_uses_caffeine

        st.session_state.stage1_has_fever_illness = s_has_fever_illness # This will be False for forecast

        st.session_state.stage1_peak_fever_temp_f_input = s_peak_fever_temp_f_input # Reset for forecast



# --- Stage 2: Bulk/Cut Forecast Simulation Function ---

def simulate_bulk_cut_forecast(stage1_data, stage2_inputs, num_forecast_days):

    """

    Simulates weight, body composition, and TDEE changes during a bulk or cut phase.

    """

    current_weight_kg = stage1_data['weight_kg']

    current_ffm_kg = stage1_data['ffm_kg']

    current_fm_kg = stage1_data['fm_kg']

    initial_bmr_baseline = stage1_data['initial_bmr_baseline'] 

    s1_is_critically_low_intake = stage1_data.get('is_critically_low_intake_scenario', False)

    s1_intake = stage1_data.get('adjusted_intake', initial_bmr_baseline * 1.5) 



    sim_inputs_for_components = {

        "weight_kg": current_weight_kg, 

        "body_fat_percentage": (current_fm_kg / current_weight_kg) * 100 if current_weight_kg > 0 else 0, 

        "sex": stage1_data['sex'], 

        "age_years": stage1_data['age_years'], 

        "height_cm": stage1_data['height_cm'],

        "avg_daily_steps": stage1_data.get('avg_daily_steps', 7500), 

        "other_exercise_kcal_per_day": stage1_data.get('other_exercise_kcal', 0),

        "typical_indoor_temp_f": stage1_data.get('typical_indoor_temp_f', 70),

        "minutes_cold_exposure_daily": stage1_data.get('minutes_cold_exposure_daily',0),

        "avg_sleep_hours": stage1_data.get('avg_sleep_hours', 7.5),

        "uses_caffeine": stage1_data.get('uses_caffeine', True),

        "has_fever_illness": False, # Assume no fever for forecast period

        "peak_fever_temp_f": 98.6, # Assume no fever for forecast

        "streamlit_object": stage1_data.get("streamlit_object", None) 

    }



    goal = stage2_inputs['goal'] 

    target_intake_s2 = stage2_inputs['target_daily_kcal']

    target_protein_g_s2 = stage2_inputs['protein_g']

    

    kcal_per_step_s2 = KCAL_PER_STEP_BASE_FACTOR * current_weight_kg 

    eat_kcal_steps_s2 = sim_inputs_for_components['avg_daily_steps'] * kcal_per_step_s2

    eat_kcal_additional_exercise_s2 = sim_inputs_for_components['other_exercise_kcal_per_day']

    current_eat_kcal = eat_kcal_steps_s2 + eat_kcal_additional_exercise_s2



    max_glycogen_capacity_g = (current_ffm_kg * GLYCOGEN_G_PER_KG_FFM_MUSCLE) + LIVER_GLYCOGEN_CAPACITY_G

    

    s1_final_tdee = stage1_data.get('final_tdee', initial_bmr_baseline * 1.5) 

    if s1_is_critically_low_intake or (s1_intake < s1_final_tdee - 400): 

        current_glycogen_g = max_glycogen_capacity_g * 0.25 

    elif s1_intake > s1_final_tdee + 400: 

        current_glycogen_g = max_glycogen_capacity_g * 0.85 

    else: 

        current_glycogen_g = max_glycogen_capacity_g * 0.60 



    current_glycogen_g = np.clip(current_glycogen_g, 0, max_glycogen_capacity_g)



    current_bmr_adaptive = initial_bmr_baseline 

    current_neat_adaptive_component = 0.0 



    min_bmr_limit_factor = 0.80; max_bmr_limit_factor = 1.15 

    min_bmr_target_limit = initial_bmr_baseline * min_bmr_limit_factor # Based on person's initial baseline

    max_bmr_target_limit = initial_bmr_baseline * max_bmr_limit_factor # Based on person's initial baseline

    min_neat_limit = -500; max_neat_limit = 750 



    s2_LAB = initial_bmr_baseline * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR # Based on person's initial baseline

    s2_UAB_initial = calculate_dlw_tdee(current_ffm_kg, current_fm_kg) 

    pal_for_s2_uab_heuristic = get_pal_multiplier_for_heuristic(sim_inputs_for_components['avg_daily_steps'])

    if s2_UAB_initial == 0 or s2_UAB_initial < s2_LAB * 1.05:

        s2_UAB_initial = initial_bmr_baseline * pal_for_s2_uab_heuristic * 1.05

    s2_intake_in_adaptive_range_initial = s2_LAB <= target_intake_s2 <= s2_UAB_initial

    

    day0_s2_tef = calculate_tef(target_intake_s2, target_protein_g_s2)

    day0_s2_cold = calculate_cold_thermogenesis(sim_inputs_for_components['typical_indoor_temp_f'], sim_inputs_for_components['minutes_cold_exposure_daily'])

    day0_s2_fever = calculate_immune_fever_effect(sim_inputs_for_components['has_fever_illness'], sim_inputs_for_components['peak_fever_temp_f'], current_bmr_adaptive) # Will be 0

    TDEE_s2_sim_start = current_bmr_adaptive + day0_s2_tef + current_eat_kcal + current_neat_adaptive_component + day0_s2_cold + day0_s2_fever



    s2_daily_log = []

    current_weight_for_sim = current_weight_kg 



    lbs_to_kg_conversion = 0.453592

    days_in_month = 30.4375

    muscle_gain_rate_kg_day = {

        "Novice": (MUSCLE_GAIN_NOVICE_LBS_MONTH * lbs_to_kg_conversion) / days_in_month,

        "Intermediate": (MUSCLE_GAIN_INTERMEDIATE_LBS_MONTH * lbs_to_kg_conversion) / days_in_month,

        "Advanced": (MUSCLE_GAIN_ADVANCED_LBS_MONTH * lbs_to_kg_conversion) / days_in_month,

    }

    

    s2_daily_log.append({

            "Day": 0, "Weight_kg": current_weight_kg, "FFM_kg": current_ffm_kg, "FM_kg": current_fm_kg,

            "Glycogen_g": current_glycogen_g, "Max_Glycogen_g": max_glycogen_capacity_g,

            "BMR_Adaptive": current_bmr_adaptive, "NEAT_Adaptive": current_neat_adaptive_component,

            "EAT": current_eat_kcal, "TEF": day0_s2_tef, "Total_Dynamic_TDEE": TDEE_s2_sim_start,

            "Target_Intake_s2": target_intake_s2,

            "Energy_Balance_Daily_vs_TDEE": target_intake_s2 - TDEE_s2_sim_start, 

            "Delta_FFM_kg_Tissue": 0, "Delta_FM_kg_Tissue": 0, "Delta_Glycogen_g":0,

            "Energy_For_Tissue_Change_kcal": 0

    })

    

    for day in range(int(num_forecast_days)): 

        kcal_per_step_s2 = KCAL_PER_STEP_BASE_FACTOR * current_weight_for_sim

        eat_kcal_steps_s2 = sim_inputs_for_components['avg_daily_steps'] * kcal_per_step_s2

        current_eat_kcal = eat_kcal_steps_s2 + sim_inputs_for_components['other_exercise_kcal_per_day']



        tef_s2 = calculate_tef(target_intake_s2, target_protein_g_s2)

        cold_s2 = day0_s2_cold # Assumed constant based on initial S1 inputs for S2 forecast

        fever_s2 = 0 # Assumed no fever during forecast



        # --- TDEE Adaptation (BMR & NEAT) for Stage 2 ---

        day_target_bmr_s2 = current_bmr_adaptive

        day_target_neat_s2 = current_neat_adaptive_component



        # Recalculate current BMR baseline based on potentially changed FFM/FM for daily checks

        current_ffm_fm_based_rmr = calculate_pontzer_ffm_based_rmr(current_ffm_kg, current_fm_kg)

        if current_ffm_fm_based_rmr <= 0: current_ffm_fm_based_rmr = initial_bmr_baseline # Fallback

        

        # Use this daily calculated RMR for critical low checks and dynamic adaptive bounds.

        # However, the absolute min/max BMR limits remain tied to the individual's initial_bmr_baseline.

        current_min_bmr_target_daily = current_ffm_fm_based_rmr * min_bmr_limit_factor

        current_max_bmr_target_daily = current_ffm_fm_based_rmr * max_bmr_limit_factor

        

        # Ensure daily targets don't exceed the person's overall physiological limits from initial_bmr_baseline

        current_min_bmr_target_daily = max(current_min_bmr_target_daily, min_bmr_target_limit)

        current_max_bmr_target_daily = min(current_max_bmr_target_daily, max_bmr_target_limit)





        is_critically_low_intake_s2_daily = target_intake_s2 < (current_ffm_fm_based_rmr * 1.35)

        

        # Re-evaluate UAB daily based on current FFM/FM for adaptive range logic

        current_UAB_daily = calculate_dlw_tdee(current_ffm_kg, current_fm_kg)

        if current_UAB_daily == 0 or current_UAB_daily < s2_LAB * 1.05: # s2_LAB is fixed to initial

            current_UAB_daily = current_ffm_fm_based_rmr * pal_for_s2_uab_heuristic * 1.05

        

        s2_intake_in_adaptive_range_daily = s2_LAB <= target_intake_s2 <= current_UAB_daily





        if is_critically_low_intake_s2_daily:

            day_target_bmr_s2 = current_min_bmr_target_daily

            target_bmr_plus_neat_floor_s2 = current_ffm_fm_based_rmr * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR

            day_target_neat_s2 = target_bmr_plus_neat_floor_s2 - day_target_bmr_s2

            day_target_neat_s2 = np.clip(day_target_neat_s2, min_neat_limit, 0)

        

        elif s2_intake_in_adaptive_range_daily: 

            s2_day0_bmr_for_adapt = current_ffm_fm_based_rmr 

            s2_day0_neat_for_adapt = 0 



            s2_tdee_start_for_adaptation_calc = s2_day0_bmr_for_adapt + tef_s2 + current_eat_kcal + s2_day0_neat_for_adapt + cold_s2 + fever_s2

            s2_total_adaptation_gap = target_intake_s2 - s2_tdee_start_for_adaptation_calc

            

            s2_neat_share = 0.60 if s2_total_adaptation_gap > 0 else 0.40

            s2_bmr_share = 1.0 - s2_neat_share

            s2_target_total_neat_change = s2_total_adaptation_gap * s2_neat_share

            s2_target_total_bmr_change = s2_total_adaptation_gap * s2_bmr_share

            

            day_target_neat_s2 = np.clip(s2_day0_neat_for_adapt + s2_target_total_neat_change, min_neat_limit, max_neat_limit)

            day_target_bmr_s2 = np.clip(s2_day0_bmr_for_adapt + s2_target_total_bmr_change, current_min_bmr_target_daily, current_max_bmr_target_daily)



        else: # Outside adaptive range, not critically low

            expenditure_for_balance_s2 = current_bmr_adaptive + tef_s2 + current_eat_kcal + current_neat_adaptive_component + cold_s2 + fever_s2

            energy_balance_s2 = target_intake_s2 - expenditure_for_balance_s2

            

            bmr_target_change_factor_s2 = 0.0

            if energy_balance_s2 > 250: bmr_target_change_factor_s2 = 0.05

            elif energy_balance_s2 < -250: bmr_target_change_factor_s2 = -0.10

            day_target_bmr_s2 = current_ffm_fm_based_rmr * (1 + bmr_target_change_factor_s2)

            day_target_bmr_s2 = np.clip(day_target_bmr_s2, current_min_bmr_target_daily, current_max_bmr_target_daily)



            neat_resp_s2 = 0.30 

            if sim_inputs_for_components['avg_daily_steps'] > 10000: neat_resp_s2 += 0.10

            if sim_inputs_for_components['avg_daily_steps'] < 5000: neat_resp_s2 -=0.10

            if sim_inputs_for_components['avg_sleep_hours'] < 6.5: neat_resp_s2 -= 0.05

            if sim_inputs_for_components['uses_caffeine']: neat_resp_s2 += 0.05

            neat_resp_s2 = np.clip(neat_resp_s2, 0.1, 0.5)

            day_target_neat_s2 = energy_balance_s2 * neat_resp_s2

            day_target_neat_s2 = np.clip(day_target_neat_s2, min_neat_limit, max_neat_limit)



        s2_current_tau_bmr = TAU_BMR_ADAPTATION * (0.7 if day < 7 else 1.0) 

        s2_current_tau_neat = TAU_NEAT_ADAPTATION * (0.7 if day < 7 else 1.0)



        delta_bmr_s2 = (day_target_bmr_s2 - current_bmr_adaptive) / s2_current_tau_bmr

        current_bmr_adaptive += delta_bmr_s2

        delta_neat_s2 = (day_target_neat_s2 - current_neat_adaptive_component) / s2_current_tau_neat

        current_neat_adaptive_component += delta_neat_s2

        

        current_bmr_adaptive = np.clip(current_bmr_adaptive, current_min_bmr_target_daily, current_max_bmr_target_daily) # Clip to daily dynamic limits

        current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit, max_bmr_target_limit) # Then clip to absolute person limits

        current_neat_adaptive_component = np.clip(current_neat_adaptive_component, min_neat_limit, max_neat_limit)

        

        if is_critically_low_intake_s2_daily:

            bmr_plus_neat_target_floor_s2 = current_ffm_fm_based_rmr * CRITICALLY_LOW_INTAKE_RMR_MULTIPLIER_FLOOR

            current_bmr_adaptive = max(current_bmr_adaptive, current_min_bmr_target_daily) # Ensure BMR doesn't go below its daily dynamic min

            current_bmr_adaptive = max(current_bmr_adaptive, min_bmr_target_limit) # And also not below absolute min

            

            required_neat_for_floor_s2 = bmr_plus_neat_target_floor_s2 - current_bmr_adaptive

            current_neat_adaptive_component = max(required_neat_for_floor_s2, min_neat_limit)

            if current_neat_adaptive_component == min_neat_limit: 

                current_bmr_adaptive = max(bmr_plus_neat_target_floor_s2 - min_neat_limit, current_min_bmr_target_daily)

                current_bmr_adaptive = max(current_bmr_adaptive, min_bmr_target_limit)

            current_bmr_adaptive = np.clip(current_bmr_adaptive, current_min_bmr_target_daily, current_max_bmr_target_daily)

            current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit, max_bmr_target_limit)





        total_tdee_today_s2 = current_bmr_adaptive + tef_s2 + current_eat_kcal + current_neat_adaptive_component + cold_s2 + fever_s2

        

        # --- Daily Glycogen Dynamics ---

        delta_glycogen_g_today = 0.0

        kcal_for_glycogen_change_today = 0.0

        carb_intake_g_s2 = stage2_inputs['carb_g'] 

        estimated_carb_oxidation_g = 125.0 + (current_eat_kcal / 4.5) 

        net_carb_for_glycogen_g = carb_intake_g_s2 - estimated_carb_oxidation_g



        max_glycogen_capacity_g = (current_ffm_kg * GLYCOGEN_G_PER_KG_FFM_MUSCLE) + LIVER_GLYCOGEN_CAPACITY_G

        min_glycogen_physio_floor_g = max_glycogen_capacity_g * 0.15 



        if net_carb_for_glycogen_g > 0 and current_glycogen_g < max_glycogen_capacity_g: 

            can_store_g = max_glycogen_capacity_g - current_glycogen_g

            max_daily_glycogen_gain_g = 150 

            delta_glycogen_g_today = min(net_carb_for_glycogen_g, can_store_g, max_daily_glycogen_gain_g)

            current_glycogen_g += delta_glycogen_g_today

            kcal_for_glycogen_change_today = delta_glycogen_g_today * KCAL_PER_G_CARB_FOR_GLYCOGEN 

        

        elif net_carb_for_glycogen_g < 0 and current_glycogen_g > min_glycogen_physio_floor_g: 

            can_lose_g = current_glycogen_g - min_glycogen_physio_floor_g

            max_daily_glycogen_loss_g = 200 

            delta_glycogen_g_today = max(net_carb_for_glycogen_g, -can_lose_g, -max_daily_glycogen_loss_g) 

            current_glycogen_g += delta_glycogen_g_today 



        current_glycogen_g = np.clip(current_glycogen_g, min_glycogen_physio_floor_g, max_glycogen_capacity_g)

        

        # --- Energy Balance for Tissue Change ---

        overall_daily_energy_balance = target_intake_s2 - total_tdee_today_s2

        energy_balance_for_tissue_today = overall_daily_energy_balance - kcal_for_glycogen_change_today



        # --- Tissue Partitioning (Fat/Lean Mass Changes) ---

        delta_ffm_kg_today = 0.0

        delta_fm_kg_today = 0.0



        if goal.startswith("Bulk"):

            if current_glycogen_g >= (max_glycogen_capacity_g * 0.80) and energy_balance_for_tissue_today > 0:

                effective_surplus_for_tissue_kcal = energy_balance_for_tissue_today * SURPLUS_EFFICIENCY

                

                target_muscle_gain_kg_day = 0.0

                training_status_str = stage2_inputs.get('training_status', "Novice") # Full string

                

                if stage2_inputs.get('weightlifting', False):

                    if "Novice" in training_status_str: target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Novice"]

                    elif "Intermediate" in training_status_str: target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Intermediate"]

                    elif "Advanced" in training_status_str: target_muscle_gain_kg_day = muscle_gain_rate_kg_day["Advanced"]

                else: 

                    target_muscle_gain_kg_day = (muscle_gain_rate_kg_day["Advanced"] * 0.20) 



                kcal_cost_per_kg_ffm_gain = 5800 

                kcal_for_target_ffm_gain_today = target_muscle_gain_kg_day * kcal_cost_per_kg_ffm_gain

                

                kcal_to_ffm_today = min(kcal_for_target_ffm_gain_today, effective_surplus_for_tissue_kcal * 0.75)

                kcal_to_ffm_today = min(kcal_to_ffm_today, effective_surplus_for_tissue_kcal) 



                delta_ffm_kg_today = kcal_to_ffm_today / kcal_cost_per_kg_ffm_gain if kcal_cost_per_kg_ffm_gain > 0 else 0

                

                remaining_surplus_for_fat_kcal = effective_surplus_for_tissue_kcal - kcal_to_ffm_today

                delta_fm_kg_today = remaining_surplus_for_fat_kcal / KCAL_PER_KG_TISSUE if KCAL_PER_KG_TISSUE > 0 else 0

                delta_fm_kg_today = max(0, delta_fm_kg_today) 



        elif goal.startswith("Cut"):

            effective_deficit_for_tissue_kcal = energy_balance_for_tissue_today * DEFICIT_EFFICIENCY 

            if effective_deficit_for_tissue_kcal < 0: 

                daily_deficit_for_partitioning = abs(effective_deficit_for_tissue_kcal)

                

                if daily_deficit_for_partitioning * 7 > (2.0 * 3500): 

                    ffm_loss_percentage_of_tissue_deficit = 0.50

                else: 

                    ffm_loss_percentage_of_tissue_deficit = 0.25

                

                kcal_lost_as_ffm = daily_deficit_for_partitioning * ffm_loss_percentage_of_tissue_deficit

                kcal_lost_as_fm = daily_deficit_for_partitioning * (1.0 - ffm_loss_percentage_of_tissue_deficit)

                

                kcal_density_ffm_loss = 1800 

                

                delta_ffm_kg_today = -(kcal_lost_as_ffm / kcal_density_ffm_loss) if kcal_density_ffm_loss > 0 else 0

                delta_fm_kg_today = -(kcal_lost_as_fm / KCAL_PER_KG_TISSUE) if KCAL_PER_KG_TISSUE > 0 else 0

        

        elif goal.startswith("Maintain"): 

            if abs(energy_balance_for_tissue_today) > 30: 

                if energy_balance_for_tissue_today > 0: 

                    delta_ffm_kg_today = (energy_balance_for_tissue_today * 0.30 * SURPLUS_EFFICIENCY) / 5800 if 5800 > 0 else 0

                    delta_fm_kg_today = (energy_balance_for_tissue_today * 0.70 * SURPLUS_EFFICIENCY) / KCAL_PER_KG_TISSUE if KCAL_PER_KG_TISSUE > 0 else 0

                else: 

                    delta_ffm_kg_today = (energy_balance_for_tissue_today * 0.20 * DEFICIT_EFFICIENCY) / 1800 if 1800 > 0 else 0

                    delta_fm_kg_today = (energy_balance_for_tissue_today * 0.80 * DEFICIT_EFFICIENCY) / KCAL_PER_KG_TISSUE if KCAL_PER_KG_TISSUE > 0 else 0



        # --- Update Body Composition and Weight ---

        current_ffm_kg += delta_ffm_kg_today

        current_fm_kg += delta_fm_kg_today

        

        min_physio_fm_kg = current_weight_for_sim * (0.03 if sim_inputs_for_components['sex']=="Male" else 0.10) 

        current_fm_kg = max(current_fm_kg, min_physio_fm_kg if min_physio_fm_kg > 0 else 0.1) # Ensure fm doesn't become zero or negative

        current_ffm_kg = max(current_ffm_kg, 20.0) # Absolute floor for FFM



        # Calculate total body weight using FFM, FM, and total glycogen+water mass

        # FFM here is "dry" FFM. Glycogen is part of "wet" FFM in some definitions, but here tracked separately for weight.

        glycogen_plus_water_mass_kg = current_glycogen_g * (1 + WATER_G_PER_G_GLYCOGEN) / 1000.0

        new_total_weight = current_ffm_kg + current_fm_kg + glycogen_plus_water_mass_kg



        # The weight change used to update current_weight_for_sim should be from the previous day's state

        # This is simpler:

        current_weight_for_sim = new_total_weight

        

        sim_inputs_for_components['weight_kg'] = current_weight_for_sim

        if current_weight_for_sim > 0:

            sim_inputs_for_components['body_fat_percentage'] = (current_fm_kg / current_weight_for_sim) * 100

        else:

            sim_inputs_for_components['body_fat_percentage'] = 0



        s2_daily_log.append({

            "Day": day + 1, 

            "Weight_kg": current_weight_for_sim, 

            "FFM_kg": current_ffm_kg, 

            "FM_kg": current_fm_kg,

            "Glycogen_g": current_glycogen_g, 

            "Max_Glycogen_g": max_glycogen_capacity_g,

            "BMR_Adaptive": current_bmr_adaptive, 

            "NEAT_Adaptive": current_neat_adaptive_component,

            "EAT": current_eat_kcal, 

            "TEF": tef_s2, 

            "Total_Dynamic_TDEE": total_tdee_today_s2,

            "Target_Intake_s2": target_intake_s2,

            "Energy_Balance_Daily_vs_TDEE": overall_daily_energy_balance, 

            "Energy_For_Tissue_Change_kcal": energy_balance_for_tissue_today,

            "Delta_FFM_kg_Tissue": delta_ffm_kg_today, 

            "Delta_FM_kg_Tissue": delta_fm_kg_today,

            "Delta_Glycogen_g": delta_glycogen_g_today

        })



    s2_final_states = {

        "final_weight_kg": current_weight_for_sim,

        "final_ffm_kg": current_ffm_kg,

        "final_fm_kg": current_fm_kg,

        "final_glycogen_g": current_glycogen_g,

        "final_tdee_s2": s2_daily_log[-1]['Total_Dynamic_TDEE'] if s2_daily_log and len(s2_daily_log) > 1 else TDEE_s2_sim_start,

        "final_bmr_adaptive_s2": current_bmr_adaptive,

        "final_neat_adaptive_s2": current_neat_adaptive_component

    }

    return pd.DataFrame(s2_daily_log), s2_final_states





# --- Streamlit App UI (Main Body - Continuation for Stage 2) ---

if st.session_state.get('stage1_results_calculated', False):

    st.markdown("---") 

    st.header("🚀 Stage 2: Bulk/Cut Forecast & Simulation")

    st.markdown("Based on your Stage 1 analysis, plan your next phase.")



    s1_final_tdee = st.session_state.get('stage1_final_tdee', 2500)

    s1_UAB = st.session_state.get('stage1_UAB', s1_final_tdee * 1.1) # Get UAB from stage 1 or estimate

    s1_initial_bmr_baseline = st.session_state.get('stage1_initial_bmr_baseline', 1600)



    stage1_data_for_s2 = {

        "weight_kg": st.session_state.get('stage1_weight_kg', 75.0),

        "ffm_kg": st.session_state.get('stage1_ffm_kg', 60.0),

        "fm_kg": st.session_state.get('stage1_fm_kg', 15.0),

        "initial_bmr_baseline": s1_initial_bmr_baseline,

        "final_tdee": s1_final_tdee, # Stage 1 adapted TDEE

        "UAB": s1_UAB, 

        "is_critically_low_intake_scenario": st.session_state.get('stage1_is_critically_low_intake', False),

        "adjusted_intake": st.session_state.get('stage1_adjusted_intake', 2500.0), # Stage 1 intake

        "protein_g_per_day": st.session_state.get('stage1_protein_g_day',150.0), # Stage 1 protein

        "sex": st.session_state.get("stage1_sex", "Male"),

        "age_years": st.session_state.get("stage1_age_years", 25),

        "height_cm": st.session_state.get("stage1_height_cm", 178.0),

        "avg_daily_steps": st.session_state.get("stage1_avg_daily_steps", 7500),

        "other_exercise_kcal": st.session_state.get("stage1_other_exercise_kcal", 0),

        "typical_indoor_temp_f": st.session_state.get("stage1_typical_indoor_temp_f", 70),

        "minutes_cold_exposure_daily": st.session_state.get("stage1_minutes_cold_exposure_daily",0),

        "avg_sleep_hours": st.session_state.get("stage1_avg_sleep_hours", 7.5),

        "uses_caffeine": st.session_state.get("stage1_uses_caffeine", True),

        "streamlit_object": st

    }

    

    st.sidebar.markdown("---")

    st.sidebar.subheader("🎯 Set Your Goal for Stage 2")



    s2_goal_options = ["Bulk (Gain Weight/Muscle)", "Cut (Lose Weight/Fat)", "Maintain (Re-evaluate Current TDEE)"]

    

    st.sidebar.radio(

        "What is your primary goal for this phase?",

        s2_goal_options,

        key="stage2_goal_radio", 

        index=s2_goal_options.index(st.session_state.stage2_goal_radio) 

    )

    s2_goal = st.session_state.stage2_goal_radio # Get current selection



    rate_col, duration_col = st.columns(2)
    s2_rate_unit = st.session_state.get("weight_unit", "lbs") # Use same unit as Stage 1 for consistency

    with rate_col:
        if st.session_state.stage2_goal_radio.startswith("Bulk"): # Use the radio key
            default_rate = 0.5 if s2_rate_unit == "lbs" else 0.23
            min_rate, max_rate, step_rate = (0.1, 2.0, 0.1) if s2_rate_unit == "lbs" else (0.05, 1.0, 0.05)
            help_text_rate = f"Recommended safe lean bulking rates are typically 0.25-0.5 {s2_rate_unit}/week for most. Higher rates risk more fat gain."
        elif st.session_state.stage2_goal_radio.startswith("Cut"): # Use the radio key
            default_rate = 1.0 if s2_rate_unit == "lbs" else 0.45
            min_rate, max_rate, step_rate = (0.25, 3.0, 0.05) if s2_rate_unit == "lbs" else (0.1, 1.5, 0.025)
            help_text_rate = f"Sustainable fat loss rates are often 0.5-2.0 {s2_rate_unit}/week. Faster rates (>2 {s2_rate_unit}/wk or >1% bodyweight/wk) increase risk of muscle loss."
        else: # Maintain
            default_rate = 0.0
            min_rate, max_rate, step_rate = (0.0, 0.0, 0.01)
            help_text_rate = "For maintenance, the goal is to keep weight stable (rate of change = 0)."

        st.sidebar.number_input( # Displaying in sidebar
            f"Desired Rate of Weight Change ({s2_rate_unit}/week):",
            min_value=min_rate, max_value=max_rate,
            step=step_rate, format="%.2f",
            key="stage2_rate_val_key", # Key for the widget
            help=help_text_rate,
            disabled=(st.session_state.stage2_goal_radio.startswith("Maintain")) # Use radio key
        )
        # s2_rate_for_calc will be retrieved from st.session_state.stage2_rate_val_key later

    with duration_col:
        st.sidebar.number_input( # Displaying in sidebar
            "Duration of this Phase (weeks):",
            min_value=1, max_value=52,
            step=1, key="stage2_duration_weeks_key", # Key for the widget
            help="How long do you plan to adhere to this bulk/cut phase?",
            disabled=(st.session_state.stage2_goal_radio.startswith("Maintain")) # Use radio key
        )

    # Calculate Target Daily Calories for Stage 2
    s2_target_daily_kcal_calc = s1_final_tdee # Default to current TDEE (maintenance)
    # Retrieve the rate value from the widget's key
    s2_rate_val_from_widget = st.session_state.stage2_rate_val_key
    s2_rate_for_actual_calc = abs(s2_rate_val_from_widget) if not st.session_state.stage2_goal_radio.startswith("Maintain") else 0.0
    kcal_change_for_rate = (s2_rate_for_actual_calc * (3500 if s2_rate_unit == "lbs" else KCAL_PER_KG_TISSUE)) / 7.0

    if st.session_state.stage2_goal_radio.startswith("Bulk"):
        target_maintenance_tdee_for_bulk = st.session_state.get("stage1_UAB", s1_final_tdee)
        if target_maintenance_tdee_for_bulk < s1_final_tdee:
            target_maintenance_tdee_for_bulk = s1_final_tdee
        s2_target_daily_kcal_calc = target_maintenance_tdee_for_bulk + kcal_change_for_rate
        st.sidebar.caption(f"Bulking target: adding {kcal_change_for_rate:,.0f} kcal/day to an est. upper maintenance of {target_maintenance_tdee_for_bulk:,.0f} kcal.")
    elif st.session_state.stage2_goal_radio.startswith("Cut"):
        s2_target_daily_kcal_calc = s1_final_tdee - kcal_change_for_rate
        st.sidebar.caption(f"Cutting target: subtracting {kcal_change_for_rate:,.0f} kcal/day from current TDEE of {s1_final_tdee:,.0f} kcal.")
    
    st.session_state.s2_target_daily_kcal = s2_target_daily_kcal_calc # Store for macro sliders and main page display

    st.sidebar.subheader(f"🥗 Target Macronutrient Split (Stage 2)")
    st.sidebar.markdown(f"Target Intake: **`{st.session_state.s2_target_daily_kcal:,.0f} kcal`**")

    # Reinitialize macros if goal changes or if it's the first time showing stage 2 macros
    if st.session_state.stage2_goal_radio != st.session_state.prev_stage2_goal_for_macros or st.session_state.stage2_macros_need_reinit:
        protein_g_default_s2 = st.session_state.get('stage1_protein_g_day', 150.0)
        st.session_state.s2_protein_g_slider = protein_g_default_s2
        protein_kcal_s2 = protein_g_default_s2 * 4
        remaining_kcal_s2 = st.session_state.s2_target_daily_kcal - protein_kcal_s2
        if remaining_kcal_s2 < 0: remaining_kcal_s2 = 0
        
        st.session_state.s2_carb_g_slider = (remaining_kcal_s2 * 0.60) / 4 # Default 60% of remaining for carbs
        st.session_state.s2_fat_g_slider = (remaining_kcal_s2 * 0.40) / 9  # Default 40% for fat
        st.session_state.stage2_macros_need_reinit = False
        st.session_state.prev_stage2_goal_for_macros = st.session_state.stage2_goal_radio


    max_protein_g_s2 = s1_weight_kg * 3.5 
    max_carb_g_s2 = st.session_state.s2_target_daily_kcal / 4 
    max_fat_g_s2 = st.session_state.s2_target_daily_kcal / 9

    st.sidebar.slider("Target Protein (g/day):", 0.0, max_protein_g_s2, key="s2_protein_g_slider", step=1.0)
    st.sidebar.slider("Target Carbohydrates (g/day):", 0.0, max_carb_g_s2, key="s2_carb_g_slider", step=1.0)
    st.sidebar.slider("Target Fat (g/day):", 0.0, max_fat_g_s2, key="s2_fat_g_slider", step=1.0)

    current_macro_kcal_s2_sidebar = (st.session_state.s2_protein_g_slider * 4) + \
                                    (st.session_state.s2_carb_g_slider * 4) + \
                                    (st.session_state.s2_fat_g_slider * 9)
    kcal_difference_s2_sidebar = current_macro_kcal_s2_sidebar - st.session_state.s2_target_daily_kcal
    st.sidebar.caption(f"Macros: {current_macro_kcal_s2_sidebar:,.0f} kcal (Diff: {kcal_difference_s2_sidebar:+.0f} kcal)")


    if st.session_state.stage2_goal_radio.startswith("Bulk"):
        st.sidebar.subheader(f"🏋️‍♂️ Bulking Phase Considerations")
        training_options = ["Novice (Less than 1 year consistent training)",
                            "Intermediate (1-3 years consistent training)",
                            "Advanced (3+ years consistent training)"]
        current_training_status = st.session_state.get("stage2_training_status_key", training_options[0])
        st.sidebar.selectbox(
            "Your Resistance Training Status:",
            training_options,
            key="stage2_training_status_key",
            index=training_options.index(current_training_status) if current_training_status in training_options else 0
        )
        st.sidebar.checkbox(
            "Actively Weightlifting/Resistance Training during this Bulk?",
            key="stage2_weightlifting_key"
        )

    # --- Main Page Display for Stage 2 ---
    st.markdown(f"Your primary goal for this phase is: **{st.session_state.stage2_goal_radio}**.")
    if not st.session_state.stage2_goal_radio.startswith("Maintain"):
        st.markdown(f"Targeting a weight change of **{st.session_state.stage2_rate_val_key:.2f} {s2_rate_unit}/week** for **{st.session_state.stage2_duration_weeks_key} weeks**.")
    st.markdown(f"This requires an estimated target daily intake of **{st.session_state.s2_target_daily_kcal:,.0f} kcal/day**.")
    st.markdown(f"Your target macronutrient split is: "
                f"Protein: **{st.session_state.s2_protein_g_slider:.0f}g**, "
                f"Carbs: **{st.session_state.s2_carb_g_slider:.0f}g**, "
                f"Fat: **{st.session_state.s2_fat_g_slider:.0f}g**.")
    
    if abs(kcal_difference_s2_sidebar) > 75:
        st.warning(f"Macro Calorie Check: Calories from your target macros ({current_macro_kcal_s2_sidebar:,.0f} kcal) differ by {kcal_difference_s2_sidebar:+.0f} kcal from the estimated target daily intake. Please adjust macros in the sidebar for closer alignment.")

    if st.button(f"🔮 Start Stage 2: Forecast {st.session_state.stage2_goal_radio.split(' ')[0]}", type="primary", use_container_width=True, key="start_stage2_button"):
        num_forecast_days = st.session_state.stage2_duration_weeks_key * 7
        
        # Prepare stage2_inputs for the simulation function using values from session_state
        stage2_inputs_for_sim = {
            "goal": st.session_state.stage2_goal_radio,
            "rate_val": st.session_state.stage2_rate_val_key, 
            "rate_unit": s2_rate_unit, 
            "duration_weeks": st.session_state.stage2_duration_weeks_key,
            "target_daily_kcal": st.session_state.s2_target_daily_kcal,
            "protein_g": st.session_state.s2_protein_g_slider,
            "carb_g": st.session_state.s2_carb_g_slider,
            "fat_g": st.session_state.s2_fat_g_slider,
            "training_status": st.session_state.stage2_training_status_key if st.session_state.stage2_goal_radio.startswith("Bulk") else "Novice", # Default if not bulk
            "weightlifting": st.session_state.stage2_weightlifting_key if st.session_state.stage2_goal_radio.startswith("Bulk") else False # Default if not bulk
        }
        
        st.markdown(f"#### Simulating: **{stage2_inputs_for_sim['goal']}** for **{stage2_inputs_for_sim['duration_weeks']} weeks**.")
        if not stage2_inputs_for_sim['goal'].startswith("Maintain"):
             st.markdown(f"Target rate: **{stage2_inputs_for_sim['rate_val']:.2f} {stage2_inputs_for_sim['rate_unit']}/week**.")
        st.markdown(f"Targeting **{stage2_inputs_for_sim['target_daily_kcal']:,.0f} kcal/day** (P: {stage2_inputs_for_sim['protein_g']:.0f}g, C: {stage2_inputs_for_sim['carb_g']:.0f}g, F: {stage2_inputs_for_sim['fat_g']:.0f}g).")
        if stage2_inputs_for_sim['goal'].startswith("Bulk"):
            st.markdown(f"Training Status: **{stage2_inputs_for_sim['training_status']}**, Weightlifting: **{'Yes' if stage2_inputs_for_sim['weightlifting'] else 'No'}**")

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
                # Apply number formatting carefully
                format_dict = {col: "{:,.1f}" for col in cols_to_display_s2 if col not in ['Day', 'Target_Intake_s2', 'Glycogen_g', 'Max_Glycogen_g', 'Total_Dynamic_TDEE', 'Energy_Balance_Daily_vs_TDEE', 'Energy_For_Tissue_Change_kcal']}
                format_dict.update({col: "{:,.0f}" for col in ['Target_Intake_s2', 'Glycogen_g', 'Max_Glycogen_g', 'Total_Dynamic_TDEE', 'Energy_Balance_Daily_vs_TDEE', 'Energy_For_Tissue_Change_kcal']})
                format_dict.update({col: "{:,.3f}" for col in ['Delta_FFM_kg_Tissue', 'Delta_FM_kg_Tissue']})


                st.dataframe(display_df_log_s2.style.format(format_dict, na_rep="-"))
        
        else:
            st.error("Stage 2 Forecast Simulation failed to produce results.")
            
# --- Final Disclaimer and Glossary outside of any button condition ---
st.sidebar.markdown("---")
st.sidebar.info("Model based on user-provided research & insights. Verify with professional advice.")

st.markdown("---")
st.markdown(f"""
    **Disclaimer & Model Limitations:** This tool provides estimates based on scientific literature and mathematical modeling.
    Individual metabolic responses vary. Results are for informational/educational purposes only, not medical/nutritional advice.
    Consult qualified professionals before changing diet/exercise. Accuracy depends on inputs and model simplifications.
    Metabolic adaptation is dynamic; plateaus can occur. FMI/FFMI HR interpretations are visual approximations.
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
