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

# --- Helper Functions ---
def kg_to_lbs(kg): return kg * 2.20462
def lbs_to_kg(lbs): return lbs / 2.20462
def ft_in_to_cm(ft, inch): return (ft * 30.48) + (inch * 2.54)

# --- Core Calculation Functions ---
def calculate_ffm_fm(weight_kg, body_fat_percentage):
    fm_kg = weight_kg * (body_fat_percentage / 100.0)
    ffm_kg = weight_kg - fm_kg
    return ffm_kg, fm_kg

def calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg):
    if ffm_kg <= 0: return 0
    try:
        fm_kg_adjusted_for_log = fm_kg if fm_kg > 0.001 else 0.001
        term_ffm = 0.707 * np.log(ffm_kg)
        term_fm = 0.019 * np.log(fm_kg_adjusted_for_log)
        rmr_mj_day = np.exp(-0.954 + term_ffm + term_fm)
        return rmr_mj_day * 238.8529999 
    except Exception: return 0

def calculate_mifflin_st_jeor_rmr(weight_kg, height_cm, age_years, sex):
    if sex == "Male": rmr = (10*weight_kg)+(6.25*height_cm)-(5*age_years)+5
    else: rmr = (10*weight_kg)+(6.25*height_cm)-(5*age_years)-161
    return rmr

def calculate_dlw_tdee(ffm_kg, fm_kg): # Pontzer FFM-based TDEE
    if ffm_kg <= 0: return 0
    try:
        term_ffm = 0.916 * np.log(ffm_kg)
        term_fm = -0.030 * np.log(fm_kg if fm_kg > 0.001 else 0.001)
        tdee = np.exp(-1.102 + term_ffm + term_fm) * 238.83
        return tdee
    except Exception: return 0

def get_pal_multiplier_for_heuristic(activity_steps):
    if activity_steps < 5000: return 1.25
    elif activity_steps < 7500: return 1.4
    elif activity_steps < 10000: return 1.55
    elif activity_steps < 12500: return 1.75
    else: return 1.95
    
def adjust_reported_intake(reported_intake_kcal, weight_trend, weight_change_rate_kg_wk):
    kcal_discrepancy_per_day = (weight_change_rate_kg_wk * KCAL_PER_KG_TISSUE) / 7.0
    adjusted_intake = reported_intake_kcal + kcal_discrepancy_per_day
    return adjusted_intake

def calculate_tef(intake_kcal, protein_g):
    protein_kcal = protein_g * 4.0
    protein_percentage = (protein_kcal / intake_kcal) * 100.0 if intake_kcal > 0 else 0.0
    if protein_percentage > 25.0: tef_factor = 0.15
    elif protein_percentage > 15.0: tef_factor = 0.12
    else: tef_factor = 0.10
    return intake_kcal * tef_factor

def calculate_cold_thermogenesis(typical_indoor_temp_f, minutes_cold_exposure_daily):
    k_c_per_degree_f_day = 4.0; comfort_temp_f = 68.0; cold_kcal_indoor = 0.0
    if typical_indoor_temp_f < comfort_temp_f:
        temp_diff_f_indoor = comfort_temp_f - typical_indoor_temp_f
        cold_kcal_indoor = temp_diff_f_indoor * k_c_per_degree_f_day * ((24.0*60.0 - minutes_cold_exposure_daily) / (24.0*60.0))
    cold_kcal_outdoor = (minutes_cold_exposure_daily / 60.0) * 30.0
    return max(0, cold_kcal_indoor) + max(0, cold_kcal_outdoor)

def calculate_immune_fever_effect(has_fever_illness, peak_fever_temp_f, current_bmr_adaptive):
    if not has_fever_illness or peak_fever_temp_f <= 99.0: return 0.0
    temp_diff_f = peak_fever_temp_f - 98.6
    if temp_diff_f <= 0: return 0.0
    return current_bmr_adaptive * temp_diff_f * 0.065

def calculate_ffmi_fmi(ffm_kg, fm_kg, height_m):
    ffmi = ffm_kg / (height_m**2) if height_m > 0 and ffm_kg >=0 else 0
    fmi = fm_kg / (height_m**2) if height_m > 0 and fm_kg >=0 else 0
    return ffmi, fmi

# New function to derive implied activity
def calculate_implied_activity_and_steps(tdee_ffm_based, rmr_ffm_based, weight_kg):
    if tdee_ffm_based == 0 or rmr_ffm_based == 0:
        return 0, 0

    tef_implicit = tdee_ffm_based * 0.10 # Assuming 10% TEF at maintenance
    implied_activity_energy = tdee_ffm_based - rmr_ffm_based - tef_implicit
    implied_activity_energy = max(0, implied_activity_energy) # Cannot be negative

    kcal_per_step = KCAL_PER_STEP_BASE_FACTOR * weight_kg
    implied_total_steps = implied_activity_energy / kcal_per_step if kcal_per_step > 0 else 0
    
    return implied_activity_energy, implied_total_steps

# --- Main Simulation Logic ---
def simulate_tdee_adaptation(inputs, num_days_to_simulate=14):
    ffm_kg, fm_kg = calculate_ffm_fm(inputs['weight_kg'], inputs['body_fat_percentage'])
    initial_bmr_baseline = calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg)
    if initial_bmr_baseline == 0: 
        initial_bmr_baseline = calculate_mifflin_st_jeor_rmr(inputs['weight_kg'], inputs['height_cm'], inputs['age_years'], inputs['sex'])
        if 'streamlit_object' in inputs and inputs['streamlit_object'] is not None:
            inputs['streamlit_object'].warning("Pontzer FFM-RMR failed. Using Mifflin as initial BMR.")
    
    LAB = initial_bmr_baseline * 1.2 
    UAB = calculate_dlw_tdee(ffm_kg, fm_kg)
    pal_for_uab_heuristic = get_pal_multiplier_for_heuristic(inputs['avg_daily_steps'])
    if UAB == 0 or UAB < LAB * 1.05: 
        UAB = initial_bmr_baseline * pal_for_uab_heuristic * 1.05 
        if 'streamlit_object' in inputs and inputs['streamlit_object'] is not None:
             inputs['streamlit_object'].warning(f"DLW-TDEE for UAB low. UAB set heuristically to {UAB:,.0f} kcal.")

    target_true_intake_for_sim = inputs['adjusted_intake']
    kcal_per_step = KCAL_PER_STEP_BASE_FACTOR * inputs['weight_kg']
    eat_kcal_steps = inputs['avg_daily_steps'] * kcal_per_step
    eat_kcal_additional_exercise = inputs['additional_exercise_kcal_per_day']
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
            target_bmr_plus_neat_floor = initial_bmr_baseline * 1.2
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
        
        current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit, max_bmr_target_limit)
        current_neat_adaptive_component = np.clip(current_neat_adaptive_component, min_neat_limit, max_neat_limit)

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

# --- generate_bulk_cut_assessment function ---
def generate_bulk_cut_assessment(
    adjusted_intake, dynamic_tdee,
    initial_bmr_baseline, # Used for critical low intake context
    ffm_kg, fm_kg, height_m, bmi, sex,
    sim_final_states 
    ):
    ffmi, fmi = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
    LAB = sim_final_states.get('LAB', initial_bmr_baseline * 1.2) 
    UAB = sim_final_states.get('UAB', initial_bmr_baseline * 1.7) 
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
                           f"~{initial_bmr_baseline*1.2:,.0f} kcal before fixed EAT & TEF.\n")
    elif intake_was_in_adaptive_range:
        advice_primary += (f"Your intake fell within the estimated primary metabolic adaptive range ({LAB:,.0f} - {UAB:,.0f} kcal). "
                           "TDEE largely adapted to this intake.\n")
    else:
        advice_primary += (f"Your intake was outside the primary adaptive range ({LAB:,.0f} - {UAB:,.0f} kcal). "
                           "This intake level is expected to primarily drive tissue change.\n")

    advice_composition = ""; fmi_hr_status = ""; ffmi_hr_status = ""; ffmi_direct_bulk_triggered = False
    FMI_HIGH_RISK_HR_THRESHOLD = 10.0; FMI_VERY_LOW_RISK_HR_THRESHOLD = 3.5
    FMI_OPTIMAL_LOWER = 4.0; FMI_OPTIMAL_UPPER = 9.0
    FFMI_DIRECT_BULK_THRESHOLD = 20.0; FFMI_LOW_HR_THRESHOLD = 17.0

    if fmi > FMI_HIGH_RISK_HR_THRESHOLD: fmi_hr_status = "High (Suggests HR > 1)"; advice_composition += f"- FMI ({fmi:.1f}): High, risk-associated. Consider fat loss.\n"
    elif fmi < FMI_VERY_LOW_RISK_HR_THRESHOLD: fmi_hr_status = "Very Low (Suggests HR > 1)"; advice_composition += f"- FMI ({fmi:.1f}): Very low. Ensure healthy/sustainable.\n"
    elif FMI_OPTIMAL_LOWER <= fmi <= FMI_OPTIMAL_UPPER: fmi_hr_status = "Optimal (Suggests HR < 1)"; advice_composition += f"- FMI ({fmi:.1f}): Appears in a lower-risk range.\n"
    else: fmi_hr_status = "Intermediate"; advice_composition += f"- FMI ({fmi:.1f}): Intermediate. Monitor.\n"

    if ffmi < FFMI_DIRECT_BULK_THRESHOLD:
        ffmi_direct_bulk_triggered = True; ffmi_hr_status = f"< {FFMI_DIRECT_BULK_THRESHOLD} (Direct Bulk Rec.)"
        advice_composition += f"- FFMI ({ffmi:.1f}): Below {FFMI_DIRECT_BULK_THRESHOLD}. Building lean mass is a priority.\n"
        if ffmi < FFMI_LOW_HR_THRESHOLD: advice_composition += f"  This is also in a risk-associated (HR > 1) low range.\n"
    elif ffmi < FFMI_LOW_HR_THRESHOLD:
        ffmi_hr_status = "Low (Suggests HR > 1)"; advice_composition += f"- FFMI ({ffmi:.1f}): Risk-associated (HR > 1). Consider lean mass gain.\n"
    else: ffmi_hr_status = "Sufficient/Optimal (Suggests HR < 1)"; advice_composition += f"- FFMI ({ffmi:.1f}): Good to high lean mass.\n"

    final_recommendation = ""; overall_status_message = f"Caloric: {status_caloric} | FMI: {fmi_hr_status} | FFMI: {ffmi_hr_status}"

    if ffmi_direct_bulk_triggered:
        if fmi_hr_status == "High (Suggests HR > 1)": final_recommendation = "REC: Body Recomp/Careful Lean Bulk. FFMI low, FMI high. Complex. Consult pro."
        else:
            final_recommendation = f"REC: BULK. FFMI ({ffmi:.1f}) < {FFMI_DIRECT_BULK_THRESHOLD}. Needs surplus."
            if status_caloric == "Deficit": final_recommendation += " Current state deficit; increase intake."
            elif status_caloric == "Maintenance": final_recommendation += " Current state maintenance; surplus needed."
    elif fmi_hr_status == "High (Suggests HR > 1)":
        final_recommendation = f"REC: CUT. FMI ({fmi:.1f}) high. Needs deficit."
        if status_caloric == "Surplus": final_recommendation += " Current state surplus; decrease intake."
        elif status_caloric == "Maintenance": final_recommendation += " Current state maintenance; deficit needed."
    elif ffmi_hr_status == "Low (Suggests HR > 1)":
        final_recommendation = f"REC: Consider Lean Bulk. FFMI ({ffmi:.1f}) suggests HR > 1 risk."
        if status_caloric == "Deficit": final_recommendation += " Current state deficit; adjust."
        elif status_caloric == "Maintenance": final_recommendation += " Current state maintenance; slight surplus?"
    elif status_caloric == "Surplus": final_recommendation = "REC: SURPLUS. If bulking, aligns. Monitor."
    elif status_caloric == "Deficit": final_recommendation = "REC: DEFICIT. If cutting, aligns. Preserve muscle."
    elif status_caloric == "Maintenance":
        if ffmi_hr_status.startswith("Sufficient/Optimal") and fmi_hr_status.startswith("Optimal"): final_recommendation = "REC: MAINTAIN/Optimize. Healthy comp. Adjust for specific goals."
        else: final_recommendation = "REC: MAINTAIN & RE-EVALUATE. Review FMI/FFMI for next phase."
    else: final_recommendation = "Review goals with FMI/FFMI context. Adjust intake."

    full_advice = (f"{advice_primary}\n\n**Body Composition Insights (FMI/FFMI & Approx. HR from Chart):**\n{advice_composition}\n**Overall Strategy Guidance:**\n{final_recommendation}")
    return full_advice, overall_status_message, daily_surplus_deficit_vs_dynamic_tdee

# --- project_weight_change_scenarios function ---
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
st.title("💪 Advanced Dynamic TDEE & Metabolic Modeler ⚙️")
st.markdown("""
This tool simulates Total Daily Energy Expenditure (TDEE) by modeling metabolic adaptations.
It incorporates body composition health risk profiles (FMI/FFMI) for nuanced nutritional strategy insights.
Inputs should reflect **current, stable conditions** for initial assessment, or **target conditions** for simulation.
""")

st.sidebar.header("📝 User Inputs")
unit_cols = st.sidebar.columns(2)
weight_unit = unit_cols[0].radio("Weight unit:", ("kg", "lbs"), key="weight_unit_radio")
height_unit = unit_cols[1].radio("Height unit:", ("cm", "ft/in"), key="height_unit_radio")

st.sidebar.subheader("👤 Body & Demographics")
if weight_unit == "lbs":
    weight_input_val = st.sidebar.number_input("Current Body Weight (lbs):", min_value=50.0, max_value=700.0, value=150.0, step=0.1, format="%.1f")
    weight_kg = lbs_to_kg(weight_input_val)
else:
    weight_input_val = st.sidebar.number_input("Current Body Weight (kg):", min_value=20.0, max_value=300.0, value=68.0, step=0.1, format="%.1f")
    weight_kg = weight_input_val

body_fat_percentage = st.sidebar.slider("Estimated Body Fat Percentage (%):", min_value=3.0, max_value=60.0, value=15.0, step=0.5, format="%.1f")
sex = st.sidebar.selectbox("Sex:", ("Male", "Female"), index=0)
age_years = st.sidebar.number_input("Age (years):", min_value=13, max_value=100, value=25, step=1)

if height_unit == "ft/in":
    h_col1, h_col2 = st.sidebar.columns(2)
    feet = h_col1.number_input("Height (feet):", min_value=3, max_value=8, value=5, step=1)
    inches = h_col2.number_input("Height (inches):", min_value=0, max_value=11, value=10, step=1)
    height_cm = ft_in_to_cm(feet, inches)
else:
    height_cm = st.sidebar.number_input("Height (cm):", min_value=100.0, max_value=250.0, value=178.0, step=0.1, format="%.1f")

st.sidebar.subheader("🏃‍♀️ Activity Profile (Step-Based)")
avg_daily_steps = st.sidebar.number_input("Average Total Daily Steps (from pedometer):", min_value=0, max_value=50000, value=7500, step=100)
additional_exercise_kcal_per_day = st.sidebar.number_input(
    "Additional Daily Exercise Kcal (non-step based or very intense cardio):",
    min_value=0, max_value=2000, value=0, step=25, help="Estimate for cycling, weights, etc., not captured by general step cost."
)

st.sidebar.subheader("🍽️ Diet (Target or Current Average)")
avg_daily_kcal_intake_reported = st.sidebar.number_input("Reported Average Daily Caloric Intake (kcal):", min_value=500, max_value=10000, value=2500, step=50)
protein_g_per_day = st.sidebar.number_input("Protein Intake (grams per day):", min_value=0.0, max_value=500.0, value=150.0, step=1.0, format="%.1f")

st.sidebar.subheader("⚖️ Observed Weight Trend (for intake calibration)")
st.sidebar.caption("This helps calibrate your *true* current intake from reported values.")
weight_trend = st.sidebar.selectbox("Recent Body Weight Trend:", ("Steady", "Gaining", "Losing"))
weight_change_rate_input_val = 0.0 # This will store the value as entered by user in their chosen unit
weight_change_rate_kg_wk = 0.0 # This will be kg/week for calculations
if weight_trend != "Steady":
    default_rate_lbs = 0.5; default_rate_kg = 0.23
    if weight_unit == "lbs":
        weight_change_rate_input_val = st.sidebar.number_input(f"Rate of {weight_trend.lower()} (lbs/week):", min_value=0.01, max_value=5.0, value=default_rate_lbs, step=0.05, format="%.2f")
        weight_change_rate_kg_wk = lbs_to_kg(weight_change_rate_input_val)
    else: # kg
        weight_change_rate_input_val_kg = st.sidebar.number_input(f"Rate of {weight_trend.lower()} (kg/week):", min_value=0.01, max_value=2.5, value=default_rate_kg, step=0.01, format="%.2f")
        weight_change_rate_kg_wk = weight_change_rate_input_val_kg
        weight_change_rate_input_val = weight_change_rate_input_val_kg # Store the kg value if input was kg
else: 
    weight_change_rate_input_val = 0.0
    weight_change_rate_kg_wk = 0.0

st.sidebar.subheader("🌡️ Environment & Physiology")
typical_indoor_temp_f = st.sidebar.slider("Typical Indoor Temperature (°F):", min_value=50, max_value=90, value=70, step=1)
minutes_cold_exposure_daily = st.sidebar.number_input("Avg. Min/Day Outdoors <60°F/15°C:", min_value=0, max_value=1440, value=0, step=15)
avg_sleep_hours = st.sidebar.slider("Habitual Nightly Sleep (hours):", min_value=4.0, max_value=12.0, value=7.5, step=0.1, format="%.1f")
uses_caffeine = st.sidebar.checkbox("Regular Caffeine User?", value=True)
has_fever_illness = st.sidebar.checkbox("Current Fever or Acute Illness?", value=False)
peak_fever_temp_f_input = 98.6
if has_fever_illness:
    peak_fever_temp_f_input = st.sidebar.number_input("Peak Fever Temp (°F):", min_value=98.6, max_value=106.0, value=100.0, step=0.1, format="%.1f")

num_days_to_simulate = st.sidebar.slider("Simulation Duration (days for TDEE graph):", 7, 90, 14, 7)

st.header("📊 Results & Analysis")

if st.sidebar.button("🚀 Calculate & Simulate TDEE", type="primary", use_container_width=True):
    if height_cm <= 0:
        st.error("Height must be a positive value. Please check your inputs.")
    else:
        ffm_kg, fm_kg = calculate_ffm_fm(weight_kg, body_fat_percentage)
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m**2) if height_m > 0 else 0

        st.subheader("📋 Initial Body Composition & Indices")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Weight", f"{weight_kg:.1f} kg / {kg_to_lbs(weight_kg):.1f} lbs")
        res_col2.metric("FFM", f"{ffm_kg:.1f} kg")
        res_col3.metric("FM", f"{fm_kg:.1f} kg ({body_fat_percentage:.1f}%)")
        res_col4.metric("BMI", f"{bmi:.1f}")

        current_ffmi, current_fmi = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
        idx_col1, idx_col2 = st.columns(2)
        idx_col1.metric("Fat-Free Mass Index (FFMI)", f"{current_ffmi:.1f} kg/m²")
        idx_col2.metric("Fat Mass Index (FMI)", f"{current_fmi:.1f} kg/m²")
        
        initial_bmr_for_display = calculate_pontzer_ffm_based_rmr(ffm_kg, fm_kg)
        if initial_bmr_for_display == 0: 
            initial_bmr_for_display = calculate_mifflin_st_jeor_rmr(weight_kg, height_cm, age_years, sex)
        
        lower_bound_tdee_static_display = initial_bmr_for_display * 1.2
        upper_bound_tdee_static_display = calculate_dlw_tdee(ffm_kg, fm_kg)
        if upper_bound_tdee_static_display == 0 or upper_bound_tdee_static_display < lower_bound_tdee_static_display:
            pal_heuristic_for_display = get_pal_multiplier_for_heuristic(avg_daily_steps)
            upper_bound_tdee_static_display = initial_bmr_for_display * pal_heuristic_for_display * 1.05

        st.subheader("↔️ Estimated Static Metabolic Range (Reference Only)")
        st.markdown(f"""
        - **Static Lower Adaptive Bound (approx. tanked BMR+NEAT): `{lower_bound_tdee_static_display:,.0f} kcal/day`** (Pontzer FFM-RMR * 1.2)
        - **Static Upper Adaptive Bound (Typical Free-living DLW TDEE): `{upper_bound_tdee_static_display:,.0f} kcal/day`**
        """)
        
        # Implied Activity Section
        implied_activity_energy, implied_total_steps = calculate_implied_activity_and_steps(
            upper_bound_tdee_static_display, # This is the FFM-based DLW TDEE
            initial_bmr_for_display,       # This is the Pontzer FFM-based RMR
            weight_kg
        )
        st.subheader("🚶‍♂️Implied Activity for FFM-Based TDEE")
        st.markdown(f"""
        The FFM-Based TDEE (DLW) of **`{upper_bound_tdee_static_display:,.0f} kcal/day`** for your body composition inherently includes a certain level of activity.
        - Estimated RMR (Pontzer FFM-based): `{initial_bmr_for_display:,.0f} kcal/day`
        - Estimated TEF (at maintenance): `{upper_bound_tdee_static_display * 0.10:,.0f} kcal/day`
        - This implies an activity energy (EAT + NEAT) of: **`{implied_activity_energy:,.0f} kcal/day`**
        - Equivalent to approximately: **`{implied_total_steps:,.0f} steps/day`** (plus any non-step exercise).
        
        Compare this to your current average daily steps (`{avg_daily_steps:,.0f}`) and additional exercise (`{additional_exercise_kcal_per_day:,.0f} kcal`). 
        If your activity is lower, you might need to adjust intake downwards from this FFM-based TDEE to maintain your weight, or increase activity to match.
        """)

        adjusted_true_intake = adjust_reported_intake(avg_daily_kcal_intake_reported, weight_trend, weight_change_rate_kg_wk)
        st.metric("Reported Avg. Daily Intake", f"{avg_daily_kcal_intake_reported:,.0f} kcal")

        if abs(adjusted_true_intake - avg_daily_kcal_intake_reported) > 20:
            unit_for_trend_display = "kg" if (weight_unit == "kg" and weight_trend != "Steady") else "lbs"
            display_rate_val_for_caption = weight_change_rate_kg_wk if unit_for_trend_display == "kg" else weight_change_rate_input_val

            if weight_trend != "Steady" :
                 st.metric("Calibrated True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")
                 st.caption(f"Adjusted based on reported weight trend of {display_rate_val_for_caption:.2f} {unit_for_trend_display}/week ({weight_trend}).")
            else: 
                 st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")
        else:
            st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")

        simulation_inputs = {
            "weight_kg": weight_kg, "body_fat_percentage": body_fat_percentage,
            "sex": sex, "age_years": age_years, "height_cm": height_cm,
            "avg_daily_steps": avg_daily_steps, 
            "additional_exercise_kcal_per_day": additional_exercise_kcal_per_day,
            "adjusted_intake": adjusted_true_intake,
            "protein_g_per_day": protein_g_per_day,
            "typical_indoor_temp_f": typical_indoor_temp_f,
            "minutes_cold_exposure_daily": minutes_cold_exposure_daily,
            "avg_sleep_hours": avg_sleep_hours, "uses_caffeine": uses_caffeine,
            "has_fever_illness": has_fever_illness, "peak_fever_temp_f": peak_fever_temp_f_input,
            "streamlit_object": st 
        }

        st.subheader(f"⏳ Simulated TDEE Adaptation Over {num_days_to_simulate} Days")
        st.caption(f"Based on maintaining a calibrated true intake of **{adjusted_true_intake:,.0f} kcal/day**.")
        
        daily_tdee_log_df, final_tdee_states = simulate_tdee_adaptation(simulation_inputs, num_days_to_simulate)

        if not daily_tdee_log_df.empty:
            current_dynamic_tdee = final_tdee_states['final_tdee']
            LAB_sim = final_tdee_states['LAB']; UAB_sim = final_tdee_states['UAB']
            st.metric(f"Simulated Dynamic TDEE (at Day {num_days_to_simulate})", f"{current_dynamic_tdee:,.0f} kcal/day")
            st.caption(f"Primary Metabolic Adaptive Range used in simulation: {LAB_sim:,.0f} - {UAB_sim:,.0f} kcal/day. Critical Low Intake mode: {'Active' if final_tdee_states.get('is_critically_low_intake_scenario') else 'Inactive'}")

            with st.expander("Show Detailed Daily Simulation Log & Component Breakdown"):
                cols_to_display = [col for col in ["Day", "Target Intake", "BMR_Adaptive", "TEF", "EAT", "NEAT_Adaptive", "Cold_Thermo", "Fever_Effect", "Total_Dynamic_TDEE", "Energy_Balance_vs_TDEE", "Target_BMR_DailyStep", "Target_NEAT_DailyStep"] if col in daily_tdee_log_df.columns]
                display_df_log = daily_tdee_log_df[cols_to_display]
                st.dataframe(display_df_log.style.format("{:,.0f}", na_rep="-", subset=pd.IndexSlice[:, [col for col in cols_to_display if col != 'Day']]))
            
            chart_cols_to_plot = ['Total_Dynamic_TDEE', 'BMR_Adaptive', 'NEAT_Adaptive', 'TEF', 'EAT']
            valid_chart_cols = [col for col in chart_cols_to_plot if col in daily_tdee_log_df.columns and daily_tdee_log_df[col].abs().sum() > 0.01]
            if 'Day' in daily_tdee_log_df.columns and len(valid_chart_cols) > 0:
                chart_data = daily_tdee_log_df[['Day'] + valid_chart_cols].copy()
                chart_data.set_index('Day', inplace=True)
                st.line_chart(chart_data)
                st.caption("Chart shows key TDEE components adapting over the simulation period.")
        else:
            st.error("Simulation failed to produce results.")
            current_dynamic_tdee = (lower_bound_tdee_static_display + upper_bound_tdee_static_display) / 2.0

        st.subheader("🎯 Nutritional Strategy Assessment")
        advice, overall_status_msg, daily_surplus_deficit_val = generate_bulk_cut_assessment(
            adjusted_true_intake, current_dynamic_tdee,
            final_tdee_states.get('initial_bmr_baseline', initial_bmr_for_display), 
            upper_bound_tdee_static_display, 
            ffm_kg, fm_kg, height_m, bmi, sex,
            final_tdee_states
        )
        
        if "SURPLUS" in overall_status_msg.upper() and "CUT" not in advice.upper(): st.success(f"{overall_status_msg}")
        elif "DEFICIT" in overall_status_msg.upper() and "BULK" not in advice.upper(): st.error(f"{overall_status_msg}")
        elif "COMPLEX" in advice.upper() or "CONSULT" in advice.upper() or "CAREFUL" in advice.upper() : st.warning(f"{overall_status_msg}")
        else: st.info(f"{overall_status_msg}")
        st.markdown(advice)

        st.subheader("📅 Future Intake Scenarios & Projected Weight Change")
        st.caption(f"Based on your simulated dynamic TDEE of **{current_dynamic_tdee:,.0f} kcal/day** (after {num_days_to_simulate}-day adaptation).")
        df_weight_scenarios = project_weight_change_scenarios(current_dynamic_tdee, weight_kg)
        st.dataframe(df_weight_scenarios, hide_index=True)
        
        st.success("✅ Analysis Complete!")
else:
    st.info("👈 Please fill in your details in the sidebar and click 'Calculate & Simulate TDEE'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Model based on concepts from user-provided research document and FMI/FFMI hazard ratio chart.")
st.markdown("---")
st.markdown("""
    **Disclaimer:** This tool provides estimates based on scientific literature and mathematical modeling.
    Individual metabolic responses can vary significantly. These results are for informational purposes only
    and do not constitute medical or nutritional advice. Always consult with qualified healthcare professionals
    or registered dietitians before making changes to your diet or exercise regimen.
    The accuracy of this model depends heavily on the accuracy of your inputs and the inherent simplifications
    in modeling complex human physiology. FMI/FFMI hazard ratio interpretations are based on visual approximation
    of a provided chart and are illustrative.
    """)
