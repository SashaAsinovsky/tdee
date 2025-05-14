import streamlit as st
import numpy as np
import pandas as pd

# --- Constants and Coefficients ---
KCAL_PER_KG_TISSUE = 7700
SURPLUS_EFFICIENCY = 0.85
DEFICIT_EFFICIENCY = 1.00
TAU_BMR_ADAPTATION = 10.0
TAU_NEAT_ADAPTATION = 2.0

# --- Helper Functions ---
def kg_to_lbs(kg): return kg * 2.20462
def lbs_to_kg(lbs): return lbs / 2.20462
def ft_in_to_cm(ft, inch): return (ft * 30.48) + (inch * 2.54)

# --- Core Calculation Functions ---
def calculate_ffm_fm(weight_kg, body_fat_percentage):
    fm_kg = weight_kg * (body_fat_percentage / 100.0)
    ffm_kg = weight_kg - fm_kg
    return ffm_kg, fm_kg

def calculate_mifflin_st_jeor_rmr(weight_kg, height_cm, age_years, sex):
    if sex == "Male":
        rmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) + 5
    else:
        rmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) - 161
    return rmr

def calculate_dlw_tdee(ffm_kg, fm_kg):
    if ffm_kg <= 0: return 0
    try:
        term_ffm = 0.916 * np.log(ffm_kg)
        term_fm = -0.030 * np.log(fm_kg if fm_kg > 0.001 else 0.001)
        tdee = np.exp(-1.102 + term_ffm + term_fm) * 238.83
        return tdee
    except Exception: return 0

def get_pal_multiplier(activity_level_str):
    pal_map = {"Sedentary": 1.2, "Lightly Active": 1.375, "Moderately Active": 1.55, "Very Active": 1.725, "Athlete": 1.9}
    return pal_map.get(activity_level_str, 1.55)

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
    k_c_per_degree_f_day = 4.0
    comfort_temp_f = 68.0
    cold_kcal_indoor = 0.0
    if typical_indoor_temp_f < comfort_temp_f:
        temp_diff_f_indoor = comfort_temp_f - typical_indoor_temp_f
        cold_kcal_indoor = temp_diff_f_indoor * k_c_per_degree_f_day * ((24.0 * 60.0 - minutes_cold_exposure_daily) / (24.0 * 60.0))
    cold_kcal_outdoor = (minutes_cold_exposure_daily / 60.0) * 30.0
    return max(0, cold_kcal_indoor) + max(0, cold_kcal_outdoor)

def calculate_immune_fever_effect(has_fever_illness, peak_fever_temp_f, current_bmr_adaptive):
    if not has_fever_illness or peak_fever_temp_f <= 99.0: return 0.0
    temp_diff_f = peak_fever_temp_f - 98.6
    if temp_diff_f <= 0: return 0.0
    return current_bmr_adaptive * temp_diff_f * 0.065

def calculate_ffmi_fmi(ffm_kg, fm_kg, height_m):
    ffmi = ffm_kg / (height_m**2) if height_m > 0 and ffm_kg >= 0 else 0
    fmi = fm_kg / (height_m**2) if height_m > 0 and fm_kg >= 0 else 0
    return ffmi, fmi

# --- Main Simulation Logic (Revised with NEAT fix) ---
def simulate_tdee_adaptation(inputs, num_days_to_simulate=14):
    ffm_kg, fm_kg = calculate_ffm_fm(inputs['weight_kg'], inputs['body_fat_percentage'])
    initial_bmr_mifflin = calculate_mifflin_st_jeor_rmr(inputs['weight_kg'], inputs['height_cm'], inputs['age_years'], inputs['sex'])
    pal_multiplier = get_pal_multiplier(inputs['activity_level_str'])

    LAB = initial_bmr_mifflin * 1.2
    UAB = calculate_dlw_tdee(ffm_kg, fm_kg)
    if UAB == 0 or UAB < LAB * 1.1:
        UAB = LAB * 1.25 + 200
        if 'streamlit_object' in inputs and inputs['streamlit_object'] is not None:
             inputs['streamlit_object'].warning(f"DLW TDEE for UAB was low/zero ({calculate_dlw_tdee(ffm_kg, fm_kg):.0f}). Upper Adaptive Bound set heuristically to {UAB:.0f} kcal.")


    target_true_intake_for_sim = inputs['adjusted_intake']

    day0_bmr = initial_bmr_mifflin
    day0_eat = (pal_multiplier - 1.0) * day0_bmr
    day0_tef = calculate_tef(target_true_intake_for_sim, inputs['protein_g_per_day'])
    cold_kcal_fixed = calculate_cold_thermogenesis(inputs['typical_indoor_temp_f'], inputs['minutes_cold_exposure_daily'])
    day0_fever = calculate_immune_fever_effect(inputs['has_fever_illness'], inputs['peak_fever_temp_f'], day0_bmr)
    day0_neat_adaptive = 0.0

    TDEE_sim_start = day0_bmr + day0_tef + day0_eat + day0_neat_adaptive + cold_kcal_fixed + day0_fever

    current_bmr_adaptive = day0_bmr
    current_neat_adaptive_component = day0_neat_adaptive

    intake_in_adaptive_range = LAB <= target_true_intake_for_sim <= UAB
    total_adaptation_gap = 0.0 # Ensure it's float

    if intake_in_adaptive_range:
        total_adaptation_gap = target_true_intake_for_sim - TDEE_sim_start
        target_total_neat_change = total_adaptation_gap * (0.60 if total_adaptation_gap > 0 else 0.30)
        target_total_bmr_change = total_adaptation_gap * (0.40 if total_adaptation_gap > 0 else 0.70)
    else: # Initialize to prevent UnboundLocalError if not set above
        target_total_neat_change = 0.0
        target_total_bmr_change = 0.0


    final_target_bmr_adaptive_in_range = initial_bmr_mifflin + target_total_bmr_change
    min_bmr_target_limit = initial_bmr_mifflin * 0.80
    max_bmr_target_limit = initial_bmr_mifflin * 1.20
    final_target_bmr_adaptive_in_range = np.clip(final_target_bmr_adaptive_in_range, min_bmr_target_limit, max_bmr_target_limit)

    final_target_neat_adaptive_in_range = target_total_neat_change
    final_target_neat_adaptive_in_range = np.clip(final_target_neat_adaptive_in_range, -350, 700)

    daily_log = []

    for day in range(num_days_to_simulate):
        eat_kcal = (pal_multiplier - 1.0) * current_bmr_adaptive
        tef_kcal = calculate_tef(target_true_intake_for_sim, inputs['protein_g_per_day'])
        fever_kcal = calculate_immune_fever_effect(inputs['has_fever_illness'], inputs['peak_fever_temp_f'], current_bmr_adaptive)

        day_target_bmr = final_target_bmr_adaptive_in_range # Default target if in adaptive range
        day_target_neat = final_target_neat_adaptive_in_range # Default target if in adaptive range

        if not intake_in_adaptive_range:
            current_expenditure_for_balance = current_bmr_adaptive + tef_kcal + eat_kcal + current_neat_adaptive_component + cold_kcal_fixed + fever_kcal
            energy_balance = target_true_intake_for_sim - current_expenditure_for_balance

            bmr_target_change_factor_ext = 0.0
            if energy_balance > 250: bmr_target_change_factor_ext = 0.05
            elif energy_balance < -250: bmr_target_change_factor_ext = -0.10
            day_target_bmr = initial_bmr_mifflin * (1 + bmr_target_change_factor_ext)
            day_target_bmr = np.clip(day_target_bmr, min_bmr_target_limit, max_bmr_target_limit)

            neat_responsiveness = 0.30
            if inputs['avg_daily_steps'] > 10000: neat_responsiveness += 0.10
            if inputs['avg_daily_steps'] < 5000: neat_responsiveness -= 0.10
            if inputs['avg_sleep_hours'] < 6.5: neat_responsiveness -= 0.05
            if inputs['uses_caffeine']: neat_responsiveness += 0.05
            neat_responsiveness = np.clip(neat_responsiveness, 0.1, 0.5)
            
            day_target_neat = energy_balance * neat_responsiveness # CORRECTED: NEAT increases in surplus
            day_target_neat = np.clip(day_target_neat, -350, 700)

        if day == 0 and intake_in_adaptive_range and total_adaptation_gap != 0:
            bmr_change_to_apply_day0 = (day_target_bmr - initial_bmr_mifflin) * 0.40
            neat_change_to_apply_day0 = (day_target_neat - 0) * 0.40 # Change from initial NEAT of 0

            current_bmr_adaptive = initial_bmr_mifflin + bmr_change_to_apply_day0
            current_neat_adaptive_component = 0 + neat_change_to_apply_day0
        else:
            delta_bmr = (day_target_bmr - current_bmr_adaptive) / TAU_BMR_ADAPTATION
            current_bmr_adaptive += delta_bmr
            delta_neat = (day_target_neat - current_neat_adaptive_component) / TAU_NEAT_ADAPTATION
            current_neat_adaptive_component += delta_neat
        
        current_bmr_adaptive = np.clip(current_bmr_adaptive, min_bmr_target_limit * 0.95 , max_bmr_target_limit * 1.05)
        current_neat_adaptive_component = np.clip(current_neat_adaptive_component, -450, 750)

        total_tdee_today = current_bmr_adaptive + tef_kcal + eat_kcal + current_neat_adaptive_component + cold_kcal_fixed + fever_kcal
        
        daily_log.append({
            "Day": day + 1, "Target Intake": target_true_intake_for_sim,
            "BMR_Adaptive": current_bmr_adaptive, "TEF": tef_kcal, "EAT": eat_kcal,
            "NEAT_Adaptive": current_neat_adaptive_component, "Cold_Thermo": cold_kcal_fixed,
            "Fever_Effect": fever_kcal, "Total_Dynamic_TDEE": total_tdee_today,
            "Energy_Balance_vs_TDEE": target_true_intake_for_sim - total_tdee_today,
            "Target_BMR_DailyStep": day_target_bmr, "Target_NEAT_DailyStep": day_target_neat
        })

    final_eat_kcal = (pal_multiplier - 1.0) * current_bmr_adaptive
    final_tef_kcal = calculate_tef(target_true_intake_for_sim, inputs['protein_g_per_day'])
    final_fever_kcal = calculate_immune_fever_effect(inputs['has_fever_illness'], inputs['peak_fever_temp_f'], current_bmr_adaptive)

    final_tdee_val = daily_log[-1]['Total_Dynamic_TDEE'] if daily_log else TDEE_sim_start
    final_states = {
        "final_bmr_adaptive": current_bmr_adaptive,
        "final_neat_adaptive_component": current_neat_adaptive_component,
        "final_tdee": final_tdee_val,
        "fixed_tef": final_tef_kcal, "fixed_eat_final_bmr": final_eat_kcal,
        "fixed_cold_thermo": cold_kcal_fixed, "fixed_fever_effect": final_fever_kcal,
        "LAB": LAB, "UAB": UAB, "intake_in_adaptive_range": intake_in_adaptive_range
    }
    return pd.DataFrame(daily_log), final_states

def generate_bulk_cut_assessment(
    adjusted_intake, dynamic_tdee,
    lower_bound_static_tdee, upper_bound_static_tdee,
    ffm_kg, fm_kg, height_m, bmi, sex,
    sim_final_states
    ):
    ffmi, fmi = calculate_ffmi_fmi(ffm_kg, fm_kg, height_m)
    LAB = sim_final_states.get('LAB', lower_bound_static_tdee)
    UAB = sim_final_states.get('UAB', upper_bound_static_tdee)
    intake_was_in_adaptive_range = sim_final_states.get('intake_in_adaptive_range', False)

    advice_primary = ""
    status_caloric = ""
    daily_surplus_deficit_vs_dynamic_tdee = adjusted_intake - dynamic_tdee

    if daily_surplus_deficit_vs_dynamic_tdee > 25:
        status_caloric = "Surplus"
        advice_primary = (f"Your intake ({adjusted_intake:,.0f} kcal) suggests a caloric surplus of **{daily_surplus_deficit_vs_dynamic_tdee:+.0f} kcal/day** vs. your simulated dynamic TDEE ({dynamic_tdee:,.0f} kcal). Supports weight gain.\n")
    elif daily_surplus_deficit_vs_dynamic_tdee < -25:
        status_caloric = "Deficit"
        advice_primary = (f"Your intake ({adjusted_intake:,.0f} kcal) suggests a caloric deficit of **{daily_surplus_deficit_vs_dynamic_tdee:+.0f} kcal/day** vs. your simulated dynamic TDEE ({dynamic_tdee:,.0f} kcal). Supports weight loss.\n")
    else:
        status_caloric = "Maintenance"
        advice_primary = (f"Your intake ({adjusted_intake:,.0f} kcal) is closely aligned with your simulated dynamic TDEE ({dynamic_tdee:,.0f} kcal, diff: {daily_surplus_deficit_vs_dynamic_tdee:+.0f} kcal/day). Supports weight maintenance.\n")
    
    if intake_was_in_adaptive_range:
        advice_primary += (f"Your intake fell within the estimated primary metabolic adaptive range ({LAB:,.0f} - {UAB:,.0f} kcal). The simulation shows your TDEE has largely adapted.\n")
    else:
        advice_primary += (f"Your intake was outside the estimated primary metabolic adaptive range ({LAB:,.0f} - {UAB:,.0f} kcal). Some adaptation occurred, but intake will primarily drive tissue change.\n")

    advice_composition = ""
    fmi_hr_status = ""; ffmi_hr_status = ""; ffmi_direct_bulk_triggered = False

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

    final_recommendation = ""
    overall_status_message = f"Caloric: {status_caloric} | FMI: {fmi_hr_status} | FFMI: {ffmi_hr_status}"

    if ffmi_direct_bulk_triggered:
        if fmi_hr_status == "High (Suggests HR > 1)": final_recommendation = "REC: Body Recomp/Careful Lean Bulk. FFMI low, FMI high. Complex. Consult pro."
        else:
            final_recommendation = f"REC: BULK. FFMI ({ffmi:.1f}) < {FFMI_DIRECT_BULK_THRESHOLD}. Needs surplus."
            if status_caloric == "Deficit": final_recommendation += " Current intake deficit; increase."
            elif status_caloric == "Maintenance": final_recommendation += " Current intake maintenance; needs surplus."
    elif fmi_hr_status == "High (Suggests HR > 1)":
        final_recommendation = f"REC: CUT. FMI ({fmi:.1f}) high. Needs deficit."
        if status_caloric == "Surplus": final_recommendation += " Current intake surplus; decrease."
        elif status_caloric == "Maintenance": final_recommendation += " Current intake maintenance; needs deficit."
    elif ffmi_hr_status == "Low (Suggests HR > 1)":
        final_recommendation = f"REC: Consider Lean Bulk. FFMI ({ffmi:.1f}) suggests HR > 1 risk."
        if status_caloric == "Deficit": final_recommendation += " Current intake deficit; adjust."
        elif status_caloric == "Maintenance": final_recommendation += " Current intake maintenance; slight surplus?"
    elif status_caloric == "Surplus": final_recommendation = "REC: SURPLUS. If bulking, aligns. Monitor."
    elif status_caloric == "Deficit": final_recommendation = "REC: DEFICIT. If cutting, aligns. Preserve muscle."
    elif status_caloric == "Maintenance":
        if ffmi_hr_status.startswith("Sufficient/Optimal") and fmi_hr_status.startswith("Optimal"): final_recommendation = "REC: MAINTAIN/Optimize. Healthy comp. Adjust for specific goals."
        else: final_recommendation = "REC: MAINTAIN & RE-EVALUATE. Review FMI/FFMI for next phase."
    else: final_recommendation = "Review goals with FMI/FFMI context. Adjust intake."

    full_advice = (f"{advice_primary}\n\n**Body Composition Insights (FMI/FFMI & Approx. HR):**\n{advice_composition}\n**Overall Strategy Guidance:**\n{final_recommendation}")
    return full_advice, overall_status_message, daily_surplus_deficit_vs_dynamic_tdee

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
        eff_factor = SURPLUS_EFFICIENCY if daily_delta_vs_tdee > 0 else (DEFICIT_EFFICIENCY if daily_delta_vs_tdee < 0 else 1.0) # Ensure eff_factor is float
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

st.sidebar.subheader("🏃‍♀️ Activity Profile")
activity_level_str = st.sidebar.selectbox("Formal Activity Level (PAL basis):", ("Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Athlete"), index=2)
avg_daily_steps = st.sidebar.number_input("Avg. Daily Step Count (for NEAT model):", min_value=0, max_value=50000, value=7000, step=100)

st.sidebar.subheader("🍽️ Diet (Target or Current Average)")
avg_daily_kcal_intake_reported = st.sidebar.number_input("Reported Average Daily Caloric Intake (kcal):", min_value=500, max_value=10000, value=2500, step=50)
protein_g_per_day = st.sidebar.number_input("Protein Intake (grams per day):", min_value=0.0, max_value=500.0, value=150.0, step=1.0, format="%.1f")

st.sidebar.subheader("⚖️ Observed Weight Trend (for intake calibration)")
st.sidebar.caption("This helps calibrate your *true* current intake from reported values.")
weight_trend = st.sidebar.selectbox("Recent Body Weight Trend:", ("Steady", "Gaining", "Losing"))
weight_change_rate_input_val = 0.0
weight_change_rate_kg_wk = 0.0
if weight_trend != "Steady":
    default_rate_lbs = 0.5; default_rate_kg = 0.23
    if weight_unit == "lbs":
        weight_change_rate_input_val = st.sidebar.number_input(f"Rate of {weight_trend.lower()} (lbs/week):", min_value=0.01, max_value=5.0, value=default_rate_lbs, step=0.05, format="%.2f")
        weight_change_rate_kg_wk = lbs_to_kg(weight_change_rate_input_val)
    else:
        weight_change_rate_input_val_kg = st.sidebar.number_input(f"Rate of {weight_trend.lower()} (kg/week):", min_value=0.01, max_value=2.5, value=default_rate_kg, step=0.01, format="%.2f")
        weight_change_rate_kg_wk = weight_change_rate_input_val_kg
        weight_change_rate_input_val = kg_to_lbs(weight_change_rate_input_val_kg) # For display consistency
else: # Steady
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
    # Ensure height_cm is positive before calculating height_m
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

        rmr_mifflin = calculate_mifflin_st_jeor_rmr(weight_kg, height_cm, age_years, sex)
        lower_bound_tdee_static_ref = rmr_mifflin
        upper_bound_tdee_static_ref = calculate_dlw_tdee(ffm_kg, fm_kg)
        if upper_bound_tdee_static_ref == 0 or upper_bound_tdee_static_ref < lower_bound_tdee_static_ref * 1.2:
            # This warning is now also inside simulate_tdee_adaptation for UAB calculation
            upper_bound_tdee_static_ref = lower_bound_tdee_static_ref * get_pal_multiplier(activity_level_str) * 1.05

        st.subheader("↔️ Estimated Static Metabolic Range (Reference Only)")
        st.markdown(f"""
        - **Static Lower Bound (Bed-rest RMR): `{lower_bound_tdee_static_ref:,.0f} kcal/day`** (Mifflin-St Jeor)
        - **Static Upper Bound (Typical Free-living DLW Average): `{upper_bound_tdee_static_ref:,.0f} kcal/day`**
        """)
        st.caption("The simulation models adaptation within a dynamically calculated primary adaptive range (LAB-UAB).")

        adjusted_true_intake = adjust_reported_intake(avg_daily_kcal_intake_reported, weight_trend, weight_change_rate_kg_wk)
        st.metric("Reported Avg. Daily Intake", f"{avg_daily_kcal_intake_reported:,.0f} kcal")
        if abs(adjusted_true_intake - avg_daily_kcal_intake_reported) > 20:
            display_rate_trend = weight_change_rate_input_val if weight_unit == "lbs" else lbs_to_kg(weight_change_rate_input_val) if weight_trend != "Steady" else 0.0
            unit_for_trend_display = "lbs" if weight_unit == "lbs" else "kg"
            if weight_trend != "Steady" : # Only show rate if not steady
                 st.metric("Calibrated True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")
                 st.caption(f"Adjusted based on reported weight trend of {weight_change_rate_input_val:.2f} {unit_for_trend_display}/week ({weight_trend}).")
            else:
                 st.metric("True Daily Intake (for simulation)", f"{adjusted_true_intake:,.0f} kcal")


        simulation_inputs = {
            "weight_kg": weight_kg, "body_fat_percentage": body_fat_percentage,
            "sex": sex, "age_years": age_years, "height_cm": height_cm,
            "activity_level_str": activity_level_str, "avg_daily_steps": avg_daily_steps,
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
            st.caption(f"Primary Metabolic Adaptive Range used in simulation: {LAB_sim:,.0f} - {UAB_sim:,.0f} kcal/day.")

            with st.expander("Show Detailed Daily Simulation Log & Component Breakdown"):
                cols_to_display = [col for col in ["Day", "Target Intake", "BMR_Adaptive", "TEF", "EAT", "NEAT_Adaptive", "Cold_Thermo", "Fever_Effect", "Total_Dynamic_TDEE", "Energy_Balance_vs_TDEE", "Target_BMR_DailyStep", "Target_NEAT_DailyStep"] if col in daily_tdee_log_df.columns]
                display_df_log = daily_tdee_log_df[cols_to_display]
                st.dataframe(display_df_log.style.format("{:,.0f}", na_rep="-", subset=pd.IndexSlice[:, [col for col in cols_to_display if col != 'Day']]))
            
            chart_cols_to_plot = ['Total_Dynamic_TDEE', 'BMR_Adaptive', 'NEAT_Adaptive', 'TEF', 'EAT']
            valid_chart_cols = [col for col in chart_cols_to_plot if col in daily_tdee_log_df.columns and daily_tdee_log_df[col].abs().sum() > 0.1]
            if 'Day' in daily_tdee_log_df.columns and len(valid_chart_cols) > 0:
                chart_data = daily_tdee_log_df[['Day'] + valid_chart_cols].copy()
                chart_data.set_index('Day', inplace=True)
                st.line_chart(chart_data)
                st.caption("Chart shows key TDEE components adapting over the simulation period.")
        else:
            st.error("Simulation failed to produce results.")
            current_dynamic_tdee = (lower_bound_tdee_static_ref + upper_bound_tdee_static_ref) / 2.0

        st.subheader("🎯 Nutritional Strategy Assessment")
        advice, overall_status_msg, daily_surplus_deficit_val = generate_bulk_cut_assessment(
            adjusted_true_intake, current_dynamic_tdee,
            lower_bound_tdee_static_ref, upper_bound_tdee_static_ref,
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
