import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

# Set page configuration
st.set_page_config(
    page_title="TDEE Modeler & Body Composition Analyzer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown("<div class='main-header'>TDEE Modeler & Body Composition Analyzer</div>", unsafe_allow_html=True)
st.markdown("""
This tool helps you understand your metabolism, model your TDEE (Total Daily Energy Expenditure) 
over time, and get personalized recommendations for bulking or cutting based on your 
current body composition.
""")

# ----------------- DATA MODELS -----------------

@dataclass
class TDEEComponent:
    """Base class for TDEE components"""
    name: str
    description: str
    onset_time: str  # e.g., "immediate", "days", "weeks"
    magnitude: str   # e.g., "5-15% of TDEE"
    
    def calculate(self, base_value: float, **kwargs) -> float:
        """Calculate the component's contribution to TDEE"""
        raise NotImplementedError("Subclasses must implement this method")

class BMR(TDEEComponent):
    """Basal Metabolic Rate component"""
    
    def __init__(self):
        super().__init__(
            name="Basal Metabolic Rate (BMR)",
            description="Energy used to maintain basic physiological functions at rest.",
            onset_time="Days to weeks for noticeable changes",
            magnitude="±5-15% of TDEE in short term adaptation"
        )
    
    def calculate(self, weight_kg: float, height_cm: float, age: int, sex: str, 
                 caloric_deficit: float = 0, days_in_deficit: int = 0,
                 lean_mass_kg: Optional[float] = None) -> float:
        """
        Calculate BMR using the Mifflin-St Jeor equation, with adaptive thermogenesis adjustments
        
        Args:
            weight_kg: Weight in kg
            height_cm: Height in cm
            age: Age in years
            sex: 'M' for male, 'F' for female
            caloric_deficit: Current daily caloric deficit/surplus (negative for deficit)
            days_in_deficit: Number of days in current deficit/surplus
            lean_mass_kg: Optional lean body mass in kg
            
        Returns:
            BMR in calories per day
        """
        # Base calculation using Mifflin-St Jeor
        if sex.upper() == 'M':
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
            
        # If we have lean mass data, refine the estimate
        if lean_mass_kg is not None:
            # Katch-McArdle Formula: BMR = 370 + (21.6 * LBM)
            bmr = 370 + (21.6 * lean_mass_kg)
        
        # Adjust for adaptive thermogenesis based on deficit and duration
        if abs(caloric_deficit) > 0 and days_in_deficit > 0:
            # Calculate adaptive response based on research data
            # Limit to max 15% reduction for severe deficits and long durations
            deficit_pct = min(abs(caloric_deficit) / 2500, 0.5)  # Cap at 50% deficit
            time_factor = min(days_in_deficit / 28, 1.0)  # Maximum effect after 4 weeks
            
            # Direction depends on surplus vs deficit
            direction = -1 if caloric_deficit < 0 else 0.5
            
            # Adjust BMR (up to 15% for deficit, up to 5% for surplus)
            max_adjustment = 0.15 if direction < 0 else 0.05
            adjustment = direction * deficit_pct * time_factor * max_adjustment
            
            bmr = bmr * (1 + adjustment)
            
        return bmr

class TEF(TDEEComponent):
    """Thermic Effect of Food component"""
    
    def __init__(self):
        super().__init__(
            name="Thermic Effect of Food (TEF)",
            description="Increase in energy expenditure after eating, due to digestion and absorption.",
            onset_time="Almost immediate (minutes after eating)",
            magnitude="≈10% of total intake on average"
        )
    
    def calculate(self, caloric_intake: float, protein_pct: float = 0.3, 
                 carb_pct: float = 0.4, fat_pct: float = 0.3) -> float:
        """
        Calculate TEF based on caloric intake and macronutrient distribution
        
        Args:
            caloric_intake: Total daily caloric intake
            protein_pct: Percentage of calories from protein (0-1)
            carb_pct: Percentage of calories from carbohydrates (0-1)
            fat_pct: Percentage of calories from fat (0-1)
            
        Returns:
            TEF in calories per day
        """
        # TEF by macronutrient: protein 20-30%, carbs 5-10%, fat 0-3%
        protein_tef = protein_pct * caloric_intake * 0.25  # 25% average for protein
        carb_tef = carb_pct * caloric_intake * 0.075      # 7.5% average for carbs
        fat_tef = fat_pct * caloric_intake * 0.015        # 1.5% average for fat
        
        return protein_tef + carb_tef + fat_tef

class EAT(TDEEComponent):
    """Exercise Activity Thermogenesis component"""
    
    def __init__(self):
        super().__init__(
            name="Exercise Activity Thermogenesis (EAT)",
            description="Energy expended during planned physical exercise.",
            onset_time="Immediate during activity",
            magnitude="Highly variable, 0-50% of TDEE depending on activity level"
        )
        
        # Define exercise intensities with MET values
        self.exercise_mets = {
            "Light": 3.0,    # Walking, light cycling
            "Moderate": 5.5,  # Jogging, recreational sports
            "Vigorous": 8.0,  # Running, intense cycling
            "Very Vigorous": 12.0  # HIIT, competitive sports
        }
    
    def calculate(self, weight_kg: float, exercise_minutes: dict, include_epoc: bool = True) -> float:
        """
        Calculate EAT based on exercise duration, intensity and body weight
        
        Args:
            weight_kg: Weight in kg
            exercise_minutes: Dict mapping intensity levels to minutes
            include_epoc: Whether to include excess post-exercise oxygen consumption
            
        Returns:
            EAT in calories per day
        """
        total_eat = 0
        total_epoc = 0
        
        for intensity, minutes in exercise_minutes.items():
            if intensity in self.exercise_mets and minutes > 0:
                # Formula: MET × weight (kg) × duration (hours)
                met_value = self.exercise_mets[intensity]
                hours = minutes / 60
                
                eat_calories = met_value * weight_kg * hours
                total_eat += eat_calories
                
                # Calculate EPOC based on exercise intensity and duration
                if include_epoc:
                    if intensity == "Light":
                        epoc_pct = 0.03  # 3% of exercise calories
                    elif intensity == "Moderate":
                        epoc_pct = 0.06  # 6% of exercise calories
                    elif intensity == "Vigorous":
                        epoc_pct = 0.10  # 10% of exercise calories
                    else:  # Very Vigorous
                        epoc_pct = 0.15  # 15% of exercise calories
                        
                    total_epoc += eat_calories * epoc_pct
        
        return total_eat + total_epoc

class NEAT(TDEEComponent):
    """Non-Exercise Activity Thermogenesis component"""
    
    def __init__(self):
        super().__init__(
            name="Non-Exercise Activity Thermogenesis (NEAT)",
            description="Energy expended on all physical activities excluding exercise.",
            onset_time="Immediate with posture/movement changes",
            magnitude="Most variable component, ~100-800 kcal/day typically"
        )
        
        # Define activity levels
        self.activity_multipliers = {
            "Sedentary": 1.2,      # Desk job, little activity
            "Lightly Active": 1.375,  # Standing job or some walking
            "Moderately Active": 1.55,  # Job with physical demands
            "Very Active": 1.725,    # Physical job with additional activity
            "Extremely Active": 1.9   # Heavy manual labor jobs
        }
    
    def calculate(self, bmr: float, activity_level: str, 
                 caloric_deficit: float = 0, days_in_deficit: int = 0,
                 overfeeding: bool = False) -> float:
        """
        Calculate NEAT based on activity level and adaptive thermogenesis
        
        Args:
            bmr: Basal Metabolic Rate in calories
            activity_level: Activity level category
            caloric_deficit: Current daily caloric deficit/surplus
            days_in_deficit: Number of days in current state
            overfeeding: Whether person is in a sustained caloric surplus
            
        Returns:
            NEAT in calories per day
        """
        # Get base multiplier from activity level
        multiplier = self.activity_multipliers.get(activity_level, 1.2)
        
        # Calculate base NEAT
        base_neat = bmr * (multiplier - 1)  # Subtract 1 because BMR is already counted
        
        # Adjust for adaptive thermogenesis
        if abs(caloric_deficit) > 0 and days_in_deficit > 0:
            deficit_pct = min(abs(caloric_deficit) / 2500, 0.5)  # Cap at 50% deficit
            time_factor = min(days_in_deficit / 7, 1.0)  # Maximum effect after 1 week
            
            # NEAT is more sensitive to caloric balance than BMR
            # Direction depends on surplus vs deficit
            if caloric_deficit < 0:  # Deficit
                # Can reduce NEAT by up to 25% in severe restriction
                adjustment = -deficit_pct * time_factor * 0.25
            else:  # Surplus
                # Overfeeding can increase NEAT, but with high individual variability
                # Some people increase NEAT a lot, others barely at all
                # Using a random factor to simulate individual variability
                individual_factor = np.random.uniform(0.1, 0.9) if overfeeding else 0.5
                adjustment = deficit_pct * time_factor * 0.2 * individual_factor
            
            base_neat = base_neat * (1 + adjustment)
            
        return base_neat

class Thermogenesis(TDEEComponent):
    """Thermoregulation and Cold-Induced Thermogenesis component"""
    
    def __init__(self):
        super().__init__(
            name="Thermoregulation (Cold-Induced Thermogenesis)",
            description="Heat production in response to cold environment.",
            onset_time="Seconds to minutes for onset",
            magnitude="5-20% increase for mild cold, much higher for shivering"
        )
    
    def calculate(self, bmr: float, temperature: float, 
                 brown_fat_activity: float = 0.5, cold_acclimated: bool = False,
                 duration_hours: float = 24.0) -> float:
        """
        Calculate thermogenesis based on environmental conditions
        
        Args:
            bmr: Basal Metabolic Rate in calories
            temperature: Environmental temperature in Celsius
            brown_fat_activity: Factor from 0-1 indicating brown fat activity level
            cold_acclimated: Whether the person is acclimated to cold
            duration_hours: Hours per day spent at this temperature
            
        Returns:
            Thermogenesis in calories per day
        """
        # Define temperature thresholds
        thermoneutral_upper = 27  # Upper limit of thermoneutral zone °C
        thermoneutral_lower = 23  # Lower limit of thermoneutral zone °C
        shivering_threshold = 16  # Temperature at which shivering typically begins °C
        
        # No additional thermogenesis in thermoneutral zone
        if thermoneutral_lower <= temperature <= thermoneutral_upper:
            return 0
        
        # Calculate thermogenesis for cold environments
        if temperature < thermoneutral_lower:
            # How far below thermoneutral zone
            cold_exposure = thermoneutral_lower - temperature
            
            # Base response - brown fat activation
            if temperature >= shivering_threshold:
                # Non-shivering thermogenesis only
                base_effect = 0.01 * cold_exposure * brown_fat_activity  # 1% of BMR per degree * BAT activity
                
                # Acclimated individuals have more efficient brown fat
                if cold_acclimated:
                    base_effect *= 2
                
                thermogenesis = bmr * base_effect
            else:
                # Temperature below shivering threshold - include shivering thermogenesis
                # Cold acclimated people shiver less but have more BAT activity
                non_shivering = 0.02 * cold_exposure * brown_fat_activity * (2 if cold_acclimated else 1)
                
                # Shivering intensity increases as temperature drops
                shiver_intensity = min((shivering_threshold - temperature) / 6, 1.0)
                shivering_factor = shiver_intensity * (0.3 if cold_acclimated else 0.5)  # Up to 50% of BMR, less if acclimated
                
                thermogenesis = bmr * (non_shivering + shivering_factor)
            
            # Pro-rate based on hours of exposure
            return thermogenesis * (duration_hours / 24.0)
        
        # For hot environments (above thermoneutral), there's a small cost for cooling
        # (e.g., sweating), but it's much smaller than cold thermogenesis
        else:
            hot_exposure = temperature - thermoneutral_upper
            # About 0.2% of BMR per degree above thermoneutral
            cooling_cost = bmr * 0.002 * hot_exposure * (duration_hours / 24.0)
            return cooling_cost

class ImmuneFunction(TDEEComponent):
    """Immune Activation and Fever component"""
    
    def __init__(self):
        super().__init__(
            name="Immune Activation & Fever",
            description="Energy expended fighting infection and maintaining elevated body temperature.",
            onset_time="Hours for fever to develop",
            magnitude="~10-12% increase in RMR per 1°C of fever"
        )
    
    def calculate(self, bmr: float, fever_celsius: float = 0, 
                 immune_activation: float = 0) -> float:
        """
        Calculate energy expenditure from immune function and fever
        
        Args:
            bmr: Basal Metabolic Rate in calories
            fever_celsius: Degrees Celsius above normal body temperature (37°C)
            immune_activation: Factor from 0-1 indicating immune system activation level
            
        Returns:
            Immune-related energy expenditure in calories per day
        """
        # No fever or immune activation
        if fever_celsius <= 0 and immune_activation <= 0:
            return 0
        
        # Calculate fever-related expenditure (~10-12% per °C)
        fever_effect = 0
        if fever_celsius > 0:
            fever_effect = bmr * 0.11 * fever_celsius  # 11% increase per degree
            
        # Calculate immune activation expenditure
        # Even without fever, fighting infection costs energy
        immune_effect = bmr * 0.05 * immune_activation  # Up to 5% of BMR for immune function
        
        return fever_effect + immune_effect

class ReproductiveFactors(TDEEComponent):
    """Hormonal and Reproductive Factors component"""
    
    def __init__(self):
        super().__init__(
            name="Hormonal and Reproductive Factors",
            description="Energy expenditure variations due to menstrual cycle and sexual activity.",
            onset_time="Gradual change over cycle (~monthly)",
            magnitude="Small increase in RMR in luteal phase (~2-10%)"
        )
    
    def calculate(self, bmr: float, sex: str, menstrual_phase: str = None,
                 sexual_activity_minutes: float = 0) -> float:
        """
        Calculate energy expenditure from reproductive and hormonal factors
        
        Args:
            bmr: Basal Metabolic Rate in calories
            sex: 'M' for male, 'F' for female
            menstrual_phase: For females, current phase of menstrual cycle
            sexual_activity_minutes: Minutes of sexual activity per day
            
        Returns:
            Reproductive-related energy expenditure in calories per day
        """
        hormonal_effect = 0
        
        # Menstrual cycle effects (females only)
        if sex.upper() == 'F' and menstrual_phase:
            if menstrual_phase == "Luteal":
                # Increased metabolism in luteal phase (post-ovulation)
                # Individual variation is high, using average of ~5%
                hormonal_effect = bmr * 0.05
            # Other phases assumed to be baseline or slightly below
        
        # Sexual activity energy expenditure
        # Research shows ~3-4 kcal per minute for sexual activity
        sexual_expenditure = 0
        if sexual_activity_minutes > 0:
            # Higher value for males due to typically higher activity level
            cal_per_minute = 4 if sex.upper() == 'M' else 3.5
            sexual_expenditure = sexual_activity_minutes * cal_per_minute
        
        return hormonal_effect + sexual_expenditure

class TDEECalculator:
    """Calculate TDEE by summing all components"""
    
    def __init__(self):
        # Initialize all TDEE components
        self.bmr = BMR()
        self.tef = TEF()
        self.eat = EAT()
        self.neat = NEAT()
        self.thermogenesis = Thermogenesis()
        self.immune = ImmuneFunction()
        self.reproductive = ReproductiveFactors()
    
    def calculate_tdee(self, user_data: dict) -> Dict[str, float]:
        """
        Calculate TDEE and all its components
        
        Args:
            user_data: Dictionary containing all required input parameters
            
        Returns:
            Dictionary with TDEE components and total
        """
        # Calculate BMR
        bmr_value = self.bmr.calculate(
            weight_kg=user_data.get('weight_kg', 70),
            height_cm=user_data.get('height_cm', 170),
            age=user_data.get('age', 30),
            sex=user_data.get('sex', 'M'),
            caloric_deficit=user_data.get('caloric_deficit', 0),
            days_in_deficit=user_data.get('days_in_deficit', 0),
            lean_mass_kg=user_data.get('lean_mass_kg')
        )
        
        # Calculate TEF
        tef_value = self.tef.calculate(
            caloric_intake=user_data.get('caloric_intake', 2000),
            protein_pct=user_data.get('protein_pct', 0.3),
            carb_pct=user_data.get('carb_pct', 0.4),
            fat_pct=user_data.get('fat_pct', 0.3)
        )
        
        # Calculate EAT
        eat_value = self.eat.calculate(
            weight_kg=user_data.get('weight_kg', 70),
            exercise_minutes=user_data.get('exercise_minutes', {'Moderate': 30}),
            include_epoc=user_data.get('include_epoc', True)
        )
        
        # Calculate NEAT
        neat_value = self.neat.calculate(
            bmr=bmr_value,
            activity_level=user_data.get('activity_level', 'Lightly Active'),
            caloric_deficit=user_data.get('caloric_deficit', 0),
            days_in_deficit=user_data.get('days_in_deficit', 0),
            overfeeding=user_data.get('caloric_deficit', 0) > 0
        )
        
        # Calculate Thermogenesis
        therm_value = self.thermogenesis.calculate(
            bmr=bmr_value,
            temperature=user_data.get('temperature', 22),
            brown_fat_activity=user_data.get('brown_fat_activity', 0.5),
            cold_acclimated=user_data.get('cold_acclimated', False),
            duration_hours=user_data.get('cold_exposure_hours', 24)
        )
        
        # Calculate Immune Function
        immune_value = self.immune.calculate(
            bmr=bmr_value,
            fever_celsius=user_data.get('fever_celsius', 0),
            immune_activation=user_data.get('immune_activation', 0)
        )
        
        # Calculate Reproductive Factors
        reproductive_value = self.reproductive.calculate(
            bmr=bmr_value,
            sex=user_data.get('sex', 'M'),
            menstrual_phase=user_data.get('menstrual_phase'),
            sexual_activity_minutes=user_data.get('sexual_activity_minutes', 0)
        )
        
        # Calculate total TDEE
        total_tdee = sum([
            bmr_value,
            tef_value,
            eat_value,
            neat_value,
            therm_value,
            immune_value,
            reproductive_value
        ])
        
        # Return all components and total
        return {
            'BMR': bmr_value,
            'TEF': tef_value,
            'EAT': eat_value,
            'NEAT': neat_value,
            'Thermogenesis': therm_value,
            'Immune': immune_value,
            'Reproductive': reproductive_value,
            'Total TDEE': total_tdee
        }

    def model_tdee_over_time(self, initial_data: dict, days: int = 30, 
                             daily_changes: Dict[str, List] = None) -> pd.DataFrame:
        """
        Model TDEE changes over time based on changing parameters
        
        Args:
            initial_data: Starting user data
            days: Number of days to model
            daily_changes: Dict mapping parameter names to daily values
            
        Returns:
            DataFrame with daily TDEE and component values
        """
        results = []
        
        # Create a copy of initial data to modify
        current_data = initial_data.copy()
        
        for day in range(days):
            # Update any parameters that change daily
            if daily_changes:
                for param, values in daily_changes.items():
                    if day < len(values):
                        current_data[param] = values[day]
            
            # Set the current day in deficit/surplus
            if 'caloric_deficit' in current_data:
                current_data['days_in_deficit'] = day
            
            # Calculate TDEE for this day
            tdee_data = self.calculate_tdee(current_data)
            
            # Add day number
            tdee_data['Day'] = day + 1
            
            results.append(tdee_data)
        
        return pd.DataFrame(results)

class BodyCompositionAnalyzer:
    """Analyze body composition and provide recommendations"""
    
    def __init__(self):
        # Define reference values and thresholds
        self.ffmi_reference = {
            'M': {'min': 18, 'untrained': 20, 'trained': 22, 'advanced': 23, 'elite': 25},
            'F': {'min': 15, 'untrained': 16.5, 'trained': 18, 'advanced': 19, 'elite': 21}
        }
        
        self.body_fat_reference = {
            'M': {'essential': 3, 'athletic': 10, 'fitness': 15, 'acceptable': 20, 'obese': 25},
            'F': {'essential': 12, 'athletic': 18, 'fitness': 23, 'acceptable': 30, 'obese': 35}
        }
    
    def calculate_indices(self, weight_kg: float, height_cm: float, 
                         body_fat_pct: float, sex: str) -> Dict[str, float]:
        """
        Calculate body composition indices
        
        Args:
            weight_kg: Weight in kg
            height_cm: Height in cm
            body_fat_pct: Body fat percentage (0-100)
            sex: 'M' or 'F'
            
        Returns:
            Dictionary with BMI, FFMI, and FMI
        """
        # Convert height to meters
        height_m = height_cm / 100
        
        # Calculate BMI
        bmi = weight_kg / (height_m ** 2)
        
        # Calculate fat mass and fat-free mass
        fat_mass_kg = weight_kg * (body_fat_pct / 100)
        lean_mass_kg = weight_kg - fat_mass_kg
        
        # Calculate FMI and FFMI
        fmi = fat_mass_kg / (height_m ** 2)
        ffmi = lean_mass_kg / (height_m ** 2)
        
        # Calculate normalized FFMI (for males, adjusts for height advantage)
        # This formula is commonly used to compare FFMI across different heights
        norm_ffmi = ffmi
        if sex.upper() == 'M':
            norm_ffmi = ffmi + (6.1 * (1.8 - height_m))
        
        return {
            'BMI': bmi,
            'FFMI': ffmi,
            'Normalized FFMI': norm_ffmi,
            'FMI': fmi,
            'Fat Mass (kg)': fat_mass_kg,
            'Lean Mass (kg)': lean_mass_kg
        }
    
    def calculate_hazard_ratio(self, fmi: float, ffmi: float) -> float:
        """
        Calculate mortality hazard ratio based on FMI and FFMI
        
        This is based on the graph provided in your image.
        
        Args:
            fmi: Fat Mass Index
            ffmi: Fat-Free Mass Index
            
        Returns:
            Estimated hazard ratio
        """
        # FMI hazard function (based on the graph)
        # U-shaped curve with minimum around 5-7
        if fmi < 5:
            # Left side of U (underweight)
            fmi_hazard = 1.0 + 0.08 * ((5 - fmi) ** 2)
        elif fmi <= 9:
            # Bottom of U (optimal)
            fmi_hazard = 0.9
        else:
            # Right side of U (increasing with fat)
            fmi_hazard = 0.9 + 0.09 * ((fmi - 9) ** 1.8)
        
        # FFMI hazard function (based on the graph)
        # J-shaped curve with minimum around 21-22
        if ffmi < 13:
            # Very low muscle mass - high hazard
            ffmi_hazard = 3.0 - 0.15 * ffmi
        elif ffmi < 18:
            # Decreasing hazard with more muscle
            ffmi_hazard = 2.0 - 0.05 * ffmi
        elif ffmi <= 22:
            # Optimal range
            ffmi_hazard = 0.8
        else:
            # Slight increase at very high FFMI (diminishing returns)
            ffmi_hazard = 0.8 + 0.01 * ((ffmi - 22) ** 1.5)
        
        # Combine hazards (multiplicative model)
        return fmi_hazard * ffmi_hazard
    
    def get_recommendation(self, indices: Dict[str, float], sex: str, 
                          current_activity: str, training_age: float) -> Dict[str, str]:
        """
        Generate personalized bulk/cut recommendation
        
        Args:
            indices: Dictionary with body composition indices
            sex: 'M' or 'F'
            current_activity: Activity level description
            training_age: Years of consistent training
            
        Returns:
            Dictionary with recommendations and analysis
        """
        bmi = indices['BMI']
        ffmi = indices['FFMI']
        fmi = indices['FMI']
        hazard_ratio = self.calculate_hazard_ratio(fmi, ffmi)
        
        # Get reference values for this sex
        ffmi_refs = self.ffmi_reference[sex.upper()]
        bf_refs = self.body_fat_reference[sex.upper()]
        
        # Analyze muscle development relative to training age
        expected_ffmi = ffmi_refs['min'] + min(training_age * 0.6, 
                                              ffmi_refs['elite'] - ffmi_refs['min'])
        muscle_development = ffmi / expected_ffmi
        
        # Determine body fat category
        body_fat_pct = indices['Fat Mass (kg)'] / indices['Fat Mass (kg)'] + indices['Lean Mass (kg)'] * 100
        
        # Determine body fat category
        if sex.upper() == 'M':
            if body_fat_pct < bf_refs['essential']:
                bf_category = "Below essential fat (health risk)"
            elif body_fat_pct < bf_refs['athletic']:
                bf_category = "Athletic range"
            elif body_fat_pct < bf_refs['fitness']:
                bf_category = "Fitness range"
            elif body_fat_pct < bf_refs['acceptable']:
                bf_category = "Acceptable range"
            else:
                bf_category = "Excess fat"
        else:  # Female
            if body_fat_pct < bf_refs['essential']:
                bf_category = "Below essential fat (health risk)"
            elif body_fat_pct < bf_refs['athletic']:
                bf_category = "Athletic range"
            elif body_fat_pct < bf_refs['fitness']:
                bf_category = "Fitness range"
            elif body_fat_pct < bf_refs['acceptable']:
                bf_category = "Acceptable range"
            else:
                bf_category = "Excess fat"
        
        # Determine FFMI category
        if ffmi < ffmi_refs['min']:
            ffmi_category = "Below average muscle development"
        elif ffmi < ffmi_refs['untrained']:
            ffmi_category = "Untrained level"
        elif ffmi < ffmi_refs['trained']:
            ffmi_category = "Trained level"
        elif ffmi < ffmi_refs['advanced']:
            ffmi_category = "Advanced level"
        elif ffmi < ffmi_refs['elite']:
            ffmi_category = "Elite level"
        else:
            ffmi_category = "Exceptional muscle development"
        
        # Make recommendation based on hazard ratio, body composition, and training status
        recommendation = {}
        
        # Primary recommendation (bulk, cut, or maintain)
        if hazard_ratio > 1.5:  # Health risk territory
            if fmi > 15:  # High fat is the primary risk
                recommendation['primary'] = "CUT: Reduce body fat to improve health markers"
                recommendation['deficit'] = "500-750 kcal deficit recommended"
                recommendation['priority'] = "Health improvement is top priority"
            elif ffmi < ffmi_refs['min']:  # Low muscle is the primary risk
                recommendation['primary'] = "LEAN BULK: Gradually build muscle with minimal fat gain"
                recommendation['surplus'] = "250-350 kcal surplus recommended"
                recommendation['priority'] = "Building muscle while maintaining health is priority"
            else:
                recommendation['primary'] = "RECOMP: Focus on body recomposition"
                recommendation['calories'] = "Maintenance calories with high protein"
                recommendation['priority'] = "Improving body composition at current weight"
        else:  # Not in immediate health risk
            # For those with low muscle development relative to training age
            if muscle_development < 0.85:
                if body_fat_pct > (bf_refs['fitness'] + 5):  # Higher body fat
                    recommendation['primary'] = "CUT FIRST, THEN BULK: Reduce body fat before building"
                    recommendation['deficit'] = "500 kcal deficit recommended"
                    recommendation['secondary'] = "After reaching fitness range body fat, transition to a lean bulk"
                else:  # Lower/moderate body fat
                    recommendation['primary'] = "BULK: Focus on muscle building"
                    recommendation['surplus'] = "350-500 kcal surplus recommended"
                    recommendation['secondary'] = "Your muscle development has room for improvement"
            # For those with good muscle development
            elif muscle_development >= 0.85:
                if body_fat_pct > bf_refs['fitness']:  # Higher body fat
                    recommendation['primary'] = "CUT: Reduce body fat to highlight muscle definition"
                    recommendation['deficit'] = "350-500 kcal deficit recommended"
                    recommendation['secondary'] = "You have good muscle mass, focus on definition"
                else:  # Lower/good body fat
                    recommendation['primary'] = "MAINTAIN or SLIGHT SURPLUS: Fine-tune your physique"
                    recommendation['calories'] = "Maintenance to 200 kcal surplus"
                    recommendation['secondary'] = "You have good overall development, focus on performance"
        
        # Additional context based on specific metrics
        if ffmi > ffmi_refs['advanced'] and body_fat_pct < bf_refs['athletic']:
            recommendation['physique'] = "Your body composition is in the advanced/athletic range"
        
        if hazard_ratio < 1.0:
            recommendation['health'] = "Your body composition is associated with favorable health markers"
        elif hazard_ratio > 2.0:
            recommendation['health'] = "Consider prioritizing health improvement through diet and exercise"
        
        # Add nuance for special cases
        if training_age < 1 and body_fat_pct < bf_refs['fitness']:
            recommendation['beginner'] = "As a beginner, you can build muscle and lose fat simultaneously with proper training and nutrition"
        
        if current_activity == "Sedentary" or current_activity == "Lightly Active":
            recommendation['activity'] = "Increasing daily activity would benefit your metabolism and health"
        
        return recommendation

def plot_hazard_ratio_heatmap():
    """Generate a heatmap visualization of mortality hazard ratio by FMI and FFMI"""
    # Create ranges for FMI and FFMI
    fmi_range = np.linspace(3, 25, 100)
    ffmi_range = np.linspace(13, 26, 100)
    
    # Create meshgrid
    fmi_grid, ffmi_grid = np.meshgrid(fmi_range, ffmi_range)
    
    # Initialize hazard ratio array
    hazard_grid = np.zeros_like(fmi_grid)
    
    # Instantiate analyzer
    analyzer = BodyCompositionAnalyzer()
    
    # Calculate hazard ratios
    for i in range(len(ffmi_range)):
        for j in range(len(fmi_range)):
            hazard_grid[i, j] = analyzer.calculate_hazard_ratio(fmi_grid[i, j], ffmi_grid[i, j])
    
    # Create heatmap with plotly
    fig = go.Figure(data=go.Contour(
        z=hazard_grid,
        x=fmi_range,
        y=ffmi_range,
        colorscale='RdBu_r',  # Red is high hazard, blue is low
        contours=dict(
            start=0.7,
            end=3,
            size=0.1,
            showlabels=True
        ),
        colorbar=dict(
            title='Hazard Ratio',
            titleside='right',
        )
    ))
    
    # Add optimal zone marker (example coordinates based on the model)
    fig.add_trace(go.Scatter(
        x=[7],
        y=[21],
        mode='markers',
        marker=dict(
            size=10,
            color='green',
            symbol='star'
        ),
        name='Optimal Zone'
    ))
    
    # Update layout
    fig.update_layout(
        title='Mortality Hazard Ratio by Body Composition',
        xaxis_title='Fat Mass Index (FMI)',
        yaxis_title='Fat-Free Mass Index (FFMI)',
        width=800,
        height=600
    )
    
    return fig

def plot_tdee_components(tdee_data: Dict[str, float]):
    """Create a pie chart of TDEE components"""
    labels = []
    values = []
    
    # Add all components except total
    for component, value in tdee_data.items():
        if component != 'Total TDEE' and value > 0:
            labels.append(component)
            values.append(value)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    
    fig.update_layout(
        title_text=f"TDEE Components: Total {tdee_data['Total TDEE']:.0f} calories",
        annotations=[dict(text='TDEE', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

def plot_tdee_over_time(tdee_df: pd.DataFrame):
    """Create a stacked area chart of TDEE components over time"""
    # Components to include
    components = ['BMR', 'TEF', 'NEAT', 'EAT', 'Thermogenesis', 'Immune', 'Reproductive']
    
    # Create stacked area chart
    fig = go.Figure()
    
    # Add each component as an area
    for component in components:
        if component in tdee_df.columns:
            fig.add_trace(go.Scatter(
                x=tdee_df['Day'],
                y=tdee_df[component],
                mode='lines',
                stackgroup='one',
                name=component
            ))
    
    # Update layout
    fig.update_layout(
        title='TDEE Components Over Time',
        xaxis_title='Day',
        yaxis_title='Calories',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# ----------------- DIFFERENTIAL EQUATION MODEL -----------------

class TDEEDifferentialModel:
    """Model TDEE changes over time using differential equations"""
    
    def __init__(self):
        self.calculator = TDEECalculator()
    
    def energy_balance_equations(self, t, y, user_data):
        """
        Differential equations for energy balance model
        
        Args:
            t: Time point (days)
            y: State vector [weight_kg, fat_mass_kg, glycogen_kg, adaptive_factor]
            user_data: Dictionary with user parameters
            
        Returns:
            Derivatives of state variables
        """
        weight_kg, fat_mass_kg, glycogen_kg, adaptive_factor = y
        
        # Calculate lean mass
        lean_mass_kg = weight_kg - fat_mass_kg - glycogen_kg
        
        # Update user data with current state
        current_data = user_data.copy()
        current_data['weight_kg'] = weight_kg
        current_data['lean_mass_kg'] = lean_mass_kg
        current_data['days_in_deficit'] = t
        
        # Calculate TDEE with current values
        tdee = self.calculator.calculate_tdee(current_data)
        current_tdee = tdee['Total TDEE'] * adaptive_factor
        
        # Energy surplus or deficit (kcal/day)
        energy_balance = current_data.get('caloric_intake', 2000) - current_tdee
        
        # Convert energy balance to weight changes
        # Energy partition depends on surplus vs deficit and current body composition
        if energy_balance > 0:  # Surplus
            # In surplus, some goes to glycogen, some to muscle, some to fat
            # Higher body fat % = more fat storage, lower = more muscle
            body_fat_pct = (fat_mass_kg / weight_kg) * 100
            
            # P-ratio: proportion of surplus going to lean tissue (varies by individual)
            if 'p_ratio' in current_data:
                p_ratio = current_data['p_ratio']
            else:
                # Estimate P-ratio based on body fat percentage
                # Leaner individuals tend to partition more energy to muscle
                p_ratio = max(0.15, 0.5 - (body_fat_pct / 100))
            
            # Energy partition
            glycogen_energy = min(energy_balance * 0.05, 400)  # Max ~400 kcal to glycogen
            lean_energy = (energy_balance - glycogen_energy) * p_ratio
            fat_energy = energy_balance - glycogen_energy - lean_energy
            
            # Convert energy to mass changes
            # Glycogen stores 4 kcal/g but binds with water (1:3 ratio)
            glycogen_change = glycogen_energy / 1000  # 1g glycogen = ~4 kcal, but with water ~1 kcal/g
            
            # Lean mass has protein (~4 kcal/g) and water
            lean_change = lean_energy / 1600  # Approximately 1600 kcal per kg of lean mass
            
            # Fat is energy dense at ~9.3 kcal/g
            fat_change = fat_energy / 9300  # kcal to kg of fat
        
        else:  # Deficit
            # In deficit, glycogen goes first, then mix of fat and lean
            energy_deficit = -energy_balance
            
            # Glycogen loss depends on deficit size and available glycogen
            max_glycogen_use = min(glycogen_kg * 1000, energy_deficit * 0.1)
            glycogen_energy = max_glycogen_use
            
            # Remaining deficit
            remaining_deficit = energy_deficit - glycogen_energy
            
            # Lean tissue loss depends on body fat and protein intake
            # Leaner individuals lose more muscle in a deficit
            body_fat_pct = (fat_mass_kg / weight_kg) * 100
            protein_intake = current_data.get('protein_intake_g', 100)
            protein_factor = min(max(protein_intake / (weight_kg * 2.2), 0.5), 1.0)  # Higher protein = more muscle sparing
            
            # Lean tissue loss factor (lower = better muscle preservation)
            lean_loss_factor = max(0.05, 0.25 - (body_fat_pct / 100) - (protein_factor * 0.1))
            lean_loss_factor = min(lean_loss_factor, 0.25)  # Cap at 25% max
            
            # Energy from lean and fat
            lean_energy = remaining_deficit * lean_loss_factor
            fat_energy = remaining_deficit - lean_energy
            
            # Convert to mass changes (negative for loss)
            glycogen_change = -glycogen_energy / 1000
            lean_change = -lean_energy / 1600
            fat_change = -fat_energy / 9300
        
        # Calculate weight change
        weight_change = glycogen_change + lean_change + fat_change
        
        # Update adaptive factor based on energy balance
        # Adaptation happens faster in deficit than surplus
        if energy_balance < 0:
            adaptive_change = -0.005 * (energy_balance / -1000)  # More negative = faster adaptation
        else:
            adaptive_change = 0.003 * (energy_balance / 1000)  # Positive adaptation is slower
        
        # Limit adaptation range
        adaptive_factor_new = adaptive_factor + adaptive_change
        adaptive_factor_new = max(0.8, min(adaptive_factor_new, 1.2))  # Limit to 80-120% range
        adaptive_change = adaptive_factor_new - adaptive_factor
        
        return [weight_change, fat_change, glycogen_change, adaptive_change]
    
    def simulate(self, user_data: dict, days: int = 30) -> pd.DataFrame:
        """
        Simulate body composition and energy expenditure changes over time
        
        Args:
            user_data: Dictionary with user parameters
            days: Number of days to simulate
            
        Returns:
            DataFrame with simulation results
        """
        # Initial conditions
        weight_kg = user_data.get('weight_kg', 70)
        body_fat_pct = user_data.get('body_fat_pct', 20)
        fat_mass_kg = weight_kg * (body_fat_pct / 100)
        glycogen_kg = weight_kg * 0.01  # ~1% of body weight is glycogen+water
        lean_mass_kg = weight_kg - fat_mass_kg - glycogen_kg
        adaptive_factor = 1.0  # Start with no adaptation
        
        # Initial state vector
        y0 = [weight_kg, fat_mass_kg, glycogen_kg, adaptive_factor]
        
        # Time points
        t_span = (0, days)
        t_eval = np.arange(0, days + 1, 1)
        
        # Solve differential equations
        solution = solve_ivp(
            fun=lambda t, y: self.energy_balance_equations(t, y, user_data),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45'
        )
        
        # Create dataframe from results
        results = pd.DataFrame({
            'Day': solution.t,
            'Weight (kg)': solution.y[0],
            'Fat Mass (kg)': solution.y[1],
            'Glycogen (kg)': solution.y[2],
            'Adaptive Factor': solution.y[3]
        })
        
        # Calculate additional metrics
        results['Lean Mass (kg)'] = results['Weight (kg)'] - results['Fat Mass (kg)'] - results['Glycogen (kg)']
        results['Body Fat %'] = (results['Fat Mass (kg)'] / results['Weight (kg)']) * 100
        
        # Calculate TDEE for each day
        tdee_values = []
        for i, row in results.iterrows():
            day_data = user_data.copy()
            day_data['weight_kg'] = row['Weight (kg)']
            day_data['lean_mass_kg'] = row['Lean Mass (kg)']
            day_data['days_in_deficit'] = row['Day']
            
            tdee = self.calculator.calculate_tdee(day_data)
            adjusted_tdee = {k: v * row['Adaptive Factor'] if k == 'Total TDEE' else v 
                            for k, v in tdee.items()}
            
            tdee_values.append(adjusted_tdee)
        
        # Add TDEE components to results
        tdee_df = pd.DataFrame(tdee_values)
        results = pd.concat([results, tdee_df], axis=1)
        
        return results

# ----------------- STREAMLIT UI -----------------

def create_sidebar_inputs():
    """Create inputs in sidebar and return user data dictionary"""
    st.sidebar.markdown("<div class='section-header'>Personal Information</div>", unsafe_allow_html=True)
    
    # Basic measurements
    sex = st.sidebar.radio("Sex", ["Male", "Female"], horizontal=True)
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    height = st.sidebar.number_input("Height (cm)", min_value=120, max_value=220, value=170, step=1)
    weight = st.sidebar.number_input("Weight (kg)", min_value=40, max_value=200, value=70, step=1)
    
    # Body composition
    st.sidebar.markdown("<div class='subsection-header'>Body Composition</div>", unsafe_allow_html=True)
    body_fat = st.sidebar.slider("Body Fat Percentage (%)", min_value=5, max_value=50, value=20, step=1)
    has_dexa = st.sidebar.checkbox("I have DEXA/accurate measurements")
    
    if has_dexa:
        lean_mass = st.sidebar.number_input("Lean Body Mass (kg)", min_value=30, max_value=120, 
                                          value=round(weight * (1 - body_fat/100)), step=1)
    else:
        lean_mass = weight * (1 - body_fat/100)
    
    # Activity and diet
    st.sidebar.markdown("<div class='subsection-header'>Activity & Diet</div>", unsafe_allow_html=True)
    
    activity_level = st.sidebar.selectbox(
        "Activity Level (excluding exercise)",
        ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
        index=1
    )
    
    training_age = st.sidebar.slider("Years of consistent training", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    
    # Exercise inputs
    exercise_light = st.sidebar.number_input("Light Exercise (min/day)", min_value=0, max_value=240, value=15, step=5)
    exercise_moderate = st.sidebar.number_input("Moderate Exercise (min/day)", min_value=0, max_value=240, value=30, step=5)
    exercise_vigorous = st.sidebar.number_input("Vigorous Exercise (min/day)", min_value=0, max_value=120, value=0, step=5)
    
    # Diet inputs
    diet_tab, tdee_tab = st.sidebar.tabs(["Diet", "Advanced TDEE Factors"])
    
    with diet_tab:
        caloric_intake = st.number_input("Daily Caloric Intake", min_value=500, max_value=5000, value=2000, step=100)
        protein_pct = st.slider("Protein (%)", min_value=10, max_value=50, value=30, step=5) / 100
        carb_pct = st.slider("Carbohydrates (%)", min_value=10, max_value=80, value=40, step=5) / 100
        fat_pct = st.slider("Fat (%)", min_value=10, max_value=80, value=30, step=5) / 100
        
        # Normalize macros if they don't add up to 100%
        total_pct = protein_pct + carb_pct + fat_pct
        if total_pct != 1.0:
            protein_pct /= total_pct
            carb_pct /= total_pct
            fat_pct /= total_pct
        
        protein_g = round((caloric_intake * protein_pct) / 4)  # 4 calories per gram of protein
        st.caption(f"Protein intake: {protein_g}g (approximately {round(protein_g / weight, 1)}g/kg)")
    
    with tdee_tab:
        # Additional TDEE factors
        st.caption("These factors can influence your TDEE:")
        
        cold_exposure = st.checkbox("Cold Exposure")
        if cold_exposure:
            temperature = st.slider("Environment Temperature (°C)", min_value=5, max_value=35, value=19, step=1)
            cold_hours = st.slider("Hours per day at this temperature", min_value=1, max_value=24, value=8, step=1)
            cold_acclimated = st.checkbox("I'm acclimated to cold (regular exposure)")
        else:
            temperature = 22  # Room temperature
            cold_hours = 24
            cold_acclimated = False
        
        has_fever = st.checkbox("Currently Sick/Feverish")
        if has_fever:
            fever_temp = st.slider("Degrees above normal (°C)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
            immune_activation = st.slider("Immune System Activation", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        else:
            fever_temp = 0
            immune_activation = 0
        
        if sex == "Female":
            menstrual_phase = st.selectbox(
                "Menstrual Cycle Phase",
                [None, "Follicular", "Ovulation", "Luteal", "Menstrual"],
                index=0
            )
        else:
            menstrual_phase = None
        
        sexual_activity = st.slider("Sexual Activity (minutes/day)", min_value=0, max_value=60, value=0, step=5)
    
    # Calculate deficit/surplus based on a quick TDEE estimation
    with st.sidebar.expander("Caloric Balance"):
        if sex == "Male":
            quick_bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            quick_bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        activity_multipliers = {
            "Sedentary": 1.2,
            "Lightly Active": 1.375,
            "Moderately Active": 1.55,
            "Very Active": 1.725,
            "Extremely Active": 1.9
        }
        
        # Quick TDEE estimate
        quick_tdee = quick_bmr * activity_multipliers[activity_level]
        # Add exercise calories (rough estimate)
        exercise_cals = (exercise_light * 0.05 + exercise_moderate * 0.1 + exercise_vigorous * 0.15) * weight
        quick_tdee += exercise_cals
        
        st.caption(f"Estimated TDEE: ~{round(quick_tdee)} calories")
        st.caption(f"Current balance: {round(caloric_intake - quick_tdee)} calories")
        
        caloric_deficit = caloric_intake - quick_tdee
        days_in_deficit = st.number_input("Days in current caloric state", min_value=0, max_value=365, value=0, step=1)
    
    # Compile all inputs into a user data dictionary
    user_data = {
        'sex': sex[0],  # Just use the first letter M/F
        'age': age,
        'height_cm': height,
        'weight_kg': weight,
        'body_fat_pct': body_fat,
        'lean_mass_kg': lean_mass,
        'activity_level': activity_level,
        'training_age': training_age,
        'exercise_minutes': {
            'Light': exercise_light,
            'Moderate': exercise_moderate,
            'Vigorous': exercise_vigorous,
            'Very Vigorous': 0
        },
        'caloric_intake': caloric_intake,
        'protein_pct': protein_pct,
        'carb_pct': carb_pct,
        'fat_pct': fat_pct,
        'protein_intake_g': protein_g,
        'temperature': temperature,
        'cold_exposure_hours': cold_hours,
        'cold_acclimated': cold_acclimated,
        'fever_celsius': fever_temp,
        'immune_activation': immune_activation,
        'menstrual_phase': menstrual_phase,
        'sexual_activity_minutes': sexual_activity,
        'caloric_deficit': caloric_deficit,
        'days_in_deficit': days_in_deficit
    }
    
    return user_data

def main():
    # Create sidebar inputs
    user_data = create_sidebar_inputs()
    
    # Create tabs for different app sections
    tab1, tab2, tab3 = st.tabs(["Current Analysis", "TDEE Modeling", "Projection & Simulation"])
    
    # Instantiate calculators
    tdee_calc = TDEECalculator()
    body_comp = BodyCompositionAnalyzer()
    diff_model = TDEEDifferentialModel()
    
    # Tab 1: Current Analysis
    with tab1:
        st.markdown("<div class='section-header'>Current TDEE Breakdown</div>", unsafe_allow_html=True)
        
        # Calculate TDEE and body composition
        tdee_result = tdee_calc.calculate_tdee(user_data)
        indices = body_comp.calculate_indices(
            user_data['weight_kg'], 
            user_data['height_cm'], 
            user_data['body_fat_pct'], 
            user_data['sex']
        )
        
        # Display TDEE
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # TDEE pie chart
            fig = plot_tdee_components(tdee_result)
            st.plotly_chart(fig)
            
        with col2:
            # TDEE breakdown in a table
            st.markdown("<div class='subsection-header'>TDEE Components</div>", unsafe_allow_html=True)
            tdee_df = pd.DataFrame({
                'Component': list(tdee_result.keys())[:-1],  # Exclude total
                'Calories': [round(val) for val in list(tdee_result.values())[:-1]],
                'Percentage': [round(val / tdee_result['Total TDEE'] * 100, 1) for val in list(tdee_result.values())[:-1]]
            })
            st.dataframe(tdee_df, hide_index=True)
            
            st.metric("Total Daily Energy Expenditure", f"{round(tdee_result['Total TDEE'])} calories")
            
            balance = user_data['caloric_intake'] - tdee_result['Total TDEE']
            st.metric("Caloric Balance", f"{round(balance)} calories", 
                     delta=None if abs(balance) < 50 else ("deficit" if balance < 0 else "surplus"))
        
        # Body composition analysis
        st.markdown("<div class='section-header'>Body Composition Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display body composition indices
            st.markdown("<div class='subsection-header'>Current Metrics</div>", unsafe_allow_html=True)
            
            metrics_df = pd.DataFrame({
                'Metric': ['BMI', 'Fat Mass Index (FMI)', 'Fat-Free Mass Index (FFMI)', 'Body Fat Percentage'],
                'Value': [
                    f"{indices['BMI']:.1f} kg/m²",
                    f"{indices['FMI']:.1f} kg/m²",
                    f"{indices['FFMI']:.1f} kg/m²",
                    f"{user_data['body_fat_pct']:.1f}%"
                ]
            })
            st.dataframe(metrics_df, hide_index=True)
            
            # Calculate and display hazard ratio
            hazard = body_comp.calculate_hazard_ratio(indices['FMI'], indices['FFMI'])
            
            risk_color = "green"
            if hazard >= 2.0:
                risk_level = "High"
                risk_color = "red"
            elif hazard >= 1.3:
                risk_level = "Moderate"
                risk_color = "orange"
            elif hazard >= 1.0:
                risk_level = "Slight"
                risk_color = "yellow"
            else:
                risk_level = "Low"
                risk_color = "green"
            
            st.markdown(f"<div style='font-size:1.2rem'>Mortality Hazard Ratio: <span style='color:{risk_color};font-weight:bold'>{hazard:.2f}</span> ({risk_level} risk)</div>", unsafe_allow_html=True)
            
            # Get recommendations
            recommendations = body_comp.get_recommendation(
                indices, 
                user_data['sex'], 
                user_data['activity_level'], 
                user_data['training_age']
            )
            
            # Display primary recommendation in a box
            st.markdown("<div class='section-header'>Recommendations</div>", unsafe_allow_html=True)
            
            if 'primary' in recommendations:
                st.markdown(f"""
                <div class='info-box' style='background-color:#f0f8ff;border-left:5px solid #1e90ff;padding:10px;'>
                <div style='font-size:1.2rem;font-weight:bold;'>{recommendations['primary']}</div>
                {recommendations.get('surplus', recommendations.get('deficit', recommendations.get('calories', '')))}
                </div>
                """, unsafe_allow_html=True)
            
            # Display additional recommendations and context
            for key, value in recommendations.items():
                if key not in ['primary', 'surplus', 'deficit', 'calories']:
                    st.markdown(f"**{key.capitalize()}**: {value}")
            
        with col2:
            # Show body composition visualization
            st.markdown("<div class='subsection-header'>Body Composition Visualization</div>", unsafe_allow_html=True)
            
            # Generate a hazard ratio heatmap
            hazard_map = plot_hazard_ratio_heatmap()
            
            # Add the user's position to the heatmap
            hazard_map.add_trace(go.Scatter(
                x=[indices['FMI']],
                y=[indices['FFMI']],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='circle',
                    line=dict(
                        color='black',
                        width=2
                    )
                ),
                name='Your Position'
            ))
            
            st.plotly_chart(hazard_map)
            
            st.caption("""
            The chart shows the relationship between Fat Mass Index (FMI) and Fat-Free Mass Index (FFMI) 
            and their impact on health risk. Lower hazard ratios (blue regions) indicate better health outcomes. 
            The star represents an optimal body composition zone.
            """)
    
    # Tab 2: TDEE Modeling
    with tab2:
        st.markdown("<div class='section-header'>TDEE Component Analysis</div>", unsafe_allow_html=True)
        
        st.markdown("""
        This section analyzes how each component of TDEE responds to changes. You can adjust various factors
        to see how they affect your energy expenditure.
        """)
        
        # Create tabs for different TDEE components
        component_tabs = st.tabs([
            "BMR & Adaptive Metabolism", 
            "Thermic Effect of Food (TEF)", 
            "Activity (NEAT & EAT)",
            "Environmental & Physiological"
        ])
        
        with component_tabs[0]:
            st.markdown("<div class='subsection-header'>Basal Metabolic Rate (BMR) Analysis</div>", unsafe_allow_html=True)
            
            st.markdown("""
            BMR is the energy used to maintain basic physiological functions at rest. It adapts to caloric 
            surpluses or deficits, but more slowly than other components.
            """)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Simulate BMR adaptation to caloric deficit
                st.markdown("##### BMR Adaptation to Caloric Deficit")
                
                deficit_days = st.slider("Days in deficit", min_value=0, max_value=30, value=14, step=1, key="deficit_days")
                deficit_pct = st.slider("Deficit (% of maintenance)", min_value=0, max_value=50, value=25, step=5, key="deficit_pct")
                
                # Calculate adaptation
                maintenance = tdee_result['Total TDEE']
                deficit_kcal = maintenance * (deficit_pct / 100)
                
                # Create a data frame for BMR adaptation
                bmr_adaptation = []
                base_bmr = tdee_result['BMR']
                
                for day in range(deficit_days + 1):
                    # Simplified model of BMR adaptation
                    # More aggressive deficits cause faster and larger adaptations
                    max_reduction = 0.15  # Maximum 15% reduction
                    adaptation_rate = deficit_pct / 100  # Higher deficit = faster adaptation
                    
                    # Exponential approach to maximum adaptation
                    adaptation_pct = max_reduction * (1 - np.exp(-adaptation_rate * day / 14))
                    adapted_bmr = base_bmr * (1 - adaptation_pct)
                    
                    bmr_adaptation.append({
                        'Day': day,
                        'BMR': adapted_bmr,
                        'Adaptation (%)': adaptation_pct * 100
                    })
                
                bmr_df = pd.DataFrame(bmr_adaptation)
                
                # Plot
                fig = px.line(bmr_df, x='Day', y=['BMR', 'Adaptation (%)'], 
                             title=f"{deficit_pct}% Deficit: BMR Adaptation Over Time",
                             labels={'value': 'Value', 'variable': 'Metric'})
                
                fig.update_layout(yaxis=dict(title='BMR (kcal)'), 
                                 yaxis2=dict(title='Adaptation (%)', overlaying='y', side='right'))
                
                st.plotly_chart(fig)
                
            with col2:
                # Show how BMR varies with body composition
                st.markdown("##### BMR by Body Composition")
                
                # Create a range of body fat percentages
                bf_range = list(range(5, 41, 5))
                
                # Calculate BMR for each body fat percentage
                bmr_by_bf = []
                weight = user_data['weight_kg']
                
                for bf in bf_range:
                    lean_mass = weight * (1 - bf/100)
                    # Katch-McArdle Formula
                    bmr = 370 + (21.6 * lean_mass)
                    bmr_by_bf.append({
                        'Body Fat %': bf,
                        'BMR': bmr,
                        'Lean Mass (kg)': lean_mass
                    })
                
                bf_df = pd.DataFrame(bmr_by_bf)
                
                # Plot
                fig = px.bar(bf_df, x='Body Fat %', y='BMR', 
                            hover_data=['Lean Mass (kg)'],
                            title="BMR vs Body Fat % (Same Total Weight)",
                            labels={'BMR': 'BMR (kcal)'})
                
                # Add marker for current body fat
                fig.add_vline(x=user_data['body_fat_pct'], line_dash="dash", line_color="red")
                fig.add_annotation(x=user_data['body_fat_pct'], y=max(bf_df['BMR']),
                                  text="Your current BF%", showarrow=True, arrowhead=1)
                
                st.plotly_chart(fig)
                
                st.markdown("""
                **Key Insights:**
                - Higher lean mass = higher BMR (more metabolically active tissue)
                - BMR adaptation occurs within days but reaches maximum after weeks
                - Larger deficits cause more significant BMR reductions
                - Diet breaks can help restore adapted BMR
                """)

        with component_tabs[1]:
            st.markdown("<div class='subsection-header'>Thermic Effect of Food (TEF) Analysis</div>", unsafe_allow_html=True)
            
            st.markdown("""
            TEF is the energy expended in digesting, absorbing, and processing nutrients. It varies significantly
            by macronutrient, with protein having the highest thermic effect.
            """)
            
            # Create sliders for TEF analysis
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("##### TEF by Macronutrient Distribution")
                
                # User inputs for hypothetical meals
                total_calories = st.slider("Total Daily Calories", min_value=1500, max_value=3500, value=2000, step=100)
                
                # Create different macronutrient distributions
                macro_distributions = {
                    "High Protein": [0.40, 0.40, 0.20],  # P/C/F
                    "High Carb": [0.15, 0.65, 0.20],
                    "High Fat": [0.15, 0.25, 0.60],
                    "Balanced": [0.25, 0.45, 0.30],
                    "Keto": [0.30, 0.10, 0.60],
                    "Current Diet": [user_data['protein_pct'], user_data['carb_pct'], user_data['fat_pct']]
                }
                
                # Calculate TEF for each distribution
                tef_results = []
                
                for diet_name, macros in macro_distributions.items():
                    protein_pct, carb_pct, fat_pct = macros
                    
                    # Calculate TEF
                    protein_cals = total_calories * protein_pct
                    carb_cals = total_calories * carb_pct
                    fat_cals = total_calories * fat_pct
                    
                    protein_tef = protein_cals * 0.25  # 25% TEF for protein
                    carb_tef = carb_cals * 0.075      # 7.5% TEF for carbs
                    fat_tef = fat_cals * 0.015        # 1.5% TEF for fat
                    
                    total_tef = protein_tef + carb_tef + fat_tef
                    
                    # Calculate protein in grams
                    protein_g = protein_cals / 4
                    
                    tef_results.append({
                        'Diet Type': diet_name,
                        'Protein (%)': round(protein_pct * 100),
                        'Carbs (%)': round(carb_pct * 100),
                        'Fat (%)': round(fat_pct * 100),
                        'Protein (g)': round(protein_g),
                        'Total TEF': round(total_tef),
                        'TEF (% of intake)': round(total_tef / total_calories * 100, 1)
                    })
                
                tef_df = pd.DataFrame(tef_results)
                
                # Create visualization
                fig = px.bar(tef_df, x='Diet Type', y='Total TEF', 
                            text='TEF (% of intake)',
                            hover_data=['Protein (%)', 'Carbs (%)', 'Fat (%)', 'Protein (g)'],
                            title=f"TEF by Diet Type ({total_calories} calories)")
                
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                
                st.plotly_chart(fig)
            
            with col2:
                st.markdown("##### TEF Timeline Analysis")
                
                # Simulate TEF over a day with different meal patterns
                meal_pattern = st.selectbox(
                    "Meal Pattern",
                    ["3 meals", "Intermittent Fasting (16:8)", "6 small meals", "OMAD (One Meal a Day)"]
                )
                
                # Define meal timings and sizes based on pattern
                meal_timings = {
                    "3 meals": [(7, 0.25), (12, 0.4), (18, 0.35)],  # (hour, portion)
                    "Intermittent Fasting (16:8)": [(12, 0.4), (16, 0.2), (20, 0.4)],
                    "6 small meals": [(7, 0.15), (10, 0.15), (13, 0.2), (16, 0.15), (19, 0.2), (21, 0.15)],
                    "OMAD (One Meal a Day)": [(18, 1.0)]
                }
                
                # TEF function over time
                def tef_curve(hours_since_meal, meal_size, protein_pct=0.3):
                    # TEF peaks around 1-2 hours post-meal and tapers off over 4-6 hours
                    if hours_since_meal <= 0:
                        return 0
                    
                    # Scale by meal size and protein content
                    tef_factor = meal_size * (0.1 + 0.15 * protein_pct)  # 10-25% depending on protein
                    
                    # TEF curve (peaks at 1-2 hours, mostly gone by 5-6 hours)
                    if hours_since_meal < 2:
                        # Rising phase
                        return tef_factor * (hours_since_meal / 2)
                    else:
                        # Declining phase
                        return tef_factor * max(0, 1 - (hours_since_meal - 2) / 4)
                
                # Generate data for a full day
                hours = np.arange(0, 24.25, 0.25)
                tef_data = []
                
                for hour in hours:
                    # Calculate TEF from all previous meals
                    total_tef = 0
                    
                    for meal_time, meal_size in meal_timings[meal_pattern]:
                        hours_since_meal = hour - meal_time
                        if hours_since_meal >= 0:
                            total_tef += tef_curve(hours_since_meal, meal_size, user_data['protein_pct'])
                    
                    # Scale to appropriate range (% of daily TEF)
                    scaled_tef = total_tef * tdee_result['TEF'] * 4  # Factor to make area under curve = TEF
                    
                    tef_data.append({
                        'Hour': hour,
                        'TEF Rate (kcal/hr)': scaled_tef
                    })
                
                tef_timeline_df = pd.DataFrame(tef_data)
                
                # Plot
                fig = px.line(tef_timeline_df, x='Hour', y='TEF Rate (kcal/hr)',
                             title=f"TEF Throughout the Day: {meal_pattern}")
                
                # Add meal markers
                for meal_time, meal_size in meal_timings[meal_pattern]:
                    fig.add_vline(x=meal_time, line_dash="dash", line_color="green")
                    fig.add_annotation(x=meal_time, y=max(tef_timeline_df['TEF Rate (kcal/hr)']),
                                      text=f"Meal: {int(meal_size*100)}%", showarrow=True, arrowhead=1)
                
                st.plotly_chart(fig)
                
                st.markdown("""
                **Key Insights:**
                - Protein has 5-10× higher TEF than fat
                - TEF typically accounts for 8-15% of TDEE
                - A high-protein diet can burn 100-200 extra calories daily
                - TEF effect begins immediately after eating and lasts 4-6 hours
                - Meal timing affects the TEF curve but not total daily TEF
                """)
                
        with component_tabs[2]:
            st.markdown("<div class='subsection-header'>Activity Thermogenesis (NEAT & EAT)</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("##### Exercise Activity (EAT)")
                
                st.markdown("""
                Exercise Activity Thermogenesis (EAT) is energy expended during planned physical exercise.
                It varies greatly by intensity and duration, and includes afterburn (EPOC) effects.
                """)
                
                # Create sliders for exercise analysis
                exercise_type = st.selectbox(
                    "Exercise Type",
                    ["Walking", "Jogging", "Running", "Cycling", "Swimming", "Weight Training", "HIIT"]
                )
                
                # MET values for different exercises
                exercise_mets = {
                    "Walking": 3.5,
                    "Jogging": 7.0,
                    "Running": 10.0,
                    "Cycling": 8.0,
                    "Swimming": 7.0,
                    "Weight Training": 5.0,
                    "HIIT": 12.0
                }
                
                exercise_duration = st.slider("Duration (minutes)", min_value=10, max_value=120, value=30, step=5)
                
                # Calculate exercise expenditure
                weight = user_data['weight_kg']
                met_value = exercise_mets[exercise_type]
                
                # Formula: MET × weight (kg) × duration (hours)
                hours = exercise_duration / 60
                exercise_calories = met_value * weight * hours
                
                # Calculate EPOC (afterburn)
                if exercise_type in ["HIIT", "Weight Training"]:
                    epoc_pct = 0.15  # 15% for high intensity
                elif exercise_type in ["Running", "Swimming"]:
                    epoc_pct = 0.10  # 10% for vigorous
                else:
                    epoc_pct = 0.05  # 5% for moderate
                
                epoc_calories = exercise_calories * epoc_pct
                
                # Display results
                col1a, col1b = st.columns(2)
                col1a.metric("During Exercise", f"{int(exercise_calories)} kcal")
                col1b.metric("Afterburn (EPOC)", f"{int(epoc_calories)} kcal")
                
                # Calculate exercise intensity relative to user's capacity
                intensity_pct = min(100, (met_value / 12) * 100)
                
                # Show gauges for intensity
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = intensity_pct,
                    title = {'text': "Relative Intensity"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if intensity_pct > 75 else "orange" if intensity_pct > 50 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "lightorange"},
                            {'range': [75, 100], 'color': "salmon"}
                        ]
                    }
                ))
                
                fig.update_layout(height=250)
                
                st.plotly_chart(fig)
                
                st.markdown(f"""
                **Exercise Insights:**
                - {exercise_type} at this intensity burns approximately **{int(exercise_calories + epoc_calories)} total calories**
                - This is equivalent to **{int((exercise_calories + epoc_calories) / 9)} grams of fat** (energy equivalent)
                - A similar daily exercise habit would increase TDEE by about **{int((exercise_calories + epoc_calories) * 7 / 7)} calories**
                """)
                
            with col2:
                st.markdown("##### Non-Exercise Activity (NEAT)")
                
                st.markdown("""
                NEAT is highly variable between individuals and can be a significant factor in weight management.
                It includes all daily movement outside of planned exercise.
                """)
                
                # Create NEAT comparison of different lifestyles
                lifestyles = [
                    "Sedentary (desk job, little activity)",
                    "Lightly Active (standing job, some walking)",
                    "Moderately Active (job with physical demands)",
                    "Very Active (physical job with additional activity)",
                    "Extremely Active (heavy manual labor)"
                ]
                
                # Corresponding activity multipliers
                multipliers = [1.2, 1.375, 1.55, 1.725, 1.9]
                
                # Calculate NEAT based on BMR
                bmr = tdee_result['BMR']
                neat_values = [(mult - 1) * bmr for mult in multipliers]
                
                # Create a dataframe for NEAT comparison
                neat_comparison = pd.DataFrame({
                    'Lifestyle': lifestyles,
                    'NEAT (kcal)': [round(neat) for neat in neat_values],
                    'Activity Multiplier': multipliers
                })
                
                # Find current lifestyle
                current_lifestyle_index = lifestyles.index(user_data['activity_level']) if user_data['activity_level'] in lifestyles else 0
                
                # Plot
                fig = px.bar(neat_comparison, x='Lifestyle', y='NEAT (kcal)',
                            title="NEAT by Lifestyle",
                            text='NEAT (kcal)')
                
                # Highlight current lifestyle
                fig.update_traces(marker_color=['lightblue' if i != current_lifestyle_index else 'darkblue' 
                                              for i in range(len(lifestyles))],
                                 texttemplate='%{text}', textposition='outside')
                
                st.plotly_chart(fig)
                
                # Show NEAT adaptation to energy balance
                st.markdown("##### NEAT Adaptation to Energy Balance")
                
                # Create data for NEAT adaptation to energy balance
                energy_balance = np.linspace(-1000, 1000, 21)  # -1000 to +1000 kcal
                
                # Individual responsiveness to energy balance
                neat_responsiveness = st.select_slider(
                    "NEAT Response Profile",
                    options=["Low Responder", "Average Responder", "High Responder"],
                    value="Average Responder"
                )
                
                # Set responsiveness factor
                if neat_responsiveness == "Low Responder":
                    response_factor = 0.1
                elif neat_responsiveness == "Average Responder":
                    response_factor = 0.2
                else:  # High Responder
                    response_factor = 0.35
                
                # Calculate NEAT adaptation
                current_neat = tdee_result['NEAT']
                neat_adaptation = []
                
                for balance in energy_balance:
                    if balance < 0:  # Deficit
                        # Decrease NEAT in deficit (more for low responders)
                        factor = response_factor * 0.8  # Dampened response in deficit
                        adaptation = max(-0.3, balance / 1000 * factor)  # Cap at 30% reduction
                    else:  # Surplus
                        # Increase NEAT in surplus (more for high responders)
                        adaptation = min(0.5, balance / 1000 * response_factor)  # Cap at 50% increase
                    
                    adapted_neat = current_neat * (1 + adaptation)
                    
                    neat_adaptation.append({
                        'Energy Balance': balance,
                        'NEAT': adapted_neat,
                        'Adaptation (%)': adaptation * 100
                    })
                
                adaptation_df = pd.DataFrame(neat_adaptation)
                
                # Plot
                fig = px.line(adaptation_df, x='Energy Balance', y='NEAT',
                             title=f"NEAT Adaptation to Energy Balance ({neat_responsiveness})")
                
                # Add reference line for current NEAT
                fig.add_hline(y=current_neat, line_dash="dash", line_color="black")
                
                st.plotly_chart(fig)
                
                st.markdown(f"""
                **NEAT Insights:**
                - Your current NEAT is approximately **{int(current_neat)} calories**
                - Going from sedentary to active can increase NEAT by 300-700+ kcal/day
                - NEAT typically decreases during caloric restriction
                - High responders may unconsciously increase NEAT by several hundred calories during overfeeding
                - Your profile suggests you are a **{neat_responsiveness.lower()}**, adapting NEAT in response to energy availability
                """)
        
        with component_tabs[3]:
            st.markdown("<div class='subsection-header'>Environmental & Physiological Factors</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("##### Thermoregulation (Cold-Induced Thermogenesis)")
                
                st.markdown("""
                Cold exposure can increase energy expenditure as the body works to maintain core temperature.
                This includes both shivering and non-shivering thermogenesis.
                """)
                
                # Create temperature slider
                temperature_range = st.slider("Environmental Temperature (°C)", 
                                            min_value=5, max_value=30, value=(16, 26), step=1)
                
                # Calculate thermogenesis across temperature range
                temp_data = []
                brown_fat_activity = st.slider("Brown Fat Activity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
                
                for temp in range(temperature_range[0], temperature_range[1] + 1):
                    # Simulate both cold-acclimated and non-acclimated responses
                    non_acclimated = Thermogenesis().calculate(
                        bmr=tdee_result['BMR'],
                        temperature=temp,
                        brown_fat_activity=brown_fat_activity,
                        cold_acclimated=False,
                        duration_hours=8  # Assuming 8 hours exposure
                    )
                    
                    acclimated = Thermogenesis().calculate(
                        bmr=tdee_result['BMR'],
                        temperature=temp,
                        brown_fat_activity=brown_fat_activity,
                        cold_acclimated=True,
                        duration_hours=8  # Assuming 8 hours exposure
                    )
                    
                    temp_data.append({
                        'Temperature (°C)': temp,
                        'Non-Acclimated (kcal)': non_acclimated,
                        'Acclimated (kcal)': acclimated
                    })
                
                temp_df = pd.DataFrame(temp_data)
                
                # Plot
                fig = px.line(temp_df, x='Temperature (°C)', 
                             y=['Non-Acclimated (kcal)', 'Acclimated (kcal)'],
                             title="Thermogenesis by Temperature (8 hours exposure)")
                
                # Add reference lines for thermoneutral zone
                fig.add_vrect(x0=23, x1=27, fillcolor="lightgreen", opacity=0.2, line_width=0,
                             annotation_text="Thermoneutral Zone")
                
                # Add reference line for shivering threshold
                fig.add_vline(x=16, line_dash="dash", line_color="blue",
                             annotation=dict(text="Shivering Threshold"))
                
                st.plotly_chart(fig)
            
            with col2:
                st.markdown("##### Immune Function & Fever")
                
                st.markdown("""
                Immune system activation, particularly fever, can significantly increase energy expenditure.
                This is often overlooked but can be substantial during illness.
                """)
                
                # Create fever slider
                fever_temp = st.slider("Fever (°C above normal)", 
                                     min_value=0.0, max_value=3.0, value=0.0, step=0.1)
                
                # Calculate immune function expenditure
                if fever_temp > 0:
                    # Calculate fever effect (approximately 10-12% increase per °C)
                    fever_effect = tdee_result['BMR'] * 0.11 * fever_temp
                    
                    # Simulate a range of immune activation levels
                    immune_data = []
                    
                    for activation in np.linspace(0, 1, 11):
                        immune_effect = ImmuneFunction().calculate(
                            bmr=tdee_result['BMR'],
                            fever_celsius=fever_temp,
                            immune_activation=activation
                        )
                        
                        immune_data.append({
                            'Immune Activation': activation,
                            'Energy Expenditure (kcal)': immune_effect,
                            'Percent Increase': (immune_effect / tdee_result['BMR']) * 100
                        })
                    
                    immune_df = pd.DataFrame(immune_data)
                    
                    # Plot
                    fig = px.line(immune_df, x='Immune Activation', y='Energy Expenditure (kcal)',
                                 title=f"Energy Cost of Fever ({fever_temp}°C) & Immune Response")
                    
                    st.plotly_chart(fig)
                    
                    # Show key metrics
                    col2a, col2b = st.columns(2)
                    
                    max_expenditure = immune_df['Energy Expenditure (kcal)'].max()
                    max_pct = immune_df['Percent Increase'].max()
                    
                    col2a.metric("Additional Energy", f"{int(max_expenditure)} kcal/day")
                    col2b.metric("BMR Increase", f"{max_pct:.1f}%")
                    
                    st.markdown(f"""
                    **Fever Insights:**
                    - A fever of {fever_temp}°C increases metabolism by approximately {max_pct:.1f}%
                    - This requires an additional {int(max_expenditure)} kcal/day
                    - Fever's effect is immediate and proportional to temperature elevation
                    - The body often reduces activity during illness, partially offsetting increased BMR
                    """)
                
                else:
                    st.info("Move the fever slider above to see the metabolic impact of fever and immune activation.")
                
                # Menstrual cycle effects (for females)
                if user_data['sex'] == 'F':
                    st.markdown("##### Menstrual Cycle Effects")
                    
                    st.markdown("""
                    The menstrual cycle can cause fluctuations in metabolism, with the luteal phase
                    often having slightly higher energy expenditure.
                    """)
                    
                    # Simulate a typical 28-day cycle
                    cycle_data = []
                    
                    for day in range(1, 29):
                        # Determine phase
                        if day <= 5:
                            phase = "Menstrual"
                            hormone_factor = 0.98  # Slightly decreased metabolism
                        elif day <= 14:
                            phase = "Follicular"
                            hormone_factor = 1.0  # Baseline
                        elif day <= 16:
                            phase = "Ovulation"
                            hormone_factor = 1.02  # Slight increase
                        else:
                            phase = "Luteal"
                            hormone_factor = 1.05  # Increased metabolism
                        
                        # Calculate metabolic effect
                        bmr_effect = tdee_result['BMR'] * hormone_factor
                        
                        cycle_data.append({
                            'Cycle Day': day,
                            'Phase': phase,
                            'BMR (kcal)': bmr_effect,
                            'Change (%)': (hormone_factor - 1) * 100
                        })
                    
                    cycle_df = pd.DataFrame(cycle_data)
                    
                    # Plot
                    fig = px.line(cycle_df, x='Cycle Day', y='BMR (kcal