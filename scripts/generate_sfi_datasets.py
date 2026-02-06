"""
Generate four demonstration datasets for PRISM/ORTHON SFI submission.

All datasets use the canonical PRISM schema:
    cohort (str)   - dataset identifier  
    date (str)     - ISO timestamp
    signal_id (str) - signal name
    value (f64)    - observed value

Each dataset has a known ground-truth event that PRISM should detect
WITHOUT being told when or what it is.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)
OUTPUT_DIR = "/home/claude/datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATASET 1: Community Fragmentation (Izabel - Social Networks)
# =============================================================================
# 
# Scenario: An online community of 30 users over 36 months.
# Each user has 5 network metrics measured monthly.
# At month ~18, a subgroup of 10 users begins forming an echo chamber.
# 
# Signals (per user): activity_rate, reciprocity, clustering_coeff,
#                     betweenness, avg_sentiment
#
# Ground truth: 
#   - Months 1-15: All users behave as one community
#   - Months 16-20: Subgroup begins separating (waterfall recruitment)
#   - Months 21-36: Two distinct cohorts with different dynamics
#
# What PRISM should detect:
#   - correlation_cosine drop around month 16
#   - Pairwise block structure forming
#   - eff_dim changes as community reorganizes
#   - Waterfall recruitment pattern in signal cohorts

def generate_community_fragmentation():
    print("Generating Dataset 1: Community Fragmentation...")
    
    n_users = 30
    n_months = 36
    n_splitters = 10  # users who will form echo chamber
    split_start = 15  # month when separation begins
    split_complete = 22  # month when fully separated
    
    # User groups: 0 = stays in main community, 1 = forms echo chamber
    # Recruitment happens progressively: 2 users, then 3, then 3, then 2
    recruitment_schedule = {
        15: [0, 1],           # first 2 users separate
        17: [2, 3, 4],        # 3 more recruited
        19: [5, 6, 7],        # 3 more
        21: [8, 9],           # final 2
    }
    
    rows = []
    base_date = datetime(2022, 1, 1)
    
    for month in range(n_months):
        date = (base_date + timedelta(days=30 * month)).strftime("%Y-%m-%d")
        
        # Determine which splitter users have separated by this month
        separated = set()
        for recruit_month, user_ids in recruitment_schedule.items():
            if month >= recruit_month:
                separated.update(user_ids)
        
        for user_idx in range(n_users):
            is_splitter = user_idx < n_splitters
            has_separated = user_idx in separated
            
            # Base signal dynamics (healthy community)
            base_activity = 5.0 + 0.5 * np.sin(2 * np.pi * month / 12)  # seasonal
            base_reciprocity = 0.65
            base_clustering = 0.45
            base_betweenness = 0.03 + 0.01 * (user_idx / n_users)  # varies by user
            base_sentiment = 0.6
            
            if has_separated:
                # Separated users: increasing internal cohesion, decreasing external
                separation_strength = min(1.0, (month - split_start) / 10.0)
                
                # Activity increases within echo chamber
                activity = base_activity * (1.0 + 0.4 * separation_strength)
                # Reciprocity increases (talking to same people)
                reciprocity = base_reciprocity + 0.25 * separation_strength
                # Clustering increases (tight subgroup)
                clustering = base_clustering + 0.35 * separation_strength
                # Betweenness drops (no longer bridging)
                betweenness = base_betweenness * (1.0 - 0.7 * separation_strength)
                # Sentiment becomes more extreme
                sentiment = base_sentiment + 0.3 * separation_strength
            
            elif is_splitter and not has_separated:
                # Splitter users who haven't separated yet: subtle drift
                drift = 0.1 * max(0, (month - 12) / 10.0)
                activity = base_activity * (1.0 + drift)
                reciprocity = base_reciprocity + 0.05 * drift
                clustering = base_clustering + 0.08 * drift
                betweenness = base_betweenness * (1.0 - 0.1 * drift)
                sentiment = base_sentiment + 0.05 * drift
            
            else:
                # Main community: gradually loses some members
                if month > split_start:
                    loss_effect = 0.15 * min(1.0, (month - split_start) / 10.0)
                else:
                    loss_effect = 0.0
                
                activity = base_activity * (1.0 - 0.1 * loss_effect)
                reciprocity = base_reciprocity - 0.05 * loss_effect
                clustering = base_clustering - 0.03 * loss_effect
                betweenness = base_betweenness * (1.0 + 0.2 * loss_effect)
                sentiment = base_sentiment - 0.05 * loss_effect
            
            # Add noise
            noise_scale = 0.08
            user_label = f"user_{user_idx:03d}"
            
            signals = {
                f"{user_label}_activity": activity + np.random.normal(0, noise_scale * activity),
                f"{user_label}_reciprocity": np.clip(reciprocity + np.random.normal(0, noise_scale), 0, 1),
                f"{user_label}_clustering": np.clip(clustering + np.random.normal(0, noise_scale), 0, 1),
                f"{user_label}_betweenness": max(0, betweenness + np.random.normal(0, noise_scale * 0.5)),
                f"{user_label}_sentiment": np.clip(sentiment + np.random.normal(0, noise_scale), -1, 1),
            }
            
            for signal_id, value in signals.items():
                rows.append({
                    "cohort": "community_alpha",
                    "date": date,
                    "signal_id": signal_id,
                    "value": round(float(value), 6),
                })
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "01_community_fragmentation.parquet")
    df.to_parquet(path, index=False)
    print(f"  -> {path}: {len(df)} rows, {df['signal_id'].nunique()} signals, {n_months} timesteps")
    print(f"  -> Ground truth: recruitment cascade starts month {split_start}, completes month {split_complete}")
    print(f"  -> {n_splitters} users form echo chamber, {n_users - n_splitters} remain")
    return df


# =============================================================================
# DATASET 2: Perception-Reality Divergence (Izabel - Social Theory)
# =============================================================================
#
# Scenario: 20 individuals in an organization surveyed monthly for 30 months.
# Two measurement sources per person:
#   - ACTUAL: derived from email/message logs (ground truth interactions)
#   - PERCEIVED: from monthly survey ("who do you interact with most?")
#
# At first, perception matches reality. Over time, cognitive biases cause
# drift: people over-report interactions with high-status individuals and
# under-report interactions with lower-status ones.
#
# An organizational restructuring at month 18 creates a sudden shift in
# actual interaction patterns, but perceived patterns lag behind.
#
# What PRISM should detect:
#   - Divergence between actual and perceived signal groups
#   - correlation_cosine tracking the perception lag
#   - eff_dim difference between actual (reorganizes) and perceived (sticky)

def generate_perception_reality():
    print("\nGenerating Dataset 2: Perception-Reality Divergence...")
    
    n_people = 20
    n_months = 30
    reorg_month = 18  # organizational restructuring
    
    # Status hierarchy (affects perception bias)
    status = np.array([1.0 - (i / n_people) for i in range(n_people)])
    
    rows = []
    base_date = datetime(2023, 1, 1)
    
    # Pre-reorg team structure: 2 teams of 10
    team_a = set(range(10))
    team_b = set(range(10, 20))
    
    # Post-reorg: 4 teams of 5 (cross-cutting the original teams)
    post_teams = [set(range(0, 5)), set(range(5, 10)), 
                  set(range(10, 15)), set(range(15, 20))]
    
    for month in range(n_months):
        date = (base_date + timedelta(days=30 * month)).strftime("%Y-%m-%d")
        
        # Transition factor for restructuring
        if month < reorg_month:
            transition = 0.0
        else:
            transition = min(1.0, (month - reorg_month) / 6.0)
        
        # Perception bias grows over time (people settle into assumptions)
        bias_strength = min(0.6, 0.02 * month)
        
        for person in range(n_people):
            # --- ACTUAL interaction metrics ---
            # Before reorg: interact mostly within original team
            if person in team_a:
                same_team_actual_pre = 0.7
            else:
                same_team_actual_pre = 0.7
            
            # After reorg: interact within new smaller teams
            same_team_actual_post = 0.8
            
            actual_in_group = same_team_actual_pre * (1 - transition) + same_team_actual_post * transition
            actual_diversity = 0.5 * (1 - transition) + 0.3 * transition  # less diverse post-reorg
            actual_frequency = 15.0 + np.random.normal(0, 2)
            
            # Reorg causes temporary spike in cross-team interaction
            if reorg_month <= month < reorg_month + 4:
                actual_diversity += 0.2 * (1 - (month - reorg_month) / 4)
                actual_frequency += 5.0
            
            actual_centrality = 0.1 + 0.4 * status[person] * (1 - 0.3 * transition)
            
            # --- PERCEIVED interaction metrics ---
            # Perception is sticky - lags behind actual changes
            perception_lag = 4  # months of lag
            if month < reorg_month + perception_lag:
                perceived_transition = 0.0
            else:
                perceived_transition = min(1.0, (month - reorg_month - perception_lag) / 8.0)
            
            # Status bias: people think they interact more with high-status individuals
            perceived_in_group = actual_in_group * (1 - bias_strength) + \
                                 (actual_in_group + 0.15 * status[person]) * bias_strength
            perceived_in_group = np.clip(perceived_in_group, 0, 1)
            
            # People overestimate their own diversity and frequency
            perceived_diversity = actual_diversity + 0.1 * bias_strength + 0.05 * status[person]
            perceived_frequency = actual_frequency * (1.0 + 0.2 * bias_strength)
            perceived_centrality = actual_centrality + 0.15 * bias_strength
            
            # Apply perception lag to restructuring
            perceived_in_group = perceived_in_group * (1 - perceived_transition) + \
                                 same_team_actual_post * perceived_transition
            
            person_label = f"person_{person:02d}"
            noise = 0.04
            
            signals = {
                f"{person_label}_actual_in_group": np.clip(actual_in_group + np.random.normal(0, noise), 0, 1),
                f"{person_label}_actual_diversity": np.clip(actual_diversity + np.random.normal(0, noise), 0, 1),
                f"{person_label}_actual_frequency": max(0, actual_frequency + np.random.normal(0, 1.5)),
                f"{person_label}_actual_centrality": np.clip(actual_centrality + np.random.normal(0, noise), 0, 1),
                f"{person_label}_perceived_in_group": np.clip(perceived_in_group + np.random.normal(0, noise), 0, 1),
                f"{person_label}_perceived_diversity": np.clip(perceived_diversity + np.random.normal(0, noise), 0, 1),
                f"{person_label}_perceived_frequency": max(0, perceived_frequency + np.random.normal(0, 1.5)),
                f"{person_label}_perceived_centrality": np.clip(perceived_centrality + np.random.normal(0, noise), 0, 1),
            }
            
            for signal_id, value in signals.items():
                rows.append({
                    "cohort": "org_beta",
                    "date": date,
                    "signal_id": signal_id,
                    "value": round(float(value), 6),
                })
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "02_perception_reality.parquet")
    df.to_parquet(path, index=False)
    print(f"  -> {path}: {len(df)} rows, {df['signal_id'].nunique()} signals, {n_months} timesteps")
    print(f"  -> Ground truth: restructuring at month {reorg_month}, perception lag ~4 months")
    print(f"  -> Bias grows linearly, status hierarchy drives perception distortion")
    return df


# =============================================================================
# DATASET 3: Ecological Regime Shift (Domain-Agnostic Demo)
# =============================================================================
#
# Scenario: A lake ecosystem with 12 species measured monthly for 48 months.
# Nutrient loading increases gradually. At month ~30, the ecosystem tips
# from a clear-water state (diverse plankton, submerged vegetation) to a
# turbid state (cyanobacteria dominated, low diversity).
#
# This is the classic Scheffer et al. shallow lake bistability model.
#
# Signals: abundance of 12 species groups
# Ground truth: regime shift at ~month 30
#
# What PRISM should detect:
#   - eff_dim declining as cyanobacteria dominate
#   - Critical slowing down (ACF lag-1 → 1) before transition
#   - Correlation structure reorganization
#   - Abrupt dimensional collapse at tipping point

def generate_ecological_regime_shift():
    print("\nGenerating Dataset 3: Ecological Regime Shift...")
    
    n_months = 48
    tipping_month = 30
    
    # Species groups and their clear-water equilibrium abundances
    species = {
        "diatoms": 800,
        "green_algae": 400,
        "cyanobacteria": 100,
        "dinoflagellates": 200,
        "daphnia": 300,
        "copepods": 250,
        "rotifers": 500,
        "submerged_veg": 600,
        "benthic_algae": 350,
        "fish_larvae": 150,
        "chironomids": 400,
        "dissolved_oxygen": 9.5,  # mg/L, not a species but critical
    }
    
    rows = []
    base_date = datetime(2020, 1, 1)
    
    # Track previous values for autocorrelation / critical slowing down
    prev_values = {sp: val for sp, val in species.items()}
    
    for month in range(n_months):
        date = (base_date + timedelta(days=30 * month)).strftime("%Y-%m-%d")
        
        # Nutrient loading increases linearly
        nutrient_pressure = month / n_months  # 0 to 1
        
        # Distance to tipping point affects noise recovery
        if month < tipping_month - 8:
            # Far from tipping: fast recovery, low autocorrelation
            recovery_rate = 0.7
            regime = "clear"
        elif month < tipping_month:
            # Approaching tipping: critical slowing down
            proximity = (tipping_month - month) / 8.0  # 1.0 → 0.0
            recovery_rate = 0.7 * proximity + 0.05 * (1 - proximity)
            regime = "approaching"
        else:
            # Post-tipping: new regime
            recovery_rate = 0.6  # recovers to new equilibrium
            regime = "turbid"
        
        # Seasonal cycle
        season = 0.15 * np.sin(2 * np.pi * (month - 3) / 12)  # peak in summer
        
        for sp_name, base_val in species.items():
            if regime == "turbid":
                post_tip_months = month - tipping_month
                transition = min(1.0, post_tip_months / 6.0)
                
                # New equilibrium values in turbid state
                turbid_targets = {
                    "cyanobacteria": base_val * 12.0,   # dominant
                    "diatoms": base_val * 0.15,          # suppressed
                    "green_algae": base_val * 0.3,       # reduced
                    "dinoflagellates": base_val * 0.2,   # reduced
                    "daphnia": base_val * 0.1,           # collapsed (grazers fail)
                    "copepods": base_val * 0.4,          # reduced
                    "rotifers": base_val * 1.3,          # slight increase
                    "submerged_veg": base_val * 0.05,    # nearly gone
                    "benthic_algae": base_val * 0.1,     # light-limited
                    "fish_larvae": base_val * 0.3,       # reduced
                    "chironomids": base_val * 1.5,       # increase in turbid
                    "dissolved_oxygen": base_val * 0.6,  # drops
                }
                target = turbid_targets.get(sp_name, base_val * 0.5)
                equilibrium = base_val * (1 - transition) + target * transition
                
            elif regime == "approaching":
                # Gradual degradation approaching tipping
                degradation = nutrient_pressure * 0.3
                if sp_name == "cyanobacteria":
                    equilibrium = base_val * (1 + degradation * 3)
                elif sp_name == "submerged_veg":
                    equilibrium = base_val * (1 - degradation * 1.5)
                elif sp_name == "daphnia":
                    equilibrium = base_val * (1 - degradation * 0.8)
                elif sp_name == "dissolved_oxygen":
                    equilibrium = base_val * (1 - degradation * 0.3)
                else:
                    equilibrium = base_val * (1 - degradation * 0.2)
            else:
                # Clear water state with minor nutrient effects
                if sp_name == "cyanobacteria":
                    equilibrium = base_val * (1 + nutrient_pressure * 0.5)
                else:
                    equilibrium = base_val * (1 + nutrient_pressure * 0.05)
            
            # Apply seasonal variation (not to dissolved oxygen the same way)
            if sp_name == "dissolved_oxygen":
                seasonal_effect = -0.05 * season  # O2 inversely related to temp
            else:
                seasonal_effect = season
            
            equilibrium *= (1 + seasonal_effect)
            
            # AR(1) process with critical slowing down
            innovation = np.random.normal(0, 0.08 * abs(equilibrium))
            value = recovery_rate * equilibrium + (1 - recovery_rate) * prev_values[sp_name] + innovation
            value = max(0.01 if sp_name != "dissolved_oxygen" else 2.0, value)
            
            prev_values[sp_name] = value
            
            rows.append({
                "cohort": "lake_gamma",
                "date": date,
                "signal_id": sp_name,
                "value": round(float(value), 4),
            })
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "03_ecological_regime_shift.parquet")
    df.to_parquet(path, index=False)
    print(f"  -> {path}: {len(df)} rows, {df['signal_id'].nunique()} signals, {n_months} timesteps")
    print(f"  -> Ground truth: tipping point at month {tipping_month}")
    print(f"  -> Critical slowing down begins ~month {tipping_month - 8}")
    print(f"  -> Clear-water → turbid regime transition (Scheffer model)")
    return df


# =============================================================================
# DATASET 4: Structural Health Monitoring (Domain-Agnostic Demo)
# =============================================================================
#
# Scenario: A bridge with 15 sensors (accelerometers, strain gauges,
# displacement sensors) monitored daily for 365 days.
# Gradual corrosion weakens a structural member. Degradation accelerates
# after day ~200. Failure-level readings by day ~340.
#
# Signals: vibration_freq_1..5, strain_1..5, displacement_1..5
# Ground truth: degradation onset ~day 120, acceleration ~day 200
#
# What PRISM should detect:
#   - Slow eff_dim decline as corrosion creates coupled responses
#   - Pairwise block structure as nearby sensors correlate
#   - correlation_cosine shift when degradation accelerates
#   - Temperature-driven seasonal variation (not degradation) 

def generate_structural_health():
    print("\nGenerating Dataset 4: Structural Health Monitoring...")
    
    n_days = 365
    degradation_onset = 120
    acceleration_point = 200
    
    # Sensor layout: 3 types × 5 locations
    sensor_types = ["vibration", "strain", "displacement"]
    n_locations = 5
    
    # Location 3 is nearest to the corroding member
    damage_proximity = {0: 0.1, 1: 0.3, 2: 0.7, 3: 1.0, 4: 0.5}
    
    rows = []
    base_date = datetime(2024, 1, 1)
    
    prev_values = {}
    
    for day in range(n_days):
        date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        
        # Temperature effect (seasonal)
        temp = 15 + 12 * np.sin(2 * np.pi * (day - 80) / 365)  # peak in July
        temp_effect = (temp - 15) / 30  # normalized
        
        # Traffic load (weekly cycle + random)
        weekday = day % 7
        traffic = 1.0 + 0.3 * (weekday < 5) + np.random.normal(0, 0.1)
        
        # Degradation factor
        if day < degradation_onset:
            degradation = 0.0
        elif day < acceleration_point:
            degradation = 0.3 * ((day - degradation_onset) / (acceleration_point - degradation_onset))
        else:
            # Accelerating degradation
            linear_part = 0.3
            accel_part = 0.7 * ((day - acceleration_point) / (n_days - acceleration_point)) ** 1.5
            degradation = linear_part + accel_part
        
        for loc in range(n_locations):
            prox = damage_proximity[loc]
            local_degradation = degradation * prox
            
            # --- Vibration frequency ---
            # Healthy: ~25 Hz, decreases with damage (stiffness loss)
            base_freq = 25.0 - 2.0 * loc * 0.1  # slight variation by location
            vib_freq = base_freq * (1 - 0.15 * local_degradation) + \
                       0.3 * temp_effect + \
                       np.random.normal(0, 0.2)
            
            # --- Strain ---
            # Healthy: ~200 microstrain under load
            base_strain = 200 + 20 * loc
            strain_val = base_strain * traffic * (1 + 0.4 * local_degradation) + \
                         15 * temp_effect + \
                         np.random.normal(0, 8)
            
            # --- Displacement ---
            # Healthy: ~2mm under load
            base_disp = 2.0 + 0.3 * loc
            disp_val = base_disp * traffic * (1 + 0.3 * local_degradation) + \
                       0.1 * temp_effect + \
                       np.random.normal(0, 0.15)
            
            # As degradation progresses, sensors near damage become correlated
            # (they respond to the same structural weakness)
            if local_degradation > 0.2:
                coupling = (local_degradation - 0.2) * 0.5
                # Add shared noise component
                shared_noise = np.random.normal(0, coupling * 5)
                strain_val += shared_noise
                disp_val += shared_noise * 0.02
                vib_freq -= abs(shared_noise) * 0.01
            
            signals = {
                f"vibration_loc{loc}": round(float(max(10, vib_freq)), 4),
                f"strain_loc{loc}": round(float(max(0, strain_val)), 4),
                f"displacement_loc{loc}": round(float(max(0, disp_val)), 4),
            }
            
            for signal_id, value in signals.items():
                rows.append({
                    "cohort": "bridge_delta",
                    "date": date,
                    "signal_id": signal_id,
                    "value": value,
                })
    
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "04_structural_health.parquet")
    df.to_parquet(path, index=False)
    print(f"  -> {path}: {len(df)} rows, {df['signal_id'].nunique()} signals, {n_days} timesteps")
    print(f"  -> Ground truth: degradation onset day {degradation_onset}, acceleration day {acceleration_point}")
    print(f"  -> Location 3 is nearest to damage, proximity-weighted effects")
    return df


# =============================================================================
# GENERATE ALL + SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PRISM Demonstration Dataset Generator")
    print("Canonical schema: cohort, date, signal_id, value")
    print("=" * 70)
    print()
    
    df1 = generate_community_fragmentation()
    df2 = generate_perception_reality()
    df3 = generate_ecological_regime_shift()
    df4 = generate_structural_health()
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Dataset':<40} {'Rows':>8} {'Signals':>8} {'Domain'}")
    print("-" * 75)
    print(f"{'01_community_fragmentation.parquet':<40} {len(df1):>8,} {df1['signal_id'].nunique():>8} Social Network")
    print(f"{'02_perception_reality.parquet':<40} {len(df2):>8,} {df2['signal_id'].nunique():>8} Social Theory")
    print(f"{'03_ecological_regime_shift.parquet':<40} {len(df3):>8,} {df3['signal_id'].nunique():>8} Ecology")
    print(f"{'04_structural_health.parquet':<40} {len(df4):>8,} {df4['signal_id'].nunique():>8} Engineering")
    print()
    print("All files in:", OUTPUT_DIR)
    print()
    print("KEY POINT FOR SFI: Same pipeline, same math, same 21 output files.")
    print("The ONLY difference is how ORTHON interprets the results.")
    print()
    print("Ground Truth Events (what PRISM should detect WITHOUT being told):")
    print("  1. Echo chamber recruitment cascade starting month 15")
    print("  2. Org restructuring at month 18, perception lags by ~4 months")  
    print("  3. Lake ecosystem tipping point at month 30 (critical slowing down from month 22)")
    print("  4. Bridge corrosion onset day 120, acceleration day 200")
