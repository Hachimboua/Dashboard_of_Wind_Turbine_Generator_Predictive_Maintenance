import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # ensure find_peaks is imported if not already at top level
import warnings
warnings.filterwarnings('ignore')

class WindTurbineDegradationSimulator:
    def __init__(self):
        # Turbine parameters
        self.params = {
            'blade_length': 40,  # m
            'rotor_diameter': 80,  # m
            'rated_power': 1650,  # kW
            'max_power': 1815,  # kW
            'mechanical_power': 1800,  # kW
            'cut_in_speed': 0.1,  # m/s (MODIFIED as per user request)
            'rated_speed': 12,  # m/s
            'cut_out_speed': 20,  # m/s
            'recut_in_speed': 18,  # m/s
            'tip_speed': 61.8,  # m/s
            'gearbox_ratio': 84.3,
            'generator_rpm': 1214,  # rpm
            'synchronous_rpm': 1200,  # rpm
            'slip': 0.0117,
            'air_density': 1.225,  # kg/m³
            'rotor_tilt': 5,  # degrees
            'poles': 6,
            'voltage': 600,  # V
            'frequency': 60,  # Hz
            'max_yaw_rate': 0.5,  # deg/sec
            # NEW Maintenance parameters
            'maintenance_threshold': 0.8,  # Degradation level (0-1) to trigger maintenance
            'maintenance_effectiveness': 0.9, # Percentage of current degradation removed by maintenance (e.g., 0.9 = 90% reduction)
            'time_between_maintenance': 200 # Minimum time steps between maintenance actions
        }
        
        # Degradation state & maintenance tracking
        self.degradation_value = 0.0
        self.time_since_last_maintenance = 0
        self.maintenance_events_count = 0
    
    def calculate_rotor_rpm(self, wind_speed):
        """Calculate rotor RPM based on wind speed and tip speed constraint"""
        tip_speed = self.params['tip_speed']
        blade_length = self.params['blade_length']
        rated_speed = self.params['rated_speed']
        
        rated_rotor_rpm = (tip_speed * 60) / (2 * np.pi * blade_length)
        
        if wind_speed <= 0 : # Avoid division by zero if rated_speed is 0, ensure wind_speed is positive for ratio
             return 0
        if rated_speed <= 0: # Avoid division by zero if rated_speed is 0
             return rated_rotor_rpm # Or some other appropriate default like 0

        if wind_speed <= rated_speed:
            return (rated_rotor_rpm * wind_speed) / rated_speed
        else:
            return rated_rotor_rpm
    
    def calculate_cp(self, tsr, pitch_angle=0):
        """Calculate power coefficient using improved model"""
        lambda_val = tsr
        beta = pitch_angle
        
        if lambda_val < 1 or lambda_val > 12: 
            return 0.0
            
        c1, c2, c3, c4, c5, c6 = 0.5176, 116, 0.4, 5, 21, 0.0068
        
        beta_cubed_plus_1 = beta**3 + 1
        if beta_cubed_plus_1 == 0: 
            lambda_inv_component2 = 0.035 # Or handle as an error/specific case
        else:
            lambda_inv_component2 = 0.035 / beta_cubed_plus_1

        lambda_plus_beta_term = lambda_val + 0.08 * beta
        if lambda_plus_beta_term == 0: 
             return 0.0 

        lambda_inv = 1 / lambda_plus_beta_term - lambda_inv_component2
        cp = c1 * (c2 * lambda_inv - c3 * beta - c4) * np.exp(-c5 * lambda_inv) + c6 * lambda_val
        
        return np.clip(cp, 0.0, 0.48) 
    
    def calculate_system_torque(self, wind_speed):
        """Calculate system torque for given wind speed"""
        if wind_speed < self.params['cut_in_speed'] or wind_speed > self.params['cut_out_speed']:
            return 0, 0  # torque, speed
        
        rotor_radius = self.params['rotor_diameter'] / 2
        rotor_area = np.pi * rotor_radius**2
        air_density = self.params['air_density']
        gearbox_ratio = self.params['gearbox_ratio']
        
        rotor_rpm = self.calculate_rotor_rpm(wind_speed)
        omega_rotor = (rotor_rpm * 2 * np.pi) / 60
        
        gen_rpm = rotor_rpm * gearbox_ratio
        
        tsr = (omega_rotor * rotor_radius) / wind_speed if wind_speed > 0 else 0
        
        cp = self.calculate_cp(tsr)
        
        wind_power = 0.5 * air_density * rotor_area * wind_speed**3
        aero_power = wind_power * cp
        
        if wind_speed <= self.params['rated_speed']:
            actual_power = aero_power
        else:
            actual_power = min(aero_power, self.params['rated_power'] * 1000)
        
        aero_torque = actual_power / omega_rotor if omega_rotor > 0 else 0
        system_torque = aero_torque / 1000  # Convert to kNm
        
        return system_torque, gen_rpm
    
    def simple_cycle_factor(self, temp_data_input):
        temp_data = pd.Series(temp_data_input).dropna() # Ensure it's a Series and drop NaNs
        if len(temp_data) < 3:
            return 0

        temp_range_threshold = 5
        temp_range_power = 1.5
        base_cycle_damage = 0.01
                
        peaks_indices, _ = find_peaks(temp_data.values)
        valleys_indices, _ = find_peaks(-temp_data.values) 

        temp_ranges = []
        
        # Simplified cycle counting: iterate through sorted unique indices of peaks and valleys
        extrema_indices = sorted(list(set(peaks_indices) | set(valleys_indices)))
        
        # This is a very simplified approach, not true rainflow counting.
        # It looks for alternating sequences or just differences between subsequent extrema.
        last_peak_val = None
        last_valley_val = None

        # Iterate through all extrema points
        for i in range(len(extrema_indices) -1):
            idx1 = extrema_indices[i]
            idx2 = extrema_indices[i+1]
            val1 = temp_data.iloc[idx1]
            val2 = temp_data.iloc[idx2]

            # Check if (val1 is peak and val2 is valley) or (val1 is valley and val2 is peak)
            # This is still tricky without full rainflow.
            # A simpler, more direct (but less accurate) way: take all temp differences > threshold
            # between consecutive identified peaks and valleys.
            # The provided find_peaks logic in the original code was also very basic.
            # Let's use a slightly more robust pairing if possible for this simple version:
            
        # Reverting to a simpler interpretation of the original code's intent for pairing for now
        # This part is complex to get right without a full rainflow algorithm.
        # The original code's peak/valley finding was simpler than scipy's find_peaks directly.
        # For now, let's assume the simple_cycle_factor calculation might need a dedicated review
        # if high accuracy thermal cycling is needed. The original one was a rough heuristic.

        # Sticking to a very simple interpretation for now: ranges between consecutive identified peaks and valleys
        # if we have at least one of each.
        if len(peaks_indices) > 0 and len(valleys_indices) > 0:
            # This part needs a more robust cycle identification method than simple iteration
            # For example, using a simplified version of ASTM E1049-85 rainflow.
            # Given the existing simple structure, we'll just sum temperature ranges above threshold
            # This is NOT rainflow counting.
            # This calculates ranges between *all* successive points if they form a peak-valley or valley-peak
            # This logic needs to be carefully considered or replaced with a proper rainflow if accuracy is paramount.
            # The original logic was:
            # min_len = min(len(peaks), len(valleys)) -> assuming peaks and valleys were pre-filtered lists of values not indices
            # if min_len > 1: for i in range(min_len -1): temp_range = abs(peaks[i] - valleys[i]) ...
            # This implies peaks and valleys were already paired. Scipy's find_peaks gives indices.

            # Let's use a simple diff approach on the temp_data itself, looking for reversals
            temp_diffs = np.abs(np.diff(temp_data))
            valid_ranges = temp_diffs[temp_diffs >= temp_range_threshold]
            if len(valid_ranges)>0:
                 weighted_ranges = valid_ranges ** temp_range_power
                 cycle_factor = np.sum(weighted_ranges) * base_cycle_damage
            else:
                 cycle_factor = 0
        else:
            cycle_factor = 0
        
        return cycle_factor

    def wind_turbine_generator_degradation(self, torque, speed, vibration, temp, humidity, cycle_factor, current_time_step):
        alpha1 = 1e-10
        alpha2 = 2e-8
        alpha3 = 1e-9
        alpha4 = 5e-10
        alpha5 = 3e-9
        
        Ea = 0.8
        kb = 8.617e-5
        T_ref = 298.15
        T_K = temp 
        
        temp_acc_factor = np.exp((Ea/kb) * (1/T_ref - 1/T_K)) if T_K > 0 and T_ref > 0 else 1.0 
        
        mech_wear = (abs(torque) * abs(speed)) ** 1.2
        vib_stress = vibration ** 2
        thermal_stress = temp_acc_factor
        humidity_stress = np.exp(0.05 * (humidity - 50)) 
        cycle_stress = alpha5 * cycle_factor
        
        dD = (alpha1 * mech_wear + 
              alpha2 * vib_stress + 
              alpha3 * thermal_stress + 
              alpha4 * humidity_stress + 
              cycle_stress)
        
        self.degradation_value = min(self.degradation_value + dD, 1.0)
        
        maintenance_performed_this_step = False
        self.time_since_last_maintenance += 1

        if (self.degradation_value >= self.params['maintenance_threshold'] and
            self.time_since_last_maintenance >= self.params['time_between_maintenance']):
            
            degradation_reduction = self.degradation_value * self.params['maintenance_effectiveness']
            self.degradation_value -= degradation_reduction
            self.degradation_value = max(self.degradation_value, 0) 
            
            maintenance_performed_this_step = True
            self.maintenance_events_count += 1
            self.time_since_last_maintenance = 0 
        
        return self.degradation_value, maintenance_performed_this_step
 








 
    def plot_torque_time_series(self, df, save_path=None):
        """Plot system torque over time with operating thresholds."""
        plt.figure(figsize=(12, 6))
        
        # Extract torque data (ensure numeric and handle NaNs)
        torque = pd.to_numeric(df['system_torque_knm'], errors='coerce').fillna(0)
        
        # Plot torque
        plt.plot(torque, 'b-', alpha=0.7, linewidth=1, label='System Torque (kNm)')
        
        # Highlight maintenance events (if any)
        if 'maintenance_performed' in df.columns:
            maintenance_steps = df[df['maintenance_performed']].index
            for step in maintenance_steps:
                plt.axvline(x=step, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Time Step')
        plt.ylabel('Torque (kNm)')
        plt.title('System Torque Time Series')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    






    def process_time_series(self, csv_file_path=None, df=None):
        if df is None:
            if csv_file_path is None:
                raise ValueError("Either csv_file_path or df must be provided")
            df = pd.read_csv(csv_file_path)
        
        print(f"Processing {len(df)} data points...")
        
        self.degradation_value = 0.0
        self.time_since_last_maintenance = 0
        self.maintenance_events_count = 0 
        
        results = {
            'system_torque_knm': [],
            'generator_speed_rpm': [],
            'degradation': [],
            'cycle_factor': [],
            'maintenance_performed': [] 
        }
        
        if 'u' in df.columns and 'v' in df.columns:
            # Ensure u and v are numeric, coercing errors to NaN
            u_comp = pd.to_numeric(df['u'], errors='coerce').fillna(0) # Fill NaN with 0 for speed calc
            v_comp = pd.to_numeric(df['v'], errors='coerce').fillna(0) # Fill NaN with 0 for speed calc
            wind_speed_series = np.sqrt(u_comp**2 + v_comp**2).values # Use .values to get numpy array
        elif 'speed' in df.columns:
            wind_speed_series = pd.to_numeric(df['speed'], errors='coerce').fillna(0).values # Use .values
        else:
            raise ValueError("No wind speed data found. Need either 'speed' or 'u','v' components")
        
        # Ensure other series are also numeric and handle NaNs appropriately (e.g., by filling with a default or mean)
        vibration_rms_series = pd.to_numeric(df['U'], errors='coerce').fillna(5).values if 'U' in df.columns else np.ones(len(df)) * 5
        humidity_series = pd.to_numeric(df['r'], errors='coerce').fillna(50).values if 'r' in df.columns else np.ones(len(df)) * 50
        
        if 'temperature' in df.columns:
            temperature_series = pd.to_numeric(df['temperature'], errors='coerce').fillna(20).values # Fill NaN with a default
        else:
            print("INFO: 'temperature' column not found. Estimating temperature profile.")
            base_temp = 20
            # Assuming 10-min data (144 points per day) or 1-min data (1440 points per day)
            # For 7200 points: if 1-min data -> 5 days. if 10-min data -> 50 days.
            # Adjust period for daily variation based on data frequency assumption.
            # If 7200 points = 5 days (1-min interval), then period is 24*60 = 1440 points.
            points_per_day = 1440 # Assuming 1-minute data for a 24h cycle for 7200 points
            daily_variation = 10 * np.sin(2 * np.pi * np.arange(len(df)) / points_per_day) 
            
            # Ensure humidity_series used for effect is 1D numpy array if it came from .values
            humidity_for_effect = humidity_series if isinstance(humidity_series, np.ndarray) else pd.Series(humidity_series).fillna(50).values

            humidity_effect = (humidity_for_effect - 50) * 0.1
            temperature_series = base_temp + daily_variation + humidity_effect + np.random.normal(0, 2, len(df))
            # Store the estimated temperature back into the original df if it's being modified,
            # or ensure it's added to the output_df correctly.
            # For safety, let's ensure 'temperature_estimated' is added to df if not present to avoid issues in plotting
            df['temperature_estimated'] = temperature_series


        cycle_factor_value = self.simple_cycle_factor(temperature_series) # Pass numpy array
        
        for i in range(len(df)): # Loop over the length of the original DataFrame
            current_wind_speed = wind_speed_series[i]
            current_vibration = vibration_rms_series[i]
            current_temp = temperature_series[i]
            current_humidity = humidity_series[i]

            torque, gen_speed = self.calculate_system_torque(current_wind_speed)
            
            degradation, maint_performed = self.wind_turbine_generator_degradation(
                torque,
                gen_speed,
                current_vibration,
                current_temp,
                current_humidity,
                cycle_factor_value,
                i 
            )
            
            results['system_torque_knm'].append(torque)
            results['generator_speed_rpm'].append(gen_speed)
            results['degradation'].append(degradation)
            results['cycle_factor'].append(cycle_factor_value) 
            results['maintenance_performed'].append(maint_performed)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(df)} points, Current degradation: {degradation:.6f}")
        
        results_df = pd.DataFrame(results)
        # Ensure 'df' used in concat has a compatible index with results_df
        output_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        
        return output_df
    
    def plot_degradation_analysis(self, df, save_path=None):
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Wind Turbine Degradation Analysis', fontsize=16)
        
        time_steps = range(len(df))
        
        # Degradation over time with Maintenance Markers
        axes[0,0].plot(time_steps, df['degradation'], 'r-', linewidth=1, label='Degradation')
        maintenance_points = df[df['maintenance_performed'] == True]
        if not maintenance_points.empty:
            axes[0,0].scatter(maintenance_points.index, maintenance_points['degradation'], 
                              marker='v', color='blue', s=60, label='Maintenance Event', zorder=5)
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Degradation Level')
        axes[0,0].set_title('Degradation Evolution with Maintenance')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # --- MODIFIED SECTION FOR WIND SPEED VS SYSTEM TORQUE ---
        plot_wind_vs_torque = True
        wind_values_for_plot = np.array([]) # Initialize as empty numpy array

        if 'speed' in df.columns:
            wind_values_for_plot = pd.to_numeric(df['speed'], errors='coerce').fillna(0).values
        elif 'u' in df.columns and 'v' in df.columns:
            u_comp = pd.to_numeric(df['u'], errors='coerce').fillna(0)
            v_comp = pd.to_numeric(df['v'], errors='coerce').fillna(0)
            wind_values_for_plot = np.sqrt(u_comp**2 + v_comp**2).values
        else:
            print("WARNING: Wind speed data ('speed' or 'u'/'v') not found in df for plot. Torque plot will be affected.")
            if 'system_torque_knm' in df.columns: # Try to get length from another column
                 wind_values_for_plot = np.zeros(len(df['system_torque_knm']))
            else: # If no other column, cannot determine length
                 plot_wind_vs_torque = False


        if plot_wind_vs_torque and 'system_torque_knm' in df.columns:
            torque_values_for_plot = pd.to_numeric(df['system_torque_knm'], errors='coerce').fillna(0).values

            if len(wind_values_for_plot) == len(torque_values_for_plot):
                axes[0,1].scatter(wind_values_for_plot, torque_values_for_plot, alpha=0.3, s=1)
            else:
                axes[0,1].text(0.5, 0.5, "Data length mismatch for\nWind Speed vs System Torque",
                               ha='center', va='center', color='red', fontsize=9)
                print(f"ERROR PLOTTING: Wind Speed data length {len(wind_values_for_plot)}, Torque data length {len(torque_values_for_plot)}")
        elif not plot_wind_vs_torque:
             axes[0,1].text(0.5, 0.5, "Insufficient data for\nWind Speed vs System Torque",
                               ha='center', va='center', color='orange', fontsize=9)
        # --- END OF MODIFIED SECTION ---

        axes[0,1].set_xlabel('Wind Speed (m/s)')
        axes[0,1].set_ylabel('System Torque (kNm)')
        axes[0,1].set_title('Wind Speed vs System Torque')
        axes[0,1].grid(True, alpha=0.3)
        
        temp_col_name = 'temperature' if 'temperature' in df.columns else 'temperature_estimated'
        if temp_col_name not in df.columns: # If even estimated temp is missing
            print(f"Warning: Temperature column '{temp_col_name}' not found for plotting.")
            df[temp_col_name] = np.zeros(len(df)) # Fallback
            
        axes[1,0].plot(time_steps, df[temp_col_name], 'b-', alpha=0.7, linewidth=0.5)
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel(f'Temperature (°C){" (Est.)" if temp_col_name == "temperature_estimated" else ""}')
        axes[1,0].set_title('Temperature Profile')
        axes[1,0].grid(True, alpha=0.3)
        
        degradation_rate = np.diff(df['degradation'].values, prepend=df['degradation'].iloc[0] if len(df['degradation']) > 0 else 0.0) 
        
        vibration_data_for_plot = np.array([])
        if 'U' in df.columns:
             vibration_data_for_plot = pd.to_numeric(df['U'], errors='coerce').fillna(0).values
        else: # Fallback if 'U' is missing
             vibration_data_for_plot = np.zeros(len(degradation_rate))

        if len(vibration_data_for_plot) == len(degradation_rate):
            axes[1,1].scatter(vibration_data_for_plot, degradation_rate, alpha=0.3, s=1) 
        else:
            axes[1,1].text(0.5,0.5, "Data length mismatch for\nVibration vs Degradation Rate", ha='center', va='center', color='red', fontsize=9)
            print(f"ERROR PLOTTING: Vibration data length {len(vibration_data_for_plot)}, Degradation rate length {len(degradation_rate)}")

        # Add this to the subplot grid (adjust indices as needed)
        axes[1, 1].plot(df['system_torque_knm'], 'b-', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('System Torque Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Torque (kNm)')
        axes[1, 1].grid(True, alpha=0.3)
        
        humidity_data_for_plot = np.array([])
        if 'r' in df.columns:
            humidity_data_for_plot = pd.to_numeric(df['r'], errors='coerce').fillna(0).values
        else: # Fallback if 'r' is missing
            humidity_data_for_plot = np.zeros(len(degradation_rate))
        
        if len(humidity_data_for_plot) == len(degradation_rate):
            axes[2,0].scatter(humidity_data_for_plot, degradation_rate, alpha=0.3, s=1, c='green') 
        else:
            axes[2,0].text(0.5,0.5, "Data length mismatch for\nHumidity vs Degradation Rate", ha='center', va='center', color='red', fontsize=9)
            print(f"ERROR PLOTTING: Humidity data length {len(humidity_data_for_plot)}, Degradation rate length {len(degradation_rate)}")

        axes[2,0].set_xlabel('Humidity (r %)')
        axes[2,0].set_ylabel('Degradation Rate (approx.)')
        axes[2,0].set_title('Humidity vs Degradation Rate')
        axes[2,0].grid(True, alpha=0.3)
        
        gen_speed_for_hist = pd.to_numeric(df['generator_speed_rpm'], errors='coerce').fillna(0)
        axes[2,1].hist(gen_speed_for_hist[gen_speed_for_hist > 0], bins=50, alpha=0.7, edgecolor='black') 
        axes[2,1].set_xlabel('Generator Speed (RPM)')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].set_title('Generator Speed Distribution (Non-Zero)')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_degradation_summary(self, df):
        print("\n" + "="*50)
        print("DEGRADATION ANALYSIS SUMMARY")
        print("="*50)
        
        final_degradation = df['degradation'].iloc[-1] if len(df) > 0 else 0
        max_torque = df['system_torque_knm'].max() if len(df) > 0 and 'system_torque_knm' in df.columns else 0
        
        torque_operational = df['system_torque_knm'][df['system_torque_knm'] > 0] if 'system_torque_knm' in df.columns else pd.Series([])
        avg_torque = torque_operational.mean() if not torque_operational.empty else 0

        max_gen_speed = df['generator_speed_rpm'].max() if len(df) > 0 and 'generator_speed_rpm' in df.columns else 0
        
        gen_speed_operational = df['generator_speed_rpm'][df['generator_speed_rpm'] > 0] if 'generator_speed_rpm' in df.columns else pd.Series([])
        avg_gen_speed = gen_speed_operational.mean() if not gen_speed_operational.empty else 0
        
        print(f"Dataset size: {len(df)} time steps")
        print(f"Final degradation level: {final_degradation:.6f} ({final_degradation*100:.4f}%)")
        print(f"Number of maintenance events: {self.maintenance_events_count}") 
        print(f"Overall cycle factor applied: {df['cycle_factor'].iloc[0]:.6f}" if len(df) > 0 and 'cycle_factor' in df.columns else "N/A")
        
        print(f"\nTorque Statistics (for operating conditions):")
        print(f"  Maximum system torque: {max_torque:.2f} kNm")
        print(f"  Average system torque (when >0): {avg_torque:.2f} kNm")
        
        print(f"\nGenerator Speed Statistics (for operating conditions):")
        print(f"  Maximum generator speed: {max_gen_speed:.1f} RPM")
        print(f"  Average generator speed (when >0): {avg_gen_speed:.1f} RPM")
        
        temp_col_name = 'temperature' if 'temperature' in df.columns else 'temperature_estimated'
        print(f"\nEnvironmental Conditions:")
        if temp_col_name in df.columns and len(df) > 0:
            print(f"  Temperature range: {df[temp_col_name].min():.1f}°C to {df[temp_col_name].max():.1f}°C")
        else:
            print(f"  Temperature data: N/A")

        if 'r' in df.columns and len(df) > 0:
             print(f"  Humidity range: {df['r'].min():.1f}% to {df['r'].max():.1f}%")
        else:
            print(f"  Humidity data: N/A")

        if 'U' in df.columns and len(df) > 0:
            print(f"  Vibration RMS (U) range: {df['U'].min():.2f} to {df['U'].max():.2f}")
        else:
            print(f"  Vibration RMS (U) data: N/A")

        if len(df) > 1 and 'degradation' in df.columns:
            degradation_rate = np.diff(df['degradation'].values)
            print(f"\nDegradation Rate Statistics (approximate step-wise):")
            print(f"  Maximum degradation rate step: {degradation_rate.max():.2e}")
            print(f"  Average degradation rate step: {degradation_rate.mean():.2e}")
        else:
            print(f"\nDegradation Rate Statistics: N/A (insufficient data)")

        print("="*50)

# Example usage
if __name__ == "__main__":
    simulator = WindTurbineDegradationSimulator()
    
    print("Wind Turbine Degradation Simulator")
    print(f"Cut-in speed set to: {simulator.params['cut_in_speed']} m/s")
    print(f"Maintenance threshold: {simulator.params['maintenance_threshold']*100}% degradation")
    print(f"Maintenance effectiveness: {simulator.params['maintenance_effectiveness']*100}% reduction")
    print(f"Min steps between maintenance: {simulator.params['time_between_maintenance']}")
    print("="*50)
    
    csv_file_to_try = "2024.csv" 
    df_input = None

    try:
        import os
        if os.path.exists(csv_file_to_try):
            print(f"Attempting to load data from '{csv_file_to_try}'...")
            df_input = pd.read_csv(csv_file_to_try)
            print(f"Successfully loaded '{csv_file_to_try}'. Shape: {df_input.shape}")
            # Basic check for essential columns (example)
            # if not ({'speed'} <= set(df_input.columns) or {'u', 'v'} <= set(df_input.columns)):
            #     print(f"Warning: Loaded CSV missing required wind speed columns ('speed' or both 'u' and 'v').")
        else:
            print(f"File '{csv_file_to_try}' not found. Generating sample data instead.")
            
    except Exception as e:
        print(f"Could not load or use '{csv_file_to_try}' (Reason: {e}). Generating sample data instead.")
    
    if df_input is None or df_input.empty: # If loading failed or file not found
        print("Generating sample 7200-point time series for demonstration...")
        np.random.seed(42)
        n_points = 7200
        time_hours = np.arange(n_points) / 60 

        base_wind = 8 + 3 * np.sin(2 * np.pi * time_hours / 24)
        wind_noise = np.random.normal(0, 2, n_points)
        wind_speeds = np.clip(base_wind + wind_noise, 0, 25) 

        vibration_rms = 3 + wind_speeds * 0.2 + np.random.normal(0, 0.5, n_points)
        vibration_rms = np.clip(vibration_rms, 0.5, 15)

        humidity = 60 + 20 * np.sin(2 * np.pi * time_hours / 24 + np.pi/4) + np.random.normal(0, 5, n_points)
        humidity = np.clip(humidity, 20, 95)
        
        temperature = 15 + 10 * np.sin(2 * np.pi * time_hours / 24 - np.pi/2) + (wind_speeds/10) + np.random.normal(0, 2, n_points)
        temperature = np.clip(temperature, -5, 40)

        df_input = pd.DataFrame({
            'speed': wind_speeds,
            'U': vibration_rms, 
            'V': np.random.gamma(2, 2, n_points) + 3, # Vibration Kurtosis (example)
            'Y': np.random.exponential(1, n_points) + 2.5, # Crest Factor (example)
            'r': humidity,
            'direction': np.random.uniform(0, 360, n_points),
            'temperature': temperature 
        })
        print(f"Sample data generated. Shape: {df_input.shape}")

    if df_input is not None and not df_input.empty:
        print("Processing time series data...")
        df_results = simulator.process_time_series(df=df_input.copy()) # Pass a copy to avoid modifying original df_input
        
        simulator.print_degradation_summary(df_results)
        
        print("\nGenerating analysis plots...")
        simulator.plot_degradation_analysis(df_results, save_path='degradation_analysis_with_maintenance.png')
        
        output_filename = 'testdata.csv'
        try:
            df_results.to_csv(output_filename, index=False)
            print(f"\nResults saved to '{output_filename}'")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")

        print(f"\nKey columns for further analysis:")
        print(f"  - 'degradation': Cumulative degradation level (0-1)")
        print(f"  - 'maintenance_performed': Boolean indicating if maintenance occurred at this step")
    else:
        print("No input data to process. Exiting.")