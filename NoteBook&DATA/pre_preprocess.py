import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class WindTurbineDegradationSimulator:
    def __init__(self):
        
        self.params = {
            'blade_length': 40, 
            'rotor_diameter': 80,  
            'rated_power': 1650,  
            'max_power': 1815,  
            'mechanical_power': 1800, 
            'cut_in_speed': 0.1,  
            'rated_speed': 12,  
            'cut_out_speed': 20,  
            'recut_in_speed': 18,  
            'tip_speed': 61.8,  
            'gearbox_ratio': 84.3,
            'generator_rpm': 1214,  
            'synchronous_rpm': 1200,  
            'slip': 0.0117,
            'air_density': 1.225,  
            'rotor_tilt': 5,  
            'poles': 6,
            'voltage': 600,  
            'frequency': 60,  
            'max_yaw_rate': 0.5,  
            
            'maintenance_threshold': 0.8,  
            'maintenance_effectiveness': 0.9,
            'time_between_maintenance': 200 
        }
        
        
        self.degradation_value = 0.0
        self.time_since_last_maintenance = 0
        self.maintenance_events_count = 0
    
    def calculate_rotor_rpm(self, wind_speed):
        """Calculate rotor RPM based on wind speed and tip speed constraint"""
        tip_speed = self.params['tip_speed']
        blade_length = self.params['blade_length']
        rated_speed = self.params['rated_speed']
        
        rated_rotor_rpm = (tip_speed * 60) / (2 * np.pi * blade_length)
        
        if wind_speed <= 0 or rated_speed <= 0:
            return 0

        if wind_speed <= rated_speed:
            return (rated_rotor_rpm * wind_speed) / rated_speed
        else:
            return rated_rotor_rpm
    
    def calculate_cp(self, tsr, pitch_angle=0):
        """Calculate power coefficient using an improved model"""
        lambda_val = tsr
        beta = pitch_angle
        
        if not (1 <= lambda_val <= 12): 
            return 0.0
            
        c1, c2, c3, c4, c5, c6 = 0.5176, 116, 0.4, 5, 21, 0.0068
        
        beta_cubed_plus_1 = beta**3 + 1
        lambda_inv_component2 = 0.035 / beta_cubed_plus_1 if beta_cubed_plus_1 != 0 else 0.035

        lambda_plus_beta_term = lambda_val + 0.08 * beta
        if lambda_plus_beta_term == 0: 
            return 0.0 

        lambda_inv = 1 / lambda_plus_beta_term - lambda_inv_component2
        cp = c1 * (c2 * lambda_inv - c3 * beta - c4) * np.exp(-c5 * lambda_inv) + c6 * lambda_val
        
        return np.clip(cp, 0.0, 0.48) 
    
    def calculate_system_torque(self, wind_speed):
        """Calculate system torque for a given wind speed"""
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
        
        actual_power = min(aero_power, self.params['rated_power'] * 1000) if wind_speed > self.params['rated_speed'] else aero_power
        
        aero_torque = actual_power / omega_rotor if omega_rotor > 0 else 0
        system_torque = aero_torque / 1000  # Convert to kNm
        
        return system_torque, gen_rpm
    
    def simple_cycle_factor(self, temp_data_input):
        """Calculates a fatigue factor based on temperature fluctuations."""
        temp_data = pd.Series(temp_data_input).dropna()
        if len(temp_data) < 3:
            return 0

        temp_range_threshold = 5
        temp_range_power = 1.5
        base_cycle_damage = 0.01
        
        temp_diffs = np.abs(np.diff(temp_data))
        valid_ranges = temp_diffs[temp_diffs >= temp_range_threshold]
        
        if len(valid_ranges) > 0:
            weighted_ranges = valid_ranges ** temp_range_power
            cycle_factor = np.sum(weighted_ranges) * base_cycle_damage
        else:
            cycle_factor = 0
            
        return cycle_factor

    def wind_turbine_generator_degradation(self, torque, speed, vibration, temp, humidity, cycle_factor, current_time_step):
        """Calculates the degradation increase for a single time step and handles maintenance."""
        alpha1, alpha2, alpha3, alpha4, alpha5 = 1e-10, 2e-8, 1e-9, 5e-10, 3e-9
        
        Ea, kb, T_ref = 0.8, 8.617e-5, 298.15
        T_K = temp + 273.15 # Convert temp from C to Kelvin
        
        temp_acc_factor = np.exp((Ea/kb) * (1/T_ref - 1/T_K)) if T_K > 0 else 1.0
        
        mech_wear = (abs(torque) * abs(speed)) ** 1.2
        vib_stress = vibration ** 2
        thermal_stress = temp_acc_factor
        humidity_stress = np.exp(0.05 * (humidity - 50)) 
        cycle_stress = alpha5 * cycle_factor
        
        dD = (alpha1 * mech_wear + alpha2 * vib_stress + alpha3 * thermal_stress + 
              alpha4 * humidity_stress + cycle_stress)
        
        self.degradation_value = min(self.degradation_value + dD, 1.0)
        
        self.time_since_last_maintenance += 1

        if (self.degradation_value >= self.params['maintenance_threshold'] and
            self.time_since_last_maintenance >= self.params['time_between_maintenance']):
            
            degradation_reduction = self.degradation_value * self.params['maintenance_effectiveness']
            self.degradation_value = max(self.degradation_value - degradation_reduction, 0)
            
            self.maintenance_events_count += 1
            self.time_since_last_maintenance = 0 
        
        return self.degradation_value

    def process_time_series(self, csv_file_path=None, df=None):
        """Processes a time series DataFrame to calculate torque, speed, and degradation."""
        if df is None:
            if csv_file_path is None:
                raise ValueError("Either csv_file_path or df must be provided")
            df = pd.read_csv(csv_file_path)
        
        print(f"Processing {len(df)} data points...")
        
        self.degradation_value = 0.0
        self.time_since_last_maintenance = 0
        self.maintenance_events_count = 0 
        
        # MODIFICATION 1: Add the new column key to the results dictionary
        results = {
            'system_torque_knm': [], 'generator_speed_rpm': [], 'degradation': [],
            'cycle_factor': [], 'maintenance_performed': [], 'time_since_last_maintenance': []
        }
        
        if 'u' in df.columns and 'v' in df.columns:
            u_comp = pd.to_numeric(df['u'], errors='coerce').fillna(0)
            v_comp = pd.to_numeric(df['v'], errors='coerce').fillna(0)
            wind_speed_series = np.sqrt(u_comp**2 + v_comp**2).values
        elif 'speed' in df.columns:
            wind_speed_series = pd.to_numeric(df['speed'], errors='coerce').fillna(0).values
        else:
            raise ValueError("No wind speed data found. Need 'speed' or 'u'/'v' columns.")
        
        if 'U' in df.columns:
            vibration_rms_series = pd.to_numeric(df['U'], errors='coerce').fillna(5).values
        else:
            print("INFO: 'U' (vibration) column not found. Using default value of 5.0.")
            vibration_rms_series = np.full(len(df), 5.0)

        if 'r' in df.columns:
            humidity_series = pd.to_numeric(df['r'], errors='coerce').fillna(50).values
        else:
            print("INFO: 'r' (humidity) column not found. Using default value of 50.0.")
            humidity_series = np.full(len(df), 50.0)
        
        if 'temperature' in df.columns:
            temperature_series = pd.to_numeric(df['temperature'], errors='coerce').fillna(20).values
        else:
            print("INFO: 'temperature' column not found. Estimating synthetic profile.")
            points_per_day = 1440
            daily_variation = 10 * np.sin(2 * np.pi * np.arange(len(df)) / points_per_day)
            humidity_effect = (humidity_series - 50) * 0.1
            temperature_series = 20 + daily_variation + humidity_effect + np.random.normal(0, 2, len(df))
            df['temperature_estimated'] = temperature_series

        cycle_factor_value = self.simple_cycle_factor(temperature_series)
        
        for i in range(len(df)):
            torque, gen_speed = self.calculate_system_torque(wind_speed_series[i])
            
            degradation = self.wind_turbine_generator_degradation(
                torque, gen_speed, vibration_rms_series[i], temperature_series[i],
                humidity_series[i], cycle_factor_value, i
            )
            
            results['system_torque_knm'].append(torque)
            results['generator_speed_rpm'].append(gen_speed)
            results['degradation'].append(degradation)
            results['cycle_factor'].append(cycle_factor_value)
            results['maintenance_performed'].append(self.maintenance_events_count)
            # MODIFICATION 2: Append the current value for the new column
            results['time_since_last_maintenance'].append(self.time_since_last_maintenance)
            
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(df)} points, Current degradation: {degradation:.6f}")
        
        results_df = pd.DataFrame(results)
        output_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        
        return output_df
    
    def plot_degradation_analysis(self, df, save_path=None):
        """Plots a comprehensive analysis of the degradation simulation results."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Wind Turbine Degradation Analysis', fontsize=16)
        
        time_steps = df.index
        
        axes[0,0].plot(time_steps, df['degradation'], 'r-', linewidth=1.5, label='Degradation')
        maintenance_points = df[df['maintenance_performed'].diff() > 0]
        if not maintenance_points.empty:
            axes[0,0].scatter(maintenance_points.index, maintenance_points['degradation'], 
                              marker='v', color='blue', s=80, label='Maintenance Event', zorder=5)
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Degradation Level')
        axes[0,0].set_title('Degradation Evolution with Maintenance')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        wind_col = 'speed' if 'speed' in df.columns else None
        if not wind_col and ('u' in df.columns and 'v' in df.columns):
            df['calculated_speed'] = np.sqrt(df['u']**2 + df['v']**2)
            wind_col = 'calculated_speed'
        if wind_col:
            axes[0,1].scatter(df[wind_col], df['system_torque_knm'], alpha=0.3, s=2)
        else:
            axes[0,1].text(0.5, 0.5, "Wind Speed Data Not Found", ha='center', va='center')
        axes[0,1].set_xlabel('Wind Speed (m/s)')
        axes[0,1].set_ylabel('System Torque (kNm)')
        axes[0,1].set_title('Wind Speed vs System Torque')
        axes[0,1].grid(True, alpha=0.3)
        
        temp_col_name = 'temperature' if 'temperature' in df.columns else 'temperature_estimated'
        axes[1,0].plot(time_steps, df[temp_col_name], 'c-', alpha=0.7, linewidth=1)
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel(f'Temperature (°C){" (Est.)" if temp_col_name == "temperature_estimated" else ""}')
        axes[1,0].set_title('Temperature Profile')
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df['system_torque_knm'], 'b-', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('System Torque Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Torque (kNm)')
        axes[1, 1].grid(True, alpha=0.3)
        
        ax5 = axes[2, 0]
        degradation_rate = df['degradation'].diff().fillna(0)
        humidity_col = 'r' if 'r' in df.columns else None
        if humidity_col:
            plot_df = pd.DataFrame({'humidity': df[humidity_col], 'rate': degradation_rate})
            plot_df['rate'][plot_df['rate'] < 0] = np.nan
            plot_df.dropna(inplace=True)
            if not plot_df.empty:
                ax5.scatter(plot_df['humidity'], plot_df['rate'], alpha=0.4, s=5, c='green')
                upper_limit = plot_df['rate'].quantile(0.99)
                ax5.set_ylim(bottom=-0.000001, top=upper_limit * 1.1)
            else:
                 ax5.text(0.5, 0.5, "No positive degradation to plot", ha='center', va='center')
        else:
            ax5.text(0.5, 0.5, "Humidity Data Not Found", ha='center', va='center')
        ax5.set_xlabel('Humidity (%)')
        ax5.set_ylabel('Degradation Rate (per step)')
        ax5.set_title('Humidity vs. Degradation Rate')
        ax5.grid(True, alpha=0.3)

        gen_speed_for_hist = pd.to_numeric(df['generator_speed_rpm'], errors='coerce').fillna(0)
        axes[2,1].hist(gen_speed_for_hist[gen_speed_for_hist > 0], bins=50, alpha=0.7, color='purple', edgecolor='black') 
        axes[2,1].set_xlabel('Generator Speed (RPM)')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].set_title('Generator Speed Distribution (Operating)')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def print_degradation_summary(self, df):
        """Prints a summary of the degradation analysis results."""
        print("\n" + "="*50)
        print("DEGRADATION ANALYSIS SUMMARY")
        print("="*50)
        
        if df.empty:
            print("No data to summarize.")
            print("="*50)
            return
            
        final_degradation = df['degradation'].iloc[-1]
        total_maintenance = int(df['maintenance_performed'].max())
        # MODIFICATION 3: Get the final value for the new column
        time_since_maint = df['time_since_last_maintenance'].iloc[-1]

        print(f"Dataset size:                 {len(df)} time steps")
        print(f"Final degradation level:      {final_degradation:.6f} ({final_degradation*100:.4f}%)")
        print(f"Total maintenance events:     {total_maintenance}")
        print(f"Time since last maintenance:  {time_since_maint} steps")

        if 'cycle_factor' in df.columns:
            print(f"Overall cycle factor applied: {df['cycle_factor'].iloc[0]:.6f}")

        print("="*50)

# Main execution block
if __name__ == "__main__":
    simulator = WindTurbineDegradationSimulator()
    
    print("--- Wind Turbine Degradation Simulator ---")
    
    csv_file_to_try = "2025.csv"
    df_input = None
    try:
        import os
        if os.path.exists(csv_file_to_try):
            print(f"Attempting to load data from '{csv_file_to_try}'...")
            df_input = pd.read_csv(csv_file_to_try)
            print(f"Successfully loaded '{csv_file_to_try}'. Shape: {df_input.shape}")
        else:
            print(f"File '{csv_file_to_try}' not found. Generating sample data instead.")
            
    except Exception as e:
        print(f"Could not load '{csv_file_to_try}' (Reason: {e}). Generating sample data.")
    
    if df_input is None or df_input.empty:
        print("Generating sample 7200-point time series for demonstration...")
        np.random.seed(42)
        n_points = 7200
        time_hours = np.arange(n_points) / 60
        wind_speeds = np.clip(8 + 3*np.sin(2*np.pi*time_hours/24) + np.random.normal(0, 2, n_points), 0, 25)
        df_input = pd.DataFrame({
            'speed': wind_speeds,
            'U': np.clip(3 + wind_speeds * 0.2 + np.random.normal(0, 0.5, n_points), 0.5, 15),
            'r': np.clip(60 + 20*np.sin(2*np.pi*time_hours/24) + np.random.normal(0, 5, n_points), 20, 95),
            'temperature': np.clip(15 + 10*np.sin(2*np.pi*time_hours/24) + np.random.normal(0,2,n_points), -5, 40)
        })
        print(f"Sample data generated. Shape: {df_input.shape}")

    if df_input is not None and not df_input.empty:
        print("\nStarting data processing...")
        df_results = simulator.process_time_series(df=df_input.copy())
        
        simulator.print_degradation_summary(df_results)
        
        print("\nGenerating analysis plots...")
        simulator.plot_degradation_analysis(df_results, save_path='degradation_analysis_final.png')
        
        output_filename = '2025updated.csv'
        try:
            df_results.to_csv(output_filename, index=False)
            print(f"\nResults saved to '{output_filename}'")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")

        print(f"\nKey columns in output file:")
        print(f"  - 'degradation':                   Cumulative degradation level (0-1)")
        print(f"  - 'maintenance_performed':         Cumulative count of maintenance events")
        # MODIFICATION 4: Add the description for the new column
        print(f"  - 'time_since_last_maintenance':   Steps elapsed since the last maintenance event")
    else:
        print("\nNo input data to process. Exiting.")