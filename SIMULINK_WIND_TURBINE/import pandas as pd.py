import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class WindTurbineDegradationSimulator:
    def __init__(self):
        # Turbine parameters from your requirements document
        self.params = {
            'blade_length': 40,  # m
            'rotor_diameter': 80,  # m (40m blade length)
            'rated_power': 1650,  # kW (from generator specs)
            'max_power': 1815,  # kW (max power at Class F)
            'mechanical_power': 1800,  # kW (from geartrain specs)
            'cut_in_speed': 3,  # m/s (typical)
            'rated_speed': 12,  # m/s (estimated)
            'cut_out_speed': 20,  # m/s (automatic stop limit)
            'recut_in_speed': 18,  # m/s (from requirements)
            'tip_speed': 61.8,  # m/s (synchronous tip speed)
            'gearbox_ratio': 84.3,  # from geartrain specs
            'generator_rpm': 1214,  # rpm at rated power
            'synchronous_rpm': 1200,  # rpm synchronous speed
            'slip': 0.0117,  # slip at rated power
            'air_density': 1.225,  # kg/m³
            'rotor_tilt': 5,  # degrees
            'poles': 6,  # generator poles
            'voltage': 600,  # V
            'frequency': 60,  # Hz
            'max_yaw_rate': 0.5  # deg/sec
        }
        
        # Degradation state (persistent variable equivalent)
        self.degradation_value = 0.0
    
    def calculate_rotor_rpm(self, wind_speed):
        """Calculate rotor RPM based on wind speed and tip speed constraint"""
        tip_speed = self.params['tip_speed']
        blade_length = self.params['blade_length']
        rated_speed = self.params['rated_speed']
        
        # At rated conditions: tip_speed = omega_rotor * radius
        rated_rotor_rpm = (tip_speed * 60) / (2 * np.pi * blade_length)  # ~29.5 RPM
        
        # Variable speed operation below rated
        if wind_speed <= rated_speed:
            return (rated_rotor_rpm * wind_speed) / rated_speed
        else:
            return rated_rotor_rpm  # Constant speed above rated
    
    def calculate_cp(self, tsr, pitch_angle=0):
        """Calculate power coefficient using improved model"""
        lambda_val = tsr
        beta = pitch_angle
        
        # Bounds check
        if lambda_val < 1 or lambda_val > 12:
            return 0
        
        # Improved Cp curve coefficients for modern 3-blade turbines
        c1, c2, c3, c4, c5, c6 = 0.5176, 116, 0.4, 5, 21, 0.0068
        
        lambda_inv = 1 / (lambda_val + 0.08 * beta) - 0.035 / (beta**3 + 1)
        cp = c1 * (c2 * lambda_inv - c3 * beta - c4) * np.exp(-c5 * lambda_inv) + c6 * lambda_val
        
        return np.clip(cp, 0, 0.48)  # Realistic maximum
    
    def calculate_system_torque(self, wind_speed):
        """Calculate system torque for given wind speed"""
        
        # Check operating limits
        if wind_speed < self.params['cut_in_speed'] or wind_speed > self.params['cut_out_speed']:
            return 0, 0  # torque, speed
        
        # Basic calculations
        rotor_radius = self.params['rotor_diameter'] / 2
        rotor_area = np.pi * rotor_radius**2
        air_density = self.params['air_density']
        gearbox_ratio = self.params['gearbox_ratio']
        
        # Rotor RPM and angular velocity
        rotor_rpm = self.calculate_rotor_rpm(wind_speed)
        omega_rotor = (rotor_rpm * 2 * np.pi) / 60  # rad/s
        
        # Generator RPM
        gen_rpm = rotor_rpm * gearbox_ratio
        
        # Tip speed ratio
        tsr = (omega_rotor * rotor_radius) / wind_speed if wind_speed > 0 else 0
        
        # Power coefficient
        cp = self.calculate_cp(tsr)
        
        # Available wind power
        wind_power = 0.5 * air_density * rotor_area * wind_speed**3  # Watts
        
        # Aerodynamic power
        aero_power = wind_power * cp  # Watts
        
        # Power limiting based on operating region
        if wind_speed <= self.params['rated_speed']:
            # Below rated - extract maximum power
            actual_power = aero_power
        else:
            # Above rated - limit to rated power
            actual_power = min(aero_power, self.params['rated_power'] * 1000)
        
        # Aerodynamic torque
        aero_torque = actual_power / omega_rotor if omega_rotor > 0 else 0  # Nm
        
        # System torque (what goes into degradation model)
        system_torque = aero_torque / 1000  # Convert to kNm for degradation model
        
        return system_torque, gen_rpm
    
    def simple_cycle_factor(self, temp_data):
        """
        Simple cycle factor calculation without rainflow counting
        Translated from your MATLAB function
        """
        # Parameters
        temp_range_threshold = 5  # Minimum temp change to count as cycle (°C)
        temp_range_power = 1.5    # Power factor for temp range
        base_cycle_damage = 0.01  # Base damage per cycle
        
        if len(temp_data) < 3:
            return 0
        
        # Find peaks and valleys (simplified cycle identification)
        peaks = []
        valleys = []
        
        for i in range(1, len(temp_data) - 1):
            if temp_data[i] > temp_data[i-1] and temp_data[i] > temp_data[i+1]:
                peaks.append(i)
            elif temp_data[i] < temp_data[i-1] and temp_data[i] < temp_data[i+1]:
                valleys.append(i)
        
        # Count cycles and their ranges
        temp_ranges = []
        
        # Simplified approach - pair adjacent peaks and valleys
        min_length = min(len(peaks), len(valleys))
        if min_length > 1:
            for i in range(min_length - 1):
                temp_range = abs(temp_data[peaks[i]] - temp_data[valleys[i]])
                if temp_range >= temp_range_threshold:
                    temp_ranges.append(temp_range)
        
        # Weight the ranges
        if temp_ranges:
            weighted_ranges = np.array(temp_ranges) ** temp_range_power
            cycle_factor = np.sum(weighted_ranges) * base_cycle_damage
        else:
            cycle_factor = 0
        
        return cycle_factor
    
    def wind_turbine_generator_degradation(self, torque, speed, vibration, temp, humidity, cycle_factor):
        """
        Wind turbine generator degradation function
        Translated from your MATLAB function
        """
        # Parameters 
        alpha1 = 1e-8   # mechanical base wear
        alpha2 = 2e-6   # vibration
        alpha3 = 1e-7   # temperature aging
        alpha4 = 5e-8   # humidity effect
        alpha5 = 3e-7   # thermal cycling
        
        # Arrhenius parameters for electrical insulation
        Ea = 0.8            # Activation energy (eV) - typical for insulation
        kb = 8.617e-5       # Boltzmann constant (eV/K)
        T_ref = 298.15      # Reference temperature (25°C in Kelvin)
        T_K = temp + 273.15 # Convert °C to Kelvin
        
        # Compute Arrhenius temperature acceleration factor
        temp_acc_factor = np.exp((Ea/kb) * (1/T_ref - 1/T_K))
        
        # Wind turbine specific components
        mech_wear = (abs(torque) * abs(speed)) ** 1.2  # Non-linear relationship
        vib_stress = vibration ** 2
        thermal_stress = temp_acc_factor
        humidity_stress = np.exp(0.05 * humidity)  # Exponential effect of humidity
        cycle_stress = alpha5 * cycle_factor  # Use pre-calculated cycle factor
        
        # Total degradation step
        dD = (alpha1 * mech_wear + 
              alpha2 * vib_stress + 
              alpha3 * thermal_stress + 
              alpha4 * humidity_stress + 
              cycle_stress)
        
        # Update degradation (clamped to 0-1)
        self.degradation_value = min(self.degradation_value + dD, 1.0)
        
        return self.degradation_value
    
    def process_time_series(self, csv_file_path=None, df=None):
        """
        Process time series data (7200 rows) and calculate degradation
        """
        if df is None:
            if csv_file_path is None:
                raise ValueError("Either csv_file_path or df must be provided")
            df = pd.read_csv(csv_file_path)
        
        print(f"Processing {len(df)} data points...")
        
        # Reset degradation state
        self.degradation_value = 0.0
        
        # Initialize results arrays
        results = {
            'system_torque_knm': [],
            'generator_speed_rpm': [],
            'degradation': [],
            'cycle_factor': []
        }
        
        # Extract data columns
        # Map your columns: U V Y = vibration rms, vibration kurtosis, crest factor
        # d = degradation, r = humidity, u v = wind components, speed, direction
        
        # Get wind speed
        if 'u' in df.columns and 'v' in df.columns:
            wind_speed = np.sqrt(df['u']**2 + df['v']**2)
        elif 'speed' in df.columns:
            wind_speed = df['speed'].values
        else:
            raise ValueError("No wind speed data found. Need either 'speed' or 'u','v' components")
        
        # Get other variables
        vibration_rms = df['U'].values if 'U' in df.columns else np.ones(len(df)) * 5
        humidity = df['r'].values if 'r' in df.columns else np.ones(len(df)) * 50
        
        # Estimate temperature (if not available, use a simple model)
        if 'temperature' in df.columns:
            temperature = df['temperature'].values
        else:
            # Simple temperature model: varies with time and has some correlation with humidity
            base_temp = 20  # Base temperature
            daily_variation = 10 * np.sin(2 * np.pi * np.arange(len(df)) / 24)  # Daily cycle
            humidity_effect = (humidity - 50) * 0.1  # Slight correlation with humidity
            temperature = base_temp + daily_variation + humidity_effect + np.random.normal(0, 2, len(df))
        
        # Calculate cycle factor for the entire temperature series
        cycle_factor_value = self.simple_cycle_factor(temperature)
        
        # Process each time step
        for i in range(len(df)):
            # Calculate system torque and speed
            torque, gen_speed = self.calculate_system_torque(wind_speed[i])
            
            # Calculate degradation for this time step
            degradation = self.wind_turbine_generator_degradation(
                torque,
                gen_speed,
                vibration_rms[i],
                temperature[i],
                humidity[i],
                cycle_factor_value
            )
            
            # Store results
            results['system_torque_knm'].append(torque)
            results['generator_speed_rpm'].append(gen_speed)
            results['degradation'].append(degradation)
            results['cycle_factor'].append(cycle_factor_value)
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(df)} points, Current degradation: {degradation:.6f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add temperature if we estimated it
        if 'temperature' not in df.columns:
            df['temperature_estimated'] = temperature
        
        # Combine with original data
        output_df = pd.concat([df, results_df], axis=1)
        
        return output_df
    
    def plot_degradation_analysis(self, df, save_path=None):
        """Create comprehensive degradation analysis plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Wind Turbine Degradation Analysis', fontsize=16)
        
        time_steps = range(len(df))
        
        # Degradation over time
        axes[0,0].plot(time_steps, df['degradation'], 'r-', linewidth=1)
        axes[0,0].set_xlabel('Time Step')
        axes[0,0].set_ylabel('Degradation Level')
        axes[0,0].set_title('Degradation Evolution')
        axes[0,0].grid(True, alpha=0.3)
        
        # Wind Speed vs System Torque
        wind_speed = df['speed'] if 'speed' in df.columns else np.sqrt(df['u']**2 + df['v']**2)
        axes[0,1].scatter(wind_speed, df['system_torque_knm'], alpha=0.3, s=1)
        axes[0,1].set_xlabel('Wind Speed (m/s)')
        axes[0,1].set_ylabel('System Torque (kNm)')
        axes[0,1].set_title('Wind Speed vs System Torque')
        axes[0,1].grid(True, alpha=0.3)
        
        # Temperature profile
        temp_col = 'temperature' if 'temperature' in df.columns else 'temperature_estimated'
        axes[1,0].plot(time_steps, df[temp_col], 'b-', alpha=0.7, linewidth=0.5)
        axes[1,0].set_xlabel('Time Step')
        axes[1,0].set_ylabel('Temperature (°C)')
        axes[1,0].set_title('Temperature Profile')
        axes[1,0].grid(True, alpha=0.3)
        
        # Vibration vs Degradation Rate
        degradation_rate = np.diff(df['degradation'].values)
        degradation_rate = np.append(degradation_rate, degradation_rate[-1])  # Pad to same length
        axes[1,1].scatter(df['U'], degradation_rate, alpha=0.3, s=1)
        axes[1,1].set_xlabel('Vibration RMS')
        axes[1,1].set_ylabel('Degradation Rate')
        axes[1,1].set_title('Vibration vs Degradation Rate')
        axes[1,1].grid(True, alpha=0.3)
        
        # Humidity effects
        axes[2,0].scatter(df['r'], degradation_rate, alpha=0.3, s=1, c='green')
        axes[2,0].set_xlabel('Humidity (%)')
        axes[2,0].set_ylabel('Degradation Rate')
        axes[2,0].set_title('Humidity vs Degradation Rate')
        axes[2,0].grid(True, alpha=0.3)
        
        # Generator Speed Distribution
        axes[2,1].hist(df['generator_speed_rpm'], bins=50, alpha=0.7, edgecolor='black')
        axes[2,1].set_xlabel('Generator Speed (RPM)')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].set_title('Generator Speed Distribution')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_degradation_summary(self, df):
        """Print summary statistics of the degradation analysis"""
        print("\n" + "="*50)
        print("DEGRADATION ANALYSIS SUMMARY")
        print("="*50)
        
        final_degradation = df['degradation'].iloc[-1]
        max_torque = df['system_torque_knm'].max()
        avg_torque = df['system_torque_knm'].mean()
        max_gen_speed = df['generator_speed_rpm'].max()
        avg_gen_speed = df['generator_speed_rpm'].mean()
        
        print(f"Dataset size: {len(df)} time steps")
        print(f"Final degradation level: {final_degradation:.6f} ({final_degradation*100:.4f}%)")
        print(f"Cycle factor: {df['cycle_factor'].iloc[0]:.6f}")
        
        print(f"\nTorque Statistics:")
        print(f"  Maximum system torque: {max_torque:.2f} kNm")
        print(f"  Average system torque: {avg_torque:.2f} kNm")
        
        print(f"\nGenerator Speed Statistics:")
        print(f"  Maximum generator speed: {max_gen_speed:.1f} RPM")
        print(f"  Average generator speed: {avg_gen_speed:.1f} RPM")
        
        print(f"\nEnvironmental Conditions:")
        temp_col = 'temperature' if 'temperature' in df.columns else 'temperature_estimated'
        print(f"  Temperature range: {df[temp_col].min():.1f}°C to {df[temp_col].max():.1f}°C")
        print(f"  Humidity range: {df['r'].min():.1f}% to {df['r'].max():.1f}%")
        print(f"  Vibration RMS range: {df['U'].min():.2f} to {df['U'].max():.2f}")
        
        # Degradation rate statistics
        degradation_rate = np.diff(df['degradation'].values)
        print(f"\nDegradation Rate Statistics:")
        print(f"  Maximum degradation rate: {degradation_rate.max():.2e}")
        print(f"  Average degradation rate: {degradation_rate.mean():.2e}")
        
        print("="*50)

# Example usage
if __name__ == "__main__":
    # Initialize simulator
    simulator = WindTurbineDegradationSimulator()
    
    print("Wind Turbine Degradation Simulator")
    print("Designed for 7200-row time series data")
    print("="*50)
    
    # For demonstration with your actual CSV file, uncomment this:
    # df_results = simulator.process_time_series('your_wind_data.csv')
    
    # Generate sample data for demonstration (7200 points)
    print("Generating sample 7200-point time series for demonstration...")
    np.random.seed(42)
    
    n_points = 7200
    # Create realistic time series data
    time_hours = np.arange(n_points) / 60  # Assuming 1-minute intervals
    
    # Realistic wind speed with diurnal variation
    base_wind = 8 + 3 * np.sin(2 * np.pi * time_hours / 24)  # Daily cycle
    wind_noise = np.random.normal(0, 2, n_points)
    wind_speeds = np.clip(base_wind + wind_noise, 2, 25)
    
    # Correlated vibration data
    vibration_rms = 3 + wind_speeds * 0.2 + np.random.normal(0, 0.5, n_points)
    vibration_rms = np.clip(vibration_rms, 1, 15)
    
    # Humidity with daily variation
    humidity = 60 + 20 * np.sin(2 * np.pi * time_hours / 24 + np.pi/4) + np.random.normal(0, 5, n_points)
    humidity = np.clip(humidity, 20, 90)
    
    # Create sample DataFrame
    sample_data = pd.DataFrame({
        'speed': wind_speeds,
        'U': vibration_rms,  # Vibration RMS
        'V': np.random.gamma(2, 2, n_points) + 3,  # Vibration Kurtosis
        'Y': np.random.exponential(1, n_points) + 2.5,  # Crest Factor
        'r': humidity,  # Humidity
        'direction': np.random.uniform(0, 360, n_points),
        'temperature': 20 + 15 * np.sin(2 * np.pi * time_hours / 24) + np.random.normal(0, 3, n_points)
    })
    
    # Process the time series
    print("Processing time series data...")
    df_results = simulator.process_time_series(csv_file_path="combined_signal_and_wind_data.csv")
    
    # Print summary
    simulator.print_degradation_summary(df_results)
    
    # Create plots
    print("\nGenerating analysis plots...")
    simulator.plot_degradation_analysis(df_results)
    
    # Save results
    output_filename = 'wind_turbine_degradation_results.csv'
    df_results.to_csv(output_filename, index=False)
    print(f"\nResults saved to '{output_filename}'")
    
    print(f"\nKey columns for further analysis:")
    print(f"  - 'system_torque_knm': System torque for each time step")
    print(f"  - 'degradation': Cumulative degradation level (0-1)")
    print(f"  - 'generator_speed_rpm': Generator speed")
    print(f"  - 'cycle_factor': Thermal cycling factor")
    
    print(f"\nTo use with your actual CSV file:")
    print(f"  df_results = simulator.process_time_series('your_file.csv')")