

# 🌬️ Wind Turbine Generator Predictive Maintenance Model

## 🔧 Development Progress & Observed Failures

During the development of the wind turbine predictive maintenance model, we encountered multiple issues and behaviors that helped us refine the system step by step. The following scopes visualize the simulation outputs at different stages, highlighting failures and diagnostic insights.

### 📊 Simulink Scopes Output

![Simulation Scopes Overview](scopes.png)

---

### 🌀 Rotor Speed Panel

- **Observation:** Rotor speed fluctuated slightly around a constant nominal value (~1400 rpm).
- **Interpretation:** Speed feedback and actuator control loops were stable, but minor noise was present.
- **Fix Applied:** Fine-tuned the actuator gain and ensured correct scaling from rad/s to pu.

---

### ⚡ Electromagnetic Torque (Te)

- **Observation:** Torque signal exhibited abrupt drops and rises during some simulation phases.
- **Issue:** Caused by mismatch between torque actuator dynamics and generator response.
- **Fix:** We introduced damping via the `Actuator Dynamics` block to smooth out the torque input.

---

### 🛠️ Degradation Growth

- **Observation:** Steady increase in degradation values over time.
- **Success:** This validated the effectiveness of our degradation model based on torque, vibration, and environmental data.
- **Usage:** These values feed into the predictive maintenance model to estimate component health.

---

### 🔁 Gearbox Output (Unexpected Behavior)

- **Observation:** Large flatline at zero — no expected gear output.
- **Issue:** Misconfigured signal connection between torque input and gearbox subsystem.
- **Action Taken:** Rewired the gearbox output and rerouted torque after confirming dynamic consistency.

---

### 📈 Vibration RMS & Statistical Metrics

- **Observation:** Vibration RMS and kurtosis showed sporadic spikes.
- **Insight:** These spikes were simulated fault injections or reflect instability from misaligned rotor/generator coupling.
- **Response:** Implemented `generator_vibration_sensor` to capture these characteristics and improve feature extraction for fault classification.

---

### ✅ Final Note

Each anomaly helped us uncover a modeling or signal routing flaw. By resolving them, the model evolved into a more accurate and reliable predictive maintenance simulator.

> ⚠️ *Note:* All scope signals shown above are time-based samples with simulation time T=100000.

## 📈 Potential Improvements

* Add fault injection models for gearbox and blade damage
* Integrate ML models for remaining useful life (RUL) estimation
* Improve cycle factor calculation based on actual turbine usage data
* Include real weather input from `.mat` or `.csv` datasets

---

## 📜 License & Credits

* Model © 2009–2020 The MathWorks, Inc.
* Extended and modified by \ Hachim & Abdellah
* For academic use only.


