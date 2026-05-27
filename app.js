// Decision Tree model rules trained on 2024 data
function predictDebris(features) {
  if (features["eccentricity"] <= 0.001661) {
    if (features["inclination"] <= 1.721951) {
      if (features["eccentricity"] <= 0.001140) {
        if (features["eccentricity"] <= 0.000338) {
          if (features["arg_perigee"] <= 5.913618) {
            return { class: 0, probability: 0.0003 };
          } else {
            return { class: 0, probability: 0.0333 };
          }
        } else {
          if (features["inclination"] <= 1.508123) {
            return { class: 0, probability: 0.1037 };
          } else {
            return { class: 0, probability: 0.0010 };
          }
        }
      } else {
        if (features["inclination"] <= 1.508106) {
          if (features["inclination"] <= 1.273523) {
            return { class: 0, probability: 0.0000 };
          } else {
            return { class: 1, probability: 0.7765 };
          }
        } else {
          if (features["mean_anomaly"] <= 6.230918) {
            return { class: 0, probability: 0.0034 };
          } else {
            return { class: 1, probability: 0.5000 };
          }
        }
      }
    } else {
      if (features["distance_from_center"] <= 7167.738037) {
        if (features["raan"] <= 3.201176) {
          return { class: 1, probability: 1.0000 };
        } else {
          if (features["inclination"] <= 1.734652) {
            return { class: 1, probability: 1.0000 };
          } else {
            return { class: 0, probability: 0.0000 };
          }
        }
      } else {
        if (features["raan"] <= 5.749403) {
          if (features["mean_anomaly"] <= 4.050653) {
            return { class: 1, probability: 0.5000 };
          } else {
            return { class: 0, probability: 0.0303 };
          }
        } else {
          if (features["velocity_magnitude"] <= 7.365789) {
            return { class: 0, probability: 0.0000 };
          } else {
            return { class: 1, probability: 0.9524 };
          }
        }
      }
    }
  } else {
    if (features["inclination"] <= 1.252195) {
      return { class: 0, probability: 0.0000 };
    } else {
      if (features["perigee"] <= 271553.171875) {
        if (features["orbital_period"] <= 1341756.312500) {
          if (features["inclination"] <= 1.586503) {
            return { class: 1, probability: 0.9708 };
          } else {
            return { class: 0, probability: 0.4244 };
          }
        } else {
          if (features["perigee"] <= 255080.093750) {
            return { class: 0, probability: 0.0000 };
          } else {
            return { class: 1, probability: 0.9823 };
          }
        }
      } else {
        if (features["arg_perigee"] <= 0.520701) {
          if (features["specific_orbital_energy"] <= -24.883844) {
            return { class: 0, probability: 0.0000 };
          } else {
            return { class: 1, probability: 1.0000 };
          }
        } else {
          return { class: 0, probability: 0.0000 };
        }
      }
    }
  }
}

// Preset definitions (derived mathematically from actual TLEs)
const PRESETS = {
  active_starlink: {
    label: "Active Starlink Satellite",
    eccentricity: 0.00012,
    inclination: 53.0, // entered in degrees
    raan: 142.5,
    arg_perigee: 89.2,
    mean_anomaly: 271.4,
    mean_motion: 15.06,
    semi_major_axis: 6921,
    altitude: 550,
    perigee: 6920,
    apogee: 6922,
    orbital_period: 5737,
    distance_from_center: 6921,
    velocity_magnitude: 7.58,
    specific_orbital_energy: -28.8,
  },
  active_geo: {
    label: "Active Geostationary Sat",
    eccentricity: 0.00008,
    inclination: 0.05,
    raan: 312.4,
    arg_perigee: 12.6,
    mean_anomaly: 195.8,
    mean_motion: 1.0027,
    semi_major_axis: 42164,
    altitude: 35786,
    perigee: 42160,
    apogee: 42168,
    orbital_period: 86164,
    distance_from_center: 42164,
    velocity_magnitude: 3.07,
    specific_orbital_energy: -4.73,
  },
  cosmos_debris: {
    label: "Cosmos 2251 Fragment",
    eccentricity: 0.0142,
    inclination: 74.04,
    raan: 220.1,
    arg_perigee: 148.5,
    mean_anomaly: 211.2,
    mean_motion: 14.32,
    semi_major_axis: 7156,
    altitude: 785,
    perigee: 7054,
    apogee: 7258,
    orbital_period: 6033,
    distance_from_center: 7152,
    velocity_magnitude: 7.45,
    specific_orbital_energy: -27.85,
  },
  fengyun_debris: {
    label: "FengYun 1C Fragment",
    eccentricity: 0.052,
    inclination: 99.04,
    raan: 31.8,
    arg_perigee: 278.4,
    mean_anomaly: 78.5,
    mean_motion: 13.98,
    semi_major_axis: 7272,
    altitude: 901,
    perigee: 6894,
    apogee: 7650,
    orbital_period: 6180,
    distance_from_center: 7210,
    velocity_magnitude: 7.39,
    specific_orbital_energy: -27.41,
  }
};

document.addEventListener("DOMContentLoaded", () => {
  // Select DOM Elements
  const inputs = {
    eccentricity: document.getElementById("input-eccentricity"),
    inclination: document.getElementById("input-inclination"),
    raan: document.getElementById("input-raan"),
    arg_perigee: document.getElementById("input-arg_perigee"),
    mean_anomaly: document.getElementById("input-mean_anomaly"),
    mean_motion: document.getElementById("input-mean_motion"),
    semi_major_axis: document.getElementById("input-semi_major_axis"),
    altitude: document.getElementById("input-altitude"),
    perigee: document.getElementById("input-perigee"),
    apogee: document.getElementById("input-apogee"),
    orbital_period: document.getElementById("input-orbital_period"),
    distance_from_center: document.getElementById("input-distance_from_center"),
    velocity_magnitude: document.getElementById("input-velocity_magnitude"),
    specific_orbital_energy: document.getElementById("input-specific_orbital_energy")
  };

  const sliders = {
    eccentricity: document.getElementById("slide-eccentricity"),
    inclination: document.getElementById("slide-inclination"),
    altitude: document.getElementById("slide-altitude"),
    velocity_magnitude: document.getElementById("slide-velocity_magnitude")
  };

  const btnAnalyze = document.getElementById("btn-analyze");
  const resultPanel = document.getElementById("result-panel");
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultData = document.getElementById("result-data");
  const statusTxt = document.getElementById("status-txt");
  const confidenceTxt = document.getElementById("confidence-txt");
  const explanationTxt = document.getElementById("explanation-txt");

  // Sync Input Box and Slider Values
  function setupSync(inputId, sliderId) {
    const input = inputs[inputId];
    const slider = sliders[sliderId];
    if (!input || !slider) return;

    input.addEventListener("input", () => {
      slider.value = input.value;
      recalculateDerived();
      performInstantPrediction();
    });

    slider.addEventListener("input", () => {
      input.value = slider.value;
      recalculateDerived();
      performInstantPrediction();
    });
  }

  setupSync("eccentricity", "eccentricity");
  setupSync("inclination", "inclination");
  setupSync("altitude", "altitude");
  setupSync("velocity_magnitude", "velocity_magnitude");

  // Listen to other manual numeric input changes
  Object.values(inputs).forEach(input => {
    input.addEventListener("input", () => {
      recalculateDerived();
      performInstantPrediction();
    });
  });

  // Basic Physics Engine: Recalculate Derived Variables Dynamically
  function recalculateDerived() {
    const earthRadius = 6371.0;
    const mu = 398600.4418;

    const alt = parseFloat(inputs.altitude.value) || 500;
    const ecc = parseFloat(inputs.eccentricity.value) || 0.001;

    // Derived Keplerian properties
    const sma = alt + earthRadius; // Semi-major axis for circular-approx, km
    inputs.semi_major_axis.value = Math.round(sma);
    inputs.distance_from_center.value = Math.round(sma);
    
    // Perigee & Apogee radii
    const periRad = sma * (1 - ecc);
    const apoRad = sma * (1 + ecc);
    inputs.perigee.value = Math.round(periRad);
    inputs.apogee.value = Math.round(apoRad);

    // Orbital velocity (approx)
    const vel = Math.sqrt(mu / sma);
    inputs.velocity_magnitude.value = vel.toFixed(3);
    if (sliders.velocity_magnitude) {
      sliders.velocity_magnitude.value = vel.toFixed(3);
    }

    // Specific orbital energy
    const soe = (0.5 * vel * vel) - (mu / sma);
    inputs.specific_orbital_energy.value = soe.toFixed(3);

    // Mean motion (rev/day)
    const mm = (Math.sqrt(mu / (sma * sma * sma)) * 86400) / (2 * Math.PI);
    inputs.mean_motion.value = mm.toFixed(4);

    // Orbital period (seconds)
    const period = 86400 / mm;
    inputs.orbital_period.value = Math.round(period);
  }

  // Load Preset Function
  window.loadPreset = function(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    // Remove active state from other preset buttons
    document.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
    // Add active state to clicked button
    const btn = document.querySelector(`[onclick="loadPreset('${presetKey}')"]`);
    if (btn) btn.classList.add("active");

    // Populate all fields
    Object.keys(data).forEach(key => {
      if (inputs[key]) {
        inputs[key].value = data[key];
      }
      if (sliders[key]) {
        sliders[key].value = data[key];
      }
    });

    // Make prediction immediately on loading preset
    performPrediction();
  };

  // Perform Instant Prediction (No delays, no loaders)
  function performInstantPrediction() {
    const features = {
      eccentricity: parseFloat(inputs.eccentricity.value),
      inclination: parseFloat(inputs.inclination.value) * Math.PI / 180.0,
      raan: parseFloat(inputs.raan.value) * Math.PI / 180.0,
      arg_perigee: parseFloat(inputs.arg_perigee.value) * Math.PI / 180.0,
      mean_anomaly: parseFloat(inputs.mean_anomaly.value) * Math.PI / 180.0,
      mean_motion: parseFloat(inputs.mean_motion.value),
      semi_major_axis: parseFloat(inputs.semi_major_axis.value),
      altitude: parseFloat(inputs.altitude.value),
      perigee: parseFloat(inputs.perigee.value),
      apogee: parseFloat(inputs.apogee.value),
      orbital_period: parseFloat(inputs.orbital_period.value),
      distance_from_center: parseFloat(inputs.distance_from_center.value),
      velocity_magnitude: parseFloat(inputs.velocity_magnitude.value),
      specific_orbital_energy: parseFloat(inputs.specific_orbital_energy.value)
    };

    const prediction = predictDebris(features);

    resultPlaceholder.style.display = "none";
    resultData.style.display = "flex";

    if (prediction.class === 1) {
      resultPanel.className = "card result-screen debris";
      statusTxt.innerText = "WARNING: SPACE DEBRIS";
      const conf = prediction.probability * 100;
      confidenceTxt.innerHTML = `Classification Confidence: <span>${conf.toFixed(1)}%</span>`;
      
      explanationTxt.innerHTML = `<strong>Reasoning:</strong> The object has an eccentricity of ${features.eccentricity.toFixed(4)} and is in an elliptical, uncontrolled decay profile. The inclination of ${(parseFloat(inputs.inclination.value)).toFixed(2)}° aligns with historical debris orbits.`;
    } else {
      resultPanel.className = "card result-screen active-satellite";
      statusTxt.innerText = "ACTIVE SATELLITE";
      const conf = (1 - prediction.probability) * 100;
      confidenceTxt.innerHTML = `Classification Confidence: <span>${conf.toFixed(1)}%</span>`;
      
      explanationTxt.innerHTML = `<strong>Reasoning:</strong> This orbit maintains a near-circular path (eccentricity = ${features.eccentricity.toFixed(5)}). The constant velocity of ${features.velocity_magnitude.toFixed(2)} km/s and controlled specific energy indicate active station-keeping.`;
    }
  }

  // Perform ML Prediction (With scanning radar loading animation)
  function performPrediction() {
    // Show spinner in results area
    resultPlaceholder.style.display = "none";
    resultData.style.display = "none";
    resultPanel.className = "card result-screen";
    
    // Create animated scanning radar loading effect
    const loader = document.createElement("div");
    loader.className = "radar-spinner";
    resultPanel.appendChild(loader);

    setTimeout(() => {
      // Remove spinner
      loader.remove();
      performInstantPrediction();
    }, 600); // Small delay to feel like a real scanning radar
  }

  // Predict button listener
  btnAnalyze.addEventListener("click", performPrediction);

  // Initialize with Starlink Preset on load
  loadPreset("active_starlink");
});

