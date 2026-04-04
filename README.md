#  Global Spatio-Temporal Early Warning Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI%2FUX-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/GenAI-Gemini%202.5%20Flash-8E75B2.svg?style=for-the-badge)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20Explainability-success.svg?style=for-the-badge)

> **Machine learning is useless if the people making life-or-death decisions can't understand it.** > 
> The Epidemic Early Warning Engine is a production-grade Decision-Support Dashboard. It ingests live, multi-modal telemetry from global APIs and satellites, processes the data through a localized Machine Learning model, and renders the intelligence in a highly interactive, human-centric interface. 

---

##  Enterprise-Grade Features

* ** GenAI Situation Reports (SITREPs):** Integrates Google's **Gemini 2.5 Flash** LLM to dynamically generate strict, military-style executive summaries based on live changing climate and capacity telemetry.
* ** Explainable AI (XAI) via SHAP:** Solves the "Black Box" problem. A live Random Forest surrogate model calculates localized feature attribution, rendering real-time **SHAP waterfall plots** so stakeholders know *exactly* why the AI predicted a specific threat level.
* ** Automated Executive Briefings:** Features a one-click **PDF Generation Pipeline** (`fpdf2`) that instantly compiles live satellite metrics, AI predictions, and capacity constraints into a downloadable executive intelligence brief.
* ** Live Multi-Modal Telemetry:** * **Meteorological:** Live weather observations via Open-Meteo.
  * **Syndromic Surveillance:** Real-time search behavior tracking via Google Trends.
  * **Space Telemetry:** Live Normalized Difference Vegetation Index (NDVI) pulled directly from NASA's MODIS satellite via Google Earth Engine.
* ** Dynamic World Generation:** Utilizes a custom Python automation script to programmatically build a spatial database of 195+ countries and their top 5 highest-risk urban nodes.

---

##  Architecture & Domain Adaptation

While currently configured to track epidemiological vectors, the underlying Spatio-Temporal architecture and GenAI reporting pipeline are entirely domain-agnostic. 

By swapping out meteorological data for socio-political event streams, and replacing disease tracking with human sentiment analysis, this exact pipeline is designed to scale into **ARES: A conflict escalation early-warning system** for military and geopolitical strategy.

---

##  Quick Start Guide (Local Deployment)

**1. Clone the repository & Install Dependencies**
```bash
git clone [https://github.com/yourusername/spatio-temporal-forecasting.git](https://github.com/yourusername/spatio-temporal-forecasting.git)
cd spatio-temporal-forecasting
pip install -r requirements.txt
```
**2. Generate the Global Database**
Build the 195+ country spatial database dynamically.

```Bash
python build_world.py
```

**3. Configure API Keys & Authentication**

Authenticate with Google Earth Engine (Required for live NASA satellite data):

```Bash
earthengine authenticate
```
Create a .streamlit/secrets.toml file in the root directory and add your Google Gemini API Key for automated SITREPs:
```
GEMINI_API_KEY = "your_api_key_here"
```

**4. Launch the Command Center**

```Bash
streamlit run app/dashboard.py
```

---

##  System File Architecture
```
Plaintext
├── .streamlit/                 # Hidden configuration (API Keys)
├── app/                        
│   └── dashboard.py            # Core UI, SHAP logic, and GenAI engine
├── data/                       
│   └── global_nodes.json       # Dynamically generated 195+ country DB
├── src/                        
│   └── live_data.py            # Backend NASA/Open-Meteo API connections
├── build_world.py              # Global node generation script
├── README.md                   # Executive documentation
└── requirements.txt            # Dependency lockfile
```

---

👨‍💻 Let's Connect
I am a 3rd-year Information Science and Engineering student passionate about the intersection of Data Science, Machine Learning, and UI/UX design. I build intuitive interfaces that help human stakeholders understand complex mathematical models.

LinkedIn: www.linkedin.com/in/tanya-deep/
