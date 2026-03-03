# 🦠 Global Epidemic Early Warning Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI%2FUX-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Data](https://img.shields.io/badge/Data-Live%20APIs-success.svg?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3D%20Analytics-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

> **What if we could predict a localized outbreak before the first patient ever walked into a hospital?** Welcome to the Epidemic Early Warning Engine. This is a production-grade Decision-Support Dashboard that moves beyond static historical datasets. It ingests live, multi-modal telemetry from global APIs and satellites, processes the data through simulated Spatio-Temporal Graph logic, and renders the intelligence in a highly interactive, human-centric interface.

The core philosophy here? **Machine learning is useless if the people making life-or-death decisions can't understand it.** This project perfectly bridges the gap between rigorous data science and intuitive UI/UX design.

---

## ✨ The "Cool Factor" (Key Features)

* **🛰️ Houston, We Have Satellite Data:** No static CSVs here. The engine pulls real-time multi-modal data:
  * **Meteorological:** Live, up-to-the-minute weather observations via Open-Meteo.
  * **Syndromic Surveillance:** Real-time search behavior tracking via Google Trends (because people Google their symptoms before going to the doctor).
  * **Space Telemetry:** Live Normalized Difference Vegetation Index (NDVI) pulled directly from NASA's MODIS satellite via Google Earth Engine to detect high-risk vector breeding grounds.
* **🎛️ The "What-If" Simulator:** A sandbox for decision-makers. Adjust temperature anomalies or simulate government vector-control interventions to instantly see how the trajectory of the outbreak changes.
* **🧊 3D Spatio-Temporal Analytics:** Why settle for flat charts? Explore multi-dimensional environmental risk clusters and 3D surface plots of disease spread across top urban nodes (e.g., watching a simulated surge move from Mumbai to Delhi).
* **🏥 Executive Threat Gauges:** Automatically calculates localized hospital ICU strain, translating raw math into immediate "Action vs. Inaction" insights.

---

## 🏗️ Architecture & Extensibility (Beyond Epidemiology)

While currently configured to track biological vectors, the underlying mathematical architecture is entirely domain-agnostic. 

By swapping out meteorological data for socio-political event streams, and replacing disease tracking with human sentiment analysis, this exact Spatio-Temporal architecture scales perfectly to power **conflict escalation early-warning systems**. It is built to be a generalized engine for tracking localized anomalies over time, whether biological or geopolitical.

---

## 🚀 Quick Start Guide

Want to run this command center locally? It takes less than two minutes.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/epidemic-warning-engine.git](https://github.com/yourusername/epidemic-warning-engine.git)
cd epidemic-warning-engine
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Authenticate Google Earth Engine (Required to talk to NASA)**

```bash
earthengine authenticate
```
(A browser window will open to grant your local machine access to Google's satellite data).

**4. Launch the Engine**

```bash
streamlit run app/dashboard.py
```
---

## 🛠️ The Tech Stack
Frontend / UI/UX: Streamlit

Data Visualization: Plotly Graph Objects & Plotly Express (3D Graphing, Mapbox, Radar, Waterfall)

Backend Data Engineering: Pandas, Numpy, Requests

Live APIs: PyTrends, Open-Meteo, Google Earth Engine API

This reads incredibly well. It shows you understand the technical side (APIs, Spatio-Temporal logic) but also care deeply about the user experience.

Are you ready to initialize your Git repository and push this masterpiece up to your GitHub

---

## 🔮 Future Roadmap (The Next Evolution)
This engine is currently a V1 prototype. The architectural roadmap includes:
* **Deep Learning Integration:** Replacing the current mathematical baseline with a full **Spatio-Temporal Graph Attention Network (STGAT)** or a **Temporal Fusion Transformer (TFT)** to learn complex, non-linear relationships between climate and outbreaks.
* **Domain Adaptation:** Abstracting the core UI/UX and data pipeline to handle socio-political event streams, effectively transforming this epidemiological tool into a conflict escalation early-warning system.
* **Automated Reporting:** Adding a cron-job pipeline to automatically generate and email localized PDF executive briefs when the model detects an anomaly that breaches hospital capacity constraints.

---

## 🧠 Technical Challenges Overcome
* **API Rate Limiting & UI/UX Resilience:** Live APIs fail. To prevent dashboard crashes during a demo, I engineered a robust `try/except` fallback pipeline. If the Google Earth Engine or Open-Meteo APIs timeout, the UI silently falls back to moderate baseline metrics rather than throwing fatal exception errors.
* **Dashboard State Management:** Re-rendering 3D Plotly maps every time a user moves a slider causes severe UI lag. By aggressively utilizing Streamlit's `@st.cache_data` decorator, API telemetry is cached for 10 minutes, making the "What-If" simulators lightning-fast.

---

## 📜 Acknowledgments & Data Sources
This project would not be possible without the incredible open-source data provided by:
* [Open-Meteo](https://open-meteo.com/) for real-time meteorological observation feeds.
* [NASA Earth Data & Google Earth Engine](https://earthengine.google.com/) for the MODIS satellite telemetry.
* [PyTrends](https://pypi.org/project/pytrends/) for syndromic surveillance access.

---

## 👨‍💻 Let's Connect
I am a 3rd-year Information Science and Engineering student passionate about the intersection of Data Science, Machine Learning, and UI/UX design. 

* **LinkedIn:** https://www.linkedin.com/in/tanya-deep/

> *"The best models are the ones people can actually use."*