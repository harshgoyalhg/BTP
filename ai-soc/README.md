# 🛡️ Next-Gen AI Security Operations Center (AI-SOC)

Welcome to the AI-SOC project! If you are presenting this to your professor, this document will explain **exactly what we built**, **how it works**, and **why we made certain choices**, using simple and clear language.

---

## 🌟 The Big Picture: What Did We Build?

Think of a traditional computer network like a massive office building. A traditional security system (like a firewall) is just a list of rules: *"If someone wears a red shirt, don't let them in."* But hackers are smart—they just put on a blue shirt to sneak past the rules.

Instead of a rule-based system, **we built an Artificial Intelligence (AI) detective.** 

We took millions of records of real network traffic—both normal traffic and hacking attempts—and trained an AI to recognize the *behavior* of an attack. Now, even if the hacker changes their "shirt," our AI can spot them based on how they act. Furthermore, if our AI catches an attacker, it will actually **explain exactly why** it thinks the person is a hacker.

### 🏗️ The 6 Parts of Our Architecture (Explained Simply)

Our project is divided into 6 pieces that talk to each other:

1. **The Security Camera (Traffic Simulator):** In a real SOC, you have sensors listening to live network cables. Because we don't have a live enterprise network to test on, we built a `traffic_simulator.py`. This script reads a massive dataset of past attacks and "replays" them, tricking our system into thinking it is under a live cyberattack.
2. **The Detective's Brain (XGBoost AI Model):** This is the core Machine Learning model. It looks at the incoming network data and instantly decides: *Is this normal? Is this a DDoS attack? Is this a Port Scan?*
3. **The Evidence Board (SHAP Explainability):** Professors often ask, *"How do you know the AI isn't just guessing?"* We solved this using **Explainable AI (SHAP)**. When the AI detects a threat, SHAP calculates exactly which network features (like packet size or connection length) made the AI suspicious. It forces the AI to "show its math."
4. **The Police Record (PostgreSQL Database):** Every time an attack is detected, we save it into a fast SQL database so we have a permanent record of the incident.
5. **The Dispatcher (FastAPI Backend):** This is the middleman. It takes the data from the Simulator, hands it to the AI Brain, takes the AI's prediction, saves it to the Database, and sends the final alert to the Dashboard.
6. **The Command Center (React Dashboard):** A beautiful, dark-themed webpage where a security analyst can sit and watch the attacks get blocked in real-time.

---

## 🚀 The Story of Our Workflow: Step-by-Step

Here is the exact story of how we built this from scratch, phase by phase. You can explain this timeline to your professor:

### Phase 1: Setting up the Workshop
First, we organized our codebase into folders (`ml`, `backend`, `frontend`, `database`, `agents`). We made sure we used separate, clean Python environments so that the heavy Machine Learning math libraries wouldn't interfere with the lightweight Web Server libraries. 

### Phase 2: Teaching the AI to Read (Data Preprocessing)
* **What we did:** We downloaded the **CICIDS2017 Dataset**, which contains millions of rows of real network traffic recorded by a Canadian university. 
* **The challenge:** Raw data is messy. It contains missing values (`NaN`) and text labels that computers can't understand. Furthermore, some network features have huge numbers (like bytes transferred), while others have small numbers (like flags).
* **The fix:** We wrote `ml/preprocessing.py` to clean the data. We removed IP addresses (so the AI doesn't just memorize hacker IPs), and we used a `StandardScaler` to shrink all the numbers down to a similar scale so the AI could learn fairly. 

### Phase 3: Training the AI Detective (XGBoost)
* **What we did:** We wrote `ml/train.py` to train our Machine Learning model.
* **Why XGBoost?:** We could have used Deep Learning (Neural Networks), but for **tabular data** (data in rows and columns, like our network flows), a model called **XGBoost (Extreme Gradient Boosting)** is actually much faster and more accurate. In cybersecurity, speed is everything. We trained the model to recognize 4 things: Normal Traffic, DDoS attacks, Port Scans, and Botnets. We saved this trained brain as a file called `model.pkl`.

### Phase 4: Building the AI Prediction Engine 
* **What we did:** We created `ml/predict.py`. 
* **How it works:** When a live network packet comes in, this script quickly scales the numbers, asks the `model.pkl` for a prediction, and returns the result along with a "Confidence Score" (e.g., *I am 99.8% sure this is a DDoS attack*). We also hooked up the SHAP logic here to generate the explanations.

### Phase 5: Building the Database Memory
* **What we did:** We used **Docker** to spin up a **PostgreSQL** database. 
* **Why Docker?:** Docker ensures that our database runs perfectly on any computer without having to manually install SQL servers. We wrote `backend/models.py` to define the tables where alerts would be saved. 

### Phase 6: Creating the API Messenger (FastAPI)
* **What we did:** We built `backend/main.py` using **FastAPI**. 
* **Why FastAPI?:** It is one of the fastest Python web frameworks available. We created a URL endpoint (`/predict`). Now, anyone can send network data to that URL, and FastAPI will automatically pass it to the ML model and save the result to the database. We also made a `/statistics` endpoint to count how many attacks have happened.

### Phase 7: Faking the Cyberattack (Traffic Simulator)
* **What we did:** We couldn't ask a real hacker to attack us for the demo, so we wrote `agents/traffic_simulator.py`. 
* **How it works:** It reads the CSV files from Phase 2, picks a random attack, packages it up, and fires it at the FastAPI backend every second. It has a `--type mix` mode that throws completely random, chaotic attacks at the AI to see if it can keep up.

### Phase 8: Building the Visual Dashboard
* **What we did:** We built a modern web application using **React** and **TailwindCSS**. 
* **How it works:** The dashboard constantly asks the FastAPI backend, *"Hey, do you have any new alerts?"* As the simulator fires attacks and the AI catches them, the dashboard dynamically updates its numbers and graphs. We made it look like a high-tech cybersecurity command center.

---

## 🎓 Why this BTP is Impressive (Talking Points for your Professor)

If your professor asks what makes this project special, highlight these three points:

1. **It is an End-to-End System:** We didn't just train an ML model in a Jupyter Notebook and stop there. We built the *entire engineering pipeline*—from the data preprocessing to the REST API, the SQL database, and the live React interface.
2. **Explainable AI (XAI):** "Black box" AI is a major problem in cybersecurity because analysts don't trust models that can't explain themselves. By integrating SHAP, we solved the trust issue. Our system is completely transparent.
3. **Real-Time Capabilities:** The architecture is decoupled. The React frontend, FastAPI backend, and Simulator all run as separate, asynchronous processes, mimicking exactly how a scalable, microservice-based architecture works in the real world.

---

## 🏃‍♂️ How to Run the Project for the Demo

To show this off to your professor, you need to open 4 terminal windows and run these commands in order:

1. **Start the Database:**
   ```bash
   docker compose up -d
   ```
2. **Start the Backend API:**
   ```bash
   python -m uvicorn backend.main:app --port 8000 --reload
   ```
3. **Start the Frontend Dashboard:**
   ```bash
   cd frontend
   npm run dev
   ```
4. **Launch the Attack Simulator (The fun part!):**
   ```bash
   python agents/traffic_simulator.py --type mix
   ```
   *(Wait a few seconds for it to load, then watch the React dashboard light up with attacks!)*
