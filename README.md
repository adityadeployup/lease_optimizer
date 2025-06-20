# 🏢 Rental Lease Optimization App

A web-based tool built with Streamlit and PuLP to optimize rental property lease plans based on various client requirements and property constraints. This application leverages Mixed-Integer Linear Programming (MILP) to find the most cost-effective solutions.

## ✨ Features

* **Flexible Property Data Input:**
    * Generate random property data for quick testing.
    * Upload your own property details via a CSV file with a predefined format.
    * Download a sample CSV to understand the expected input structure.
* **Customizable Client Requirements:**
    * Specify target units required for the lease.
    * Set a minimum cost target (budget) with an upper tolerance.
    * Define the lease period (start and end dates).
* **Advanced Optimization Controls:**
    * **Full Property Penalty:** Penalize solutions that lease only a portion of a property's available units.
    * **Group Constraints:** Ensure a minimum number of properties are selected from specific property groups (e.g., specific geographical areas, property types).
    * **Occupancy Rating Penalty:** Discourage selecting properties with high existing occupancy ratings by applying a penalty.
    * **Pre-selected Properties:** Force specific properties to always be included in the lease plan.
* **Solver Tuning:**
    * Set time limits for the optimization solver to ensure quick responses.
    * Define an optimality gap tolerance to balance solution quality and computation time.
* **Multiple Solutions:** Generate and compare several distinct optimal or near-optimal lease plans.
* **Clear Output Summary:** Detailed breakdown of each optimized plan, including costs, units, penalties, and constraint fulfillment status.

## 🚀 Try the App Online!

You can access and experiment with the application directly in your web browser:

**[Click here to open the Lease Optimization App]([YOUR_APP_DEPLOYMENT_LINK_HERE])**

## 💻 How to Run Locally (For Developers)

If you wish to run and modify this application on your own machine, follow these steps:

### Prerequisites

* Python 3.8+ installed on your system.
* `pip` (Python package installer).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/my_lease_optimizer_app.git
    cd my_lease_optimizer_app
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

1.  **Ensure your virtual environment is active.**
2.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser (usually at `http://localhost:8501`).

## 📊 CSV Data Format

If you choose to upload your own property data via CSV, please ensure it adheres to the following format:

**`properties.csv` example:**

```csv
property_id,total_units,rate_per_day,occupancy_rating,group
MyProp1,10,1250.50,6,A
BigOffice,25,2000.00,9,B
SmallUnit,4,800.00,3,C