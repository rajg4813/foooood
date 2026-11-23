# NutriPlan AI — Indian Food Nutrition Recommender

A beginner-friendly BTech CSE AI/ML project built with Python and Streamlit. It uses real food-nutrient records from the **Anuvaad Indian Nutrient Databank (INDB), November 2024 workbook** provided for this project.

## What's inside

- `app.py` — Streamlit web application
- `data/indb_foods.csv` — prepared 16-field extract used by the app
- `data/Anuvaad_INDB_2024.11.xlsx` — original workbook, included unchanged
- `requirements.txt` — Python dependencies

## Dataset details

The source workbook contains:
- **1,014 food records** (excluding the header)
- **82 original columns**
- Three primary source labels: `asc_manual`, `bfp_manual`, and `open_source_recipes`

The app uses 16 selected fields: food code/name/source; energy, carbohydrate, protein, fat, free sugar and fibre per 100 g; plus serving-unit nutrition fields when available.

Some per-serving nutrition values are missing. To keep comparisons consistent, the recommender and food library use nutrient values **per 100 g**. Per-serving fields are retained in the CSV for reference but are not used to drive the recommendation model.

## AI/ML used

The project uses **K-Nearest Neighbors (KNN)** for content-based nutrient matching.

1. Select six nutrient fields for each food: energy, carbohydrates, protein, fat, free sugar and fibre.
2. Standardize the fields using `StandardScaler` so values with different units/scales can be compared.
3. Fit a `NearestNeighbors` model on the food records.
4. For each meal slot and selected goal, compare a small, documented example nutrient profile with the food records.
5. Display the closest nutrient-profile matches from the actual dataset.

This is nearest-neighbor retrieval, not a supervised prediction model. It does not claim a classification accuracy score. The goal-specific nutrient profiles are simple demonstration heuristics, not official dietary guidelines.

## Calorie estimate

The plan page uses the Mifflin–St Jeor equation plus an activity multiplier to produce a rough daily energy estimate. It is a transparent formula, not machine learning. The macro split is a simple illustrative 25% protein / 45% carbohydrate / 30% fat split.

## Run locally (Windows / VS Code)

1. Extract the ZIP.
2. Open the extracted `NutriPlan_AI_Project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Install the libraries:

   ```powershell
   pip install -r requirements.txt
   ```

5. Start the website:

   ```powershell
   streamlit run app.py
   ```

6. Open `http://localhost:8501` if it does not open automatically.

To stop the website, press `Ctrl + C` in the terminal.

## Pages

- **Overview** — introduction and dataset/model summary
- **Build my plan** — enter basic profile details and receive goal-based food matches
- **Food library** — search, filter and compare food records
- **Dataset & insights** — row/column counts, data source distribution, calorie histogram and preview

## Important limitations

- Nutrition information is shown per 100 g, not personalized serving sizes.
- The dataset does not provide complete ingredient or allergen lists. The optional “foods to avoid” field only filters food names; it is not an allergy-safety check.
- The app is an educational prototype and is not a medical nutrition service. It does not diagnose, treat, or prescribe diets. Personal requirements differ; consult a qualified dietitian for individualized advice.

## Explain it in viva

“My project is NutriPlan AI, an Indian food nutrition recommender. It uses the Anuvaad Indian Nutrient Databank, containing 1,014 food records. I use K-Nearest Neighbors after standardizing six nutrient features—energy, carbohydrates, protein, fat, free sugar and fibre. The model retrieves foods whose nutrient profiles are closest to a selected goal profile. The website also provides dataset exploration, food search and a transparent calorie-estimation formula.”
