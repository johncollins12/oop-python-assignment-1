# OOP with Python – Assignment 1 (Advent 2026): Five Mini-Projects

**Author:** Wesonga John Collins · MSDS, OOP with Python

> Note: the cover page of the brief is headed "Assignment 2", but every page footer reads
> "Assignment 1, Advent 2026" and the content is the five mini-projects below. This repository is that
> assignment. Assignment 2 (weather forecasting and Uganda's tax engine) is a separate project.

Five mini-projects in a Ugandan context, each modelled with tested, reusable classes in `src/a1/` and
analysed in its own Jupyter notebook.

| # | Mini-project | Notebook | Module |
|---|---|---|---|
| 1 | UBOS district population forecaster | `notebooks/project1_population.ipynb` | `population.py` |
| 2 | Solar micro-grid dispatch planner (Kasese) | `notebooks/project2_microgrid.ipynb` | `microgrid.py` |
| 3 | Lake Victoria fish stock & export risk (Jinja) | `notebooks/project3_fish_stock.ipynb` | `fish_stock.py` |
| 4 | Rainfall pattern & crop suitability | `notebooks/project4_rainfall.ipynb` | `rainfall.py` |
| 5 | Taxi route revenue, pricing & fleet planner | `notebooks/project5_taxi_routes.ipynb` | `taxi_routes.py` |

Shared code: `forecasting.py` (abstract `Forecaster` base class and its subclasses, walk-forward
evaluation, bootstrap intervals; used by Projects 1 and 5) and `metrics.py` (MAE, RMSE, MAPE).

## Repository layout

```
src/a1/        reusable package (installed in editable mode)
tests/         pytest unit tests (160 tests)
notebooks/     one notebook per mini-project, saved with outputs
data/raw/      original downloaded data, unmodified (with SOURCES.md)
data/processed/ cleaned data produced by the scripts
scripts/       data-building and download scripts
```

## Setup (Windows Command Prompt; tested with Python 3.14.8)

```
git clone https://github.com/johncollins12/oop-python-assignment-1.git
cd oop-python-assignment-1
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` pins the libraries and installs the `a1` package in editable mode (`-e .`), so the
notebooks and tests can `import a1` from any folder. On macOS/Linux use `python3 -m venv .venv` and
`source .venv/bin/activate`.

## Running

```
pytest -q                                   # all unit tests
jupyter notebook                            # open and run notebooks interactively
jupyter nbconvert --to notebook --execute --inplace notebooks\project1_population.ipynb
```

The `nbconvert` command runs a notebook top to bottom in a fresh kernel (equivalent to *Restart & Run All*).
All random steps use `numpy.random.default_rng(seed)` with fixed seeds, so results are reproducible.

The processed data are committed, so no download is needed. To rebuild them:

```
python scripts\build_population_data.py     # MP1: UBOS files in data\raw\population -> population.csv
python scripts\fetch_nasa_power.py          # MP4 extension: downloads NASA POWER rainfall (internet needed)
```

Project 2 regenerates `data/microgrid_demand.csv` from its seed when the notebook runs. Its interactive
input mode is off by default (`INTERACTIVE = False`) so that automated runs never wait for typing.

## Data sources

* **UBOS** – Census population counts (2002, 2014) and mid-year projections 2015–2021; NPHC 2024
  sub-county profiles. Details, URLs and the boundary assumption for Jinja are in
  `data/raw/population/SOURCES.md`.
* **NASA POWER** – monthly `PRECTOTCORR` precipitation, 2014–2023, for Kampala, Gulu and Mbarara.
  *These data were obtained from the NASA Langley Research Center POWER Project funded through the NASA
  Earth Science Directorate Applied Science Program* (https://power.larc.nasa.gov).
* **Crop water needs** – Critchley, W. & Siegert, K. (1991) *Water Harvesting*, FAO, Rome
  (https://www.fao.org/4/u3160e/u3160e04.htm); NaCRRI (2010) *Establishment and field management of Robusta
  coffee*, NARO (https://www.naroinfohub.naro.go.ug/RepDetails.aspx?pn=3940).
* **Uganda rainfall regimes** – Kisembe et al. (2018), *Theoretical and Applied Climatology*
  (https://link.springer.com/article/10.1007/s00704-018-2643-x).
* All other data are the illustrative values given in the brief, or simulated with fixed seeds as stated in
  each notebook.

## Summary of findings

Each notebook ends with a full *Findings & Limitations* section; the key points are:

1. **Population (MP1).** Constant-percentage (CAGR) growth predicted the held-out years 2022–2024 best in four
   of five districts; Busia's slowdown favoured a linear trend. Wakiso grows fastest (CAGR 6.47 %) and needs the
   most additional classrooms by 2029 (about 2,100), Kampala about 1,550. The Fibonacci-ratio model implies
   roughly 62 % annual growth and failed badly (test MAPE 78–94 %). For the real districts (Jinja and Busia),
   the 2024 census departed from UBOS's projected path in opposite directions, and bootstrap intervals are
   narrower than these real errors.
2. **Micro-grid (MP2).** The system is well-posed (det = −5, condition number 5.83). The battery is the more
   volatile source (CV 41 % vs 9 % for solar) and makes up 42 % of the monthly cost of about UGX 213,000. One of
   30 simulated days was infeasible and was handled with non-negative least squares. The Monte Carlo
   amplification (5.76) stayed below the condition-number bound. A diesel generator keeps the system solvable
   but more than doubles cost; a linearly dependent constraint makes it singular.
3. **Fish stock (MP3).** A harvest rate of 0.20 per week achieves the maximum sustainable yield (rK/4 =
   1,000 t/week); 0.30 is overfishing, giving the same catch as 0.10 from a stock one third the size. Revenue
   risk is "moderate" (CV about 10 %) at every rate because prices drive it, so financial and biological risk
   need separate measures. A raw variance threshold is meaningless because variance is in UGX². A closed season
   raises long-run revenue only when the stock is over-fished.
4. **Rainfall (MP4).** Cosine similarity rated all regions alike (0.75–0.89) while Pearson correlations were
   near zero, because cosine ignores magnitude. Season detection found Mbarara bimodal and Gulu with two peaks
   in one long wet season, but Kampala unimodal on the illustrative data. Ten years of NASA POWER data showed
   all three regions bimodal (Kampala peaks in April and November), with 17–23 % year-to-year variation in
   annual rainfall. Beans suit Mbarara's two seasons best.
5. **Taxi routes (MP5).** The Ntinda fare of UGX 2,000 is below the equilibrium of UGX 2,200, leaving a
   shortage of about 10 passengers per trip-hour. Tuned exponential smoothing chose α = 1 (the
   "same as yesterday" forecast) on every route, and every route needs only one vehicle on day 11 at 40–57 %
   seat use. With a simulated weekly pattern, seasonal naïve forecasting cut error by about two thirds
   compared with the moving average.

## AI-use declaration

I used AI tools to support the development of this assignment. The AI assistant provided assistance with interpreting the assignment requirements, exploring software design approaches and debugging Python code. Artificial Intelligence (AI) tools were used to support the development of this assignment. The AI assistant provided assistance with interpreting the assignment requirements, exploring software design approaches, explaining programming concepts, developing and debugging Python code, preparing automated tests, and structuring Jupyter notebooks and documentation.

