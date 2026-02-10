# Fatigue Database Intelligent Platform for Metallic Materials

## Overview

Metal fatigue plays a decisive role in the selection and qualification of materials for automotive, aerospace, and other safety-critical structural applications. Although fatigue testing is experimentally intensive, the resulting data is often fragmented across spreadsheets, machine exports, and image repositories, severely limiting reuse, comparison, and integration with ICME (Integrated Computational Materials Engineering) workflows.

This repository presents a **fatigue-specific database intelligence platform** that converts raw experimental fatigue data into a **structured, queryable, and analytics-ready digital asset**. The framework is demonstrated using **low-cycle fatigue (LCF) datasets of Aluminium 6063 alloy** processed through **Heat Treatment (HT), Deep Cryogenic Treatment (DCT), and Equal Channel Angular Pressing (ECAP)**.

The platform integrates **experimental metallurgy, database management systems (DBMS), SQL-based analytics, and demonstrative AI/ML modelling**, while preserving fatigue physics and experimental traceability.

---

## Key Objectives

- Standardise raw experimental fatigue data into a fatigue-aware schema  
- Preserve **Process–Structure–Property–Performance (PSPP)** linkages  
- Enable **query-driven fatigue knowledge extraction** using SQL  
- Demonstrate **ML-based fatigue life prediction** using structured datasets  
- Support **AI-enabled ICME workflows** and industrial adoption, particularly for MSMEs  

---

## Experimental Foundation

The platform is built on experimentally validated datasets generated from **Aluminium 6063 alloy**, including:

### Processing Routes
- Heat Treatment (HT)
- Deep Cryogenic Treatment (DCT)
- Equal Channel Angular Pressing (ECAP – 90° and 120°)

### Mechanical and Fatigue Testing
- Tensile testing
- Microhardness testing
- Strain-controlled Low-Cycle Fatigue (LCF) testing (ASTM E606)

### Fatigue Outputs
- Cycles to failure
- Mean stress evolution
- Stress amplitude
- Plastic strain amplitude
- Cyclic stress–strain hysteresis loops

### Microstructural and Fractographic Descriptors
- Grain size
- Mg₂Si precipitation
- Fracture modes and crack propagation features

---

## Platform Architecture

The fatigue database framework follows a **five-layer architecture**:

1. **Data Collection**  
   Raw experimental outputs from fatigue testing machines (Excel/CSV), including cycle-wise hysteresis loop data, tensile properties, microhardness values, and microstructural observations.

2. **Data Cleaning and Standardisation**  
   Removal of non-physical post-failure data, unit normalisation, and harmonisation of parameter naming conventions.

3. **Database Design and ETL**  
   Python-based Extract–Transform–Load (ETL) pipeline that maps experimental hierarchy into a relational database while preserving experimental lineage.

4. **Query-Driven Knowledge Extraction**  
   SQL-based retrieval and comparison of fatigue behaviour across processing routes and test conditions.

5. **AI/ML Demonstration**  
   Proof-of-concept ML models for fatigue life prediction using physics-consistent experimental features.

---

## Repository Structure
fatigue-dbms-platform/
│
├── data/
│ ├── raw/
│ │ ├── fatigue_spreadsheets/
│ │ ├── hysteresis_loops/
│ │ └── tensile_hardness/
│ ├── cleaned/
│ └── metadata/
│
├── etl/
│ ├── extract.py
│ ├── transform.py
│ ├── load.py
│ └── etl_pipeline.py
│
├── database/
│ ├── schema.sql
│ ├── table_definitions.sql
│ └── sample_queries.sql
│
├── analysis/
│ ├── sql_queries/
│ ├── fatigue_statistics.ipynb
│ └── visualization.ipynb
│
├── ml/
│ ├── feature_engineering.py
│ ├── fatigue_life_model.py
│ └── model_evaluation.ipynb
│
├── figures/
│ └── workflow_diagrams/
│
├── README.md
└── LICENSE




---

## Technologies Used

- **Python** – ETL pipeline, preprocessing, ML  
- **PostgreSQL / SQLite** – relational database  
- **SQL** – query-driven fatigue analysis  
- **Pandas, NumPy** – data handling  
- **Scikit-learn** – ML demonstration  
- **Matplotlib / Seaborn** – visualisation  

---

## Example Use Cases

- Compare fatigue life of HT, DCT, and ECAP-processed Al 6063 at fixed strain amplitude  
- Query cyclic response trends across processing routes  
- Correlate tensile properties, microhardness, and fatigue life  
- Demonstrate ML-based fatigue life prediction using experimental features  
- Prepare fatigue-aware datasets for ICME workflows  

---

## Scope and Limitations

- Demonstrated using **Al 6063 LCF datasets**  
- ML models are **illustrative**, not production-grade predictors  
- Focus is on **data structuring and knowledge extraction**, not black-box optimisation  
- Framework is extensible to other metallic systems and fatigue regimes  

---

## Intended Users

- Fatigue researchers and materials scientists  
- Experimental metallurgists  
- ICME practitioners  
- MSMEs and industrial R&D teams  
- Graduate students learning fatigue data management and analytics  

---

## Citation

If you use this platform or methodology in your research, please cite the associated journal article:

> **Fatigue Database Intelligent Platform for Metallic Materials**  
> (Journal details to be updated)

---

## Author

**Sreearravind M.**  
Ph.D. in Mechanical Engineering (Fatigue and Metallurgy)  
Research focus: Aluminium alloy fatigue, ICME, materials data systems

---

## License

This project is released for **academic and research use**.  
Commercial use may require prior permission.

---

## Quick Start: Create SQLite Database from CSV

This repository includes a helper script to build a SQLite database directly from `Pilot_v1.csv`.

```bash
python3 create_database.py
```

This command generates:
- `fatigue_data.db` (SQLite database file)
- `pilot_data` table containing all rows from `Pilot_v1.csv`

### Download / Transfer Notes

- GitHub does **not preview** `.db` files in diffs (it shows “Binary file not shown”). This is expected behavior.
- Instead of downloading a committed binary database, create it locally using the command above.
- To transfer the database file, zip it first if needed:

```bash
zip fatigue_data.zip fatigue_data.db
```

- Then upload/share `fatigue_data.zip` through GitHub Releases, cloud storage, or email.
