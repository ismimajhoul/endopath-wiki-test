# ENDOPATH
## Description
Endopath is an AI-assisted diagnostic tool designed to analyze gynecological medical reports and help diagnose complex endometriosis. The goal is to reduce invasive procedures and shorten the patient journey by leveraging natural language processing on clinical texts.

## Key Features
Medical text preprocessing with domain-specific spell correction and abbreviation normalization.

Handling of missing data by filtering patients with high missingness rates.

Predictive modeling using XGBoost classifier to identify endometriosis with targeted sensitivity and specificity.

Behavior-driven development (BDD) tests to ensure reliability and accuracy.

## Technologies
Python, pytest-bdd for testing

Pandas for data processing

XGBoost for machine learning

NLP techniques for text normalization

## Development Approach
Built with Domain-Driven Design (DDD) to closely align with clinical needs and terminology in gynecology.