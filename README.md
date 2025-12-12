# Apex Project: Customer Segmentation Analysis

This project performs a customer segmentation analysis on Blinkit order data using K-Means clustering. It is part of the Apex Project 1 by Team Plotmasters.

## Project Overview

The analysis aims to identify distinct customer segments to tailor marketing strategies and improve business outcomes. The notebook covers:

- **Data Loading & Validation**: Initial inspection of the Blinkit order dataset.
- **Data Cleaning**: Handling missing values, outliers, and duplicates.
- **Exploratory Data Analysis (EDA)**: Understanding data distributions and relationships.
- **Feature Engineering**: Creating customer-centric features (e.g., total spent, average delay).
- **Cluster Modelling**: Applying K-Means clustering to segment customers.
- **Inferences**: deriving insights from the 2 identified clusters.

## Team Plotmasters (BITS Pilani)

- Utkarsh Tripathi
- Juwaria Qadri
- Mohammed Omar
- Merin Ann Cherian

## Usage

This project is built with [Quarto](https://quarto.org/).

### Prerequisites

- Python 3.x
- Dependencies listed in `requirements.txt`

### Local Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Render the website:**
    ```bash
    quarto preview
    ```
    Or to render to the `docs` folder:
    ```bash
    quarto render
    ```

## Deployment

This website is automatically deployed to GitHub Pages via GitHub Actions.
