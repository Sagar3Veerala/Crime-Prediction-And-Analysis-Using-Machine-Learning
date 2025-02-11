# Crime Rate Prediction App

This Streamlit application provides a user-friendly interface for exploring crime data, performing crime rate analysis, and building a linear regression model to predict crime rates. Users can upload their own datasets in CSV or Excel format.

## Table of Contents

1.  [Project Description](#project-description)
2.  [Installation](#installation)
3.  [Usage](#usage)
4.  [Dataset](#dataset)
5.  [Code Structure](#code-structure)
6.  [Dependencies](#dependencies)
7.  [Features](#features)
8.  [Error Handling](#error-handling)


## Project Description

The Crime Rate Prediction App is designed to help users analyze and understand crime trends. The app includes the following functionalities:

*   **Data Exploration:** Allows users to explore uploaded datasets through an interactive dataframe explorer.
*   **Crime Rate Analysis:**  Provides visualizations (bar charts, plotly charts, area charts) to analyze crime patterns based on state, year, and district.
*   **Prediction:** Enables users to train a linear regression model on the uploaded data and predict crime rates for specific districts and states.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd crime-rate-prediction-app
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Create a `requirements.txt` file with the following content:

    ```
    streamlit
    streamlit-option-menu
    pandas
    plotly
    streamlit-extras
    scikit-learn
    matplotlib
    ```

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run crime_rate_app.py  # Replace crime_rate_app.py if your file has a different name
    ```

2.  **Upload your dataset:**  Use the file uploader in the sidebar to upload your crime data in CSV or Excel format.

3.  **Navigate the app:**  Use the horizontal menu to switch between the "Home", "Crime Rate Analysis", and "Prediction" sections.

    *   **Home:**  Provides a brief description of the project and displays the raw dataframe.
    *   **Crime Rate Analysis:**  Allows you to explore and filter the data using the interactive dataframe explorer and view visualizations of crime rates.
    *   **Prediction:**  Train a linear regression model and predict crime rates for specific districts and states.

## Dataset

The app expects a dataset in CSV or Excel format with columns that represent relevant information about crime, such as:

*   `STATE/UT`:  State or Union Territory
*   `DISTRICT`:  District name
*   `YEAR`:  Year of the crime
*   `MURDER`:  Number of murder cases (or another crime statistic of interest)
*   `TOTAL IPC CRIMES`: Total number of crimes

You may need to adapt the column names in the script to match the column names in your dataset.

## Code Structure

*   `crime_rate_app.py`:  The main Python script containing the Streamlit application code.

## Dependencies

The app relies on the following Python libraries:

*   Streamlit
*   streamlit\_option\_menu
*   Pandas
*   Plotly
*   streamlit\_extras
*   Scikit-learn
*   Matplotlib

## Features

*   **Interactive Data Exploration:** Explore and filter the data with the `dataframe_explorer`.
*   **Data Visualization:**  Generate insightful charts and graphs using Plotly and Matplotlib.
*   **Linear Regression Modeling:** Train and evaluate a linear regression model for crime rate prediction.
*   **User-Friendly Interface:**  Simple and intuitive interface for uploading data, selecting parameters, and viewing results.
*   **Error Handling:**  Robust error handling to provide informative messages to the user in case of data loading or processing issues.
*   **K-Fold Cross Validation:** Implemented K-Fold cross validation to assess and refine model performance.

## Error Handling

The app includes error handling to catch potential issues:

*   **File Upload Errors:** Catches errors related to reading CSV or Excel files.
*   **Data Type Errors:** Checks for incorrect data types and provides informative error messages.
*   **Key Errors:**  Handles missing column names.
*   **AttributeError:** Catches errors related to data formats.

