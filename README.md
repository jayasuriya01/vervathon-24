Team Name - Data Mavericks

Problem Statement -
  The task is to develop a smart finance tracking system
  that allows users to monitor their income, expenses,
  savings, and investments while providing insightful
  visualizations of their financial health. The system should
  leverage data analytics to not only help users track their
  finances but also offer predictive insights, trend analysis,
  and actionable recommendations based on their
  spending and saving patterns.
  
Team Leader Email - fazilabdul2005@gmail.com

Prototype Brief: Smart Finance Tracking System
  This prototype is designed to track financial metrics and predict future economic outcomes using a web-based interface built with Streamlit. The system integrates multiple financial datasets, performs data processing, and visualizes key financial trends over time. Key functionalities include:
  
  Data Loading & Processing:
  The system loads Excel datasets related to various financial metrics such as Economic Output, Infrastructure Asset Investments, Operational Costs, and more.
  Data is cleaned and reshaped to enable state-wise and year-wise analysis.
  
  Data Visualization:
  The app provides a summary of the merged datasets and displays a high-level financial overview for each state.
  Users can visualize financial trends over time with dynamic line plots using Seaborn and Matplotlib.
  
  Predictive Analysis:
  A linear regression model is applied to predict future economic output for selected states.
  Users can select a state and predict its economic output for the upcoming years (e.g., 2024-2030).
  Performance metrics like R-squared and Mean Squared Error (MSE) are shown to evaluate model accuracy.
  
  Power BI Dashboard:
  The prototype complements the Streamlit app with a Power BI dashboard for enhanced data visualization, enabling deeper insights through interactive and customizable financial reports.

Tech Stack:
  Programming Languages:

Python: Core language used for data processing, visualization, and machine learning.
Web Framework:

Streamlit: Used to build the interactive web application for data visualization and model predictions.
Data Handling:

Pandas: For data manipulation, cleaning, and merging financial datasets.
OpenPyXL: To load and work with Excel files.
Data Visualization:

Matplotlib: For plotting financial trends and predictive results.
Seaborn: To create aesthetically pleasing and insightful line plots.
Machine Learning:

Scikit-learn: Used for linear regression modeling and performance evaluation (R-squared and Mean Squared Error).
PyTorch: Included for potential future deep learning extensions.
Dashboarding:

Power BI: To create an advanced visualization dashboard for interactive data analysis.
Caching:

Streamlit Cache: To optimize data loading and improve performance by caching datasets.
Excel File Management:

XLSX and XLS Files: The datasets are stored in Excel files, which are loaded into the system for processing.

what i learned:
  The biggest learning was combining data processing, machine learning, and visualization. I learned how to clean and merge messy datasets, which is crucial for analysis. Using **Linear Regression**, I predicted future financial outputs and evaluated model performance with metrics like **R-squared** and **Mean Squared Error**. Streamlit made building the interactive web app simple, while tools like **Pandas** for data manipulation and **Matplotlib/Seaborn** for visualization helped in creating clear financial insights.
