Flight Fare Prediction

Project Overview
This project aims to predict flight fares based on various features like airline, source city, destination city, number of stops, departure time, and class, among others. It uses machine learning models to analyze how these factors influence flight prices.

Dataset
The dataset contains 300,153 rows and 11 columns with features such as:
- `airline`: The airline operating the flight
- `flight`: The flight number
- `source_city`: The city from which the flight originates
- `destination_city`: The destination city of the flight
- `departure_time`: Time of flight departure
- `arrival_time`: Time of flight arrival
- `stops`: Number of stops during the flight
- `class`: Travel class (Economy, Business)
- `duration`: Flight duration in hours
- `days_left`: Days left before departure
- `price`: The flight fare

Libraries Used
- Pandas: For data manipulation and analysis
- NumPy: For numerical computing
- Matplotlib & Seaborn: For data visualization
- Scikit-learn: For machine learning models and preprocessing
- XGBoost: For implementing gradient boosting algorithms
- LabelEncoder: For encoding categorical variables

Data Preprocessing
- Removed unnecessary columns
- Encoded categorical variables using LabelEncoder
- Split the data into training and test sets (70% training, 30% test)
- Scaled features using MinMaxScaler for model training

Exploratory Data Analysis
- Airline Popularity: Visualized airline counts, finding Indigo as the most popular airline.
- Class Distribution: Pie chart showing that most flights are Economy class.
- Price Analysis: Compared ticket prices between airlines, classes, stops, departure/arrival times, and cities.
- Days Left for Departure vs Price: Price tends to increase as the departure date approaches.

Machine Learning Models
The following machine learning models were trained to predict flight prices:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
 - K-Nearest Neighbors (KNN)
- Extra Trees Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Bagging Regressor
- Ridge Regression
- Lasso Regression

 Model Evaluation
The models were evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-Squared (R²)
- Adjusted R-Squared
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Log Error (RMSLE)

 Best Performing Model:
- Random Forest Regressor
  - MAE: 1168.713
  - MSE: 8,199,761.81
  - RMSE: 2863.523
  - R²: 0.984068
  - MAPE: 7.91%

Conclusion
- Indigo is the most frequent airline, and Vistara has the highest ticket prices.
- Business class tickets are generally more expensive than Economy class.
- Flights with more stops tend to have higher fares.
- Flight duration and the number of days left before departure also significantly affect prices.

Usage
1. Clone this repository.
2. Ensure the dataset (`fair_airline.csv`) is in the working directory.
3. Run the Jupyter Notebook or the Python script to explore the analysis and train models.

Future Work
- Experiment with other regression algorithms such as Neural Networks.
- Implement hyperparameter tuning to improve model accuracy.
- Consider feature engineering to enhance the predictive capability of the models.

License
This project is licensed under the MIT License.
