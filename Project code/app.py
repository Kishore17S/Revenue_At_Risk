import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

def calculate_supply_readiness(inventory_to_demand_ratio):
    # Assuming Supply Readiness is ready if ratio > 0.8
    return (inventory_to_demand_ratio > 0.8).astype(int)

# Load the trained model
model = joblib.load('revenue_status_model_selected_features_v2.pkl')

# Streamlit App
st.title('Revenue Status Prediction')
st.sidebar.header('Input Parameters')

# Collect input from user
demand_forecast = st.sidebar.text_input('Demand Forecast (comma-separated values):', '1000, 5000, 1500')
current_inventory = st.sidebar.text_input('Current Inventory (comma-separated values):', '1200, 3000, 2600')

# Process input into lists of floats
demand_forecast = list(map(float, demand_forecast.split(',')))
current_inventory = list(map(float, current_inventory.split(',')))

# Calculate additional features
inventory_to_demand_ratio = [inv / dem for inv, dem in zip(current_inventory, demand_forecast)]
inventory_demand_difference = [inv - dem for inv, dem in zip(current_inventory, demand_forecast)]
supply_ready = calculate_supply_readiness(np.array(inventory_to_demand_ratio))


# Create input dataframe
input_data = pd.DataFrame({
    'Demand_Forecast': demand_forecast,
    'Current_Inventory': current_inventory,
    'Inventory_to_Demand_Ratio': inventory_to_demand_ratio,
    'Inventory_Demand_Difference': inventory_demand_difference,
    'Supply_Ready': supply_ready
})

# Display input data
st.subheader('Input Data')
st.write(input_data)

# Visualization of total demand vs. inventory
st.subheader('Input Data Overview')
fig, ax = plt.subplots()
ax.bar(['Demand Forecast', 'Current Inventory'], [sum(demand_forecast), sum(current_inventory)])
st.pyplot(fig)

# Prediction functionality
if st.button('Predict Revenue Status'):
    if input_data.empty or len(demand_forecast) == 0:
        st.error("Input data is empty! Please provide valid input values.")
    else:
        # Generate predictions using the model
        predictions = model.predict(input_data)

        # Override predictions based on supply readiness condition
        prediction_labels = []
        for supply, pred in zip(supply_ready, predictions):
            if supply == 0:
                prediction_labels.append("At Risk")
            elif supply == 1:
                prediction_labels.append("Confirmed")
            else:
                # Use model prediction if no explicit condition is met
                prediction_labels.append('Confirmed' if pred == 1 else 'At Risk')

        # Add predictions to the input data
        input_data['Revenue_Status'] = prediction_labels

        # Display predictions
        st.subheader('Predictions')
        st.write(input_data[['Demand_Forecast', 'Current_Inventory', 'Supply_Ready', 'Revenue_Status']])

        # Insights for future actions
        st.subheader('Recommended Actions')
        actions = []
        for status in prediction_labels:
            if status == "Confirmed":
                actions.append("Revenue is confirmed. Proceed with planned operations.")
            else:
                actions.append("Revenue is at risk! Consider taking the following actions:\n"
                               "- Increase inventory levels\n"
                               "- Review demand forecasts\n"
                               "- Communicate with relevant teams")
        st.write("\n\n".join(actions))



