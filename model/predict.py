import joblib

# Load model
model = joblib.load('flood_model.pkl')

# Example input
rainfall = float(input("Rainfall: "))
water_level = float(input("Water Level: "))
humidity = float(input("Humidity: "))

prediction = model.predict([[rainfall, water_level, humidity]])

if prediction[0] == 1:
    print("Flood Risk: HIGH 🔴")
else:
    print("Flood Risk: LOW 🟢")
