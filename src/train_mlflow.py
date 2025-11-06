import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# ---------- 1. Set experiment name ----------
mlflow.set_experiment("FakeNews_Fusion_Test")

# ---------- 2. Start run ----------
with mlflow.start_run(run_name="test_run"):
    # Dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    # Simple model (we’ll later replace this with your real fusion model)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train briefly
    history = model.fit(X, y, epochs=3, batch_size=8, verbose=1)

    # Log metrics and parameters
    mlflow.log_metric("final_accuracy", history.history['accuracy'][-1])
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 8)

    # Log the trained model
    mlflow.keras.log_model(model, "model")

print("✅ MLflow tracking test complete!")
