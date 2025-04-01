import mlflow

model_name = "fasttext"
version = 5

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

list_libs = ["vendeur d'huitres", "boulanger"]

results = model.predict(list_libs, params={"k": 3})
print(results)