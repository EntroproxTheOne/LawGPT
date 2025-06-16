import google.generativeai as genai

# Step 1: Set up your API key
genai.configure(api_key="AIzaSyAc_IaJ5dTGKL6VOjpPQK1gX7CjiPiNnrw")

# Step 2: List all available models
models = genai.list_models()

# Step 3: Print their names
for model in models:
    print(model.name)
