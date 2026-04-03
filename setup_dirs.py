import os

os.makedirs('ml_engine/saved_model', exist_ok=True)
os.makedirs('api/routes', exist_ok=True)
os.makedirs('api/models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print('Done')
