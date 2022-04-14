# DiabeticRetinopathyDetection

### Streamlit
Procfile: 
web: streamlit run --server.enableCORS false --server.port $PORT app.py

requirements:
streamlit
matplotlib
numpy
https://download.pytorch.org/whl/cpu/torch-1.7.1%2Bcpu-cp38-cp38-linux_x86_64.whl
https://download.pytorch.org/whl/cpu/torchvision-0.8.2%2Bcpu-cp38-cp38-linux_x86_64.whl
fastai==2.5.2

### Voila
Procfile: 
web: voila --port=$PORT --no-browser --enable_nbextensions=True dr_voila.ipynb

requirement:
voila
ipywidgets
https://download.pytorch.org/whl/cpu/torch-1.7.1%2Bcpu-cp38-cp38-linux_x86_64.whl
https://download.pytorch.org/whl/cpu/torchvision-0.8.2%2Bcpu-cp38-cp38-linux_x86_64.whl
fastai==2.5.2
