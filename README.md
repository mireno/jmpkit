# jmpkit
Implementing JMP-style statistical analysis in Python

1. Start a new venv (make sure to use python 3.12)
    python3.12 -m venv .venv
    . .venv/bin/activate

2. Install pip requirements
    pip install --upgrade pip
    pip install -r requirements.txt

4. Launch streamlit from inside the vnenv
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0