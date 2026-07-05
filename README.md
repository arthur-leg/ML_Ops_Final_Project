MLOps Final Project


To run the backend : 
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m flask --app backend.api run --host 0.0.0.0 --port 5000

To run the frontend :
cd frontend
npm install
npm run dev