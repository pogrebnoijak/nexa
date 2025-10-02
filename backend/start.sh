DIR=`pwd`
export PYTHONPATH="${PYTHONPATH}:${DIR}"

# uvicorn app.api:app --reload --port 8000
uvicorn app.api:app --port 8000
