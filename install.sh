

# check python version is larger than 3.10
if python -c 'import sys; exit(0 if sys.version_info < (3,10) else 1)'; then
  echo "Python version is less than 3.10. Python version with 3.10 or higher is required."
  exit 1


# install uv
else
  if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Installing uv..."
    brew install uv
  fi
fi

# make virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv sync
