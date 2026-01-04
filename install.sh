

# check python version is larger than 3.10
if python -c 'import sys; exit(0 if sys.version_info < (3,12) else 1)'; then
  echo "Python version is less than 3.12. Python version with 3.12 or higher is required."
  exit 1


# If file config.json file not exists in ~/.config/easylocai, copy it
if [ ! -f ~/.config/easylocai/config.json ]; then
  mkdir -p ~/.config/easylocai
  cp default_config.json ~/.config/easylocai/config.json
  echo "Copied default config.json to ~/.config/easylocai/"
fi

pipx install --python python3.12 .
