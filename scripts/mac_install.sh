#!/bin/bash

echo "starting run, use 'q' to quit more" >> tmp.out
echo "" >> tmp.err

# install homebrew
if ! which -s brew; then
    echo "----------------------------------------------------------------------"
    echo "----------------------------------------------------------------------"
    echo "Do you want to install homebrew (yes/no) "
    read -r install_homebrew
    if [[ $install_homebrew == "yes" ]]; then
      ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    else
      echo "not installing homebrew" >> tmp.out
    fi
else
    brew update
fi


# install x-code
# https://stackoverflow.com/questions/15371925/how-to-check-if-command-line-tools-is-installed
xcode_exist=$(xcode-select -p 1>/dev/null;echo $?) # 0 == exists, 2 != exists
if [[ $xcode_exist -eq 2 ]]; then
  echo "----------------------------------------------------------------------"
  echo "----------------------------------------------------------------------"
  echo "Do you want to install xcode (yes/no) "
  read -r install_xcode
  if [[ $install_xcode == "yes" ]]; then
    xcode-select --install
  else
    echo "not installing xcode" >> tmp.out
  fi
else
  echo "xcode already installed, skipping xcode" >> tmp.out
fi

# instal conda
if ! which -s conda; then
  echo "----------------------------------------------------------------------"
  echo "----------------------------------------------------------------------"
  echo "Do you want to install conda (yes/no) "
  read -r install_conda
  if [[ $install_conda == "yes" ]]; then
    if [[ $(uname -m) == 'arm64' ]]; then
      echo "getting conda from m1, arm64" >> tmp.out
      wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh | bash
    else
      echo "getting conda from x86_64" >> tmp.out
      wget miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh | bash
    fi
    # turning off base environment
    echo "turning off base environment" >> tmp.out
    conda config --set auto_activate_base false
    conda create --name E4579 python=3.8
    conda init bash zsh
    conda activate E4579
  else
    echo "not installing conda, going to use venv" >> tmp.out
    if ! which -s python3; then
      if ! which -s brew; then
        echo "brew and python3 not installed, can't figure it out, bye" >> tmp.err
        exit 1
      else
        echo "installing python3 with brew" >> tmp.out
        brew install python
      fi
    fi
    python3 -m venv E4579
    source E4579/bin/activate
    python3 -m pip install --user --upgrade pip
  fi
else
  echo "conda already installed, will update conda" >> tmp.out
  conda update -n base conda
  if [[ $( conda env list | grep E4579 | wc -l) -eq 1 ]]; then
    echo "env E4579 already exists, wooo" >> tmp.out
  else
    conda create --name E4579 python=3.8
  fi
fi

# install .env file
if [[ $(find . -maxdepth 1 -name '.env' | wc -l) -eq 0 ]]; then
  echo "Create .env (yes/no) "
  read -r create_dot_env
  if [[ $create_dot_env == "yes" ]]; then
    echo "aws_db_password=
  aws_db_endpoint=
  aws_db_username=
  aws_db_port=
  aws_db_schema=
  use_aws_db=0
  FLASK_APP=project
  FLASK_DEBUG=1
  SQLALCHEMY_TRACK_MODIFICATIONS=False
  use_picsum=1" | sudo tee .env
    echo "If you want an AWS DB you will need to fill in the details in the .env file" >> tmp.out
  else
    echo "not creating .env file" >> tmp.out
  fi
else
  echo ".env already exists, not overwriting" >> tmp.out
fi

echo "PRINTING OUT LOGS, use 'q' to quit"

more tmp.out

echo "ERRORS:"
more tmp.err


