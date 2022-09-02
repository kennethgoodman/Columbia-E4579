#!/bin/bash

YES_ALL=0
while getopts 'y' OPTION; do
  case "$OPTION" in
    y)
      YES_ALL=1
      echo "yes mode"
      ;;
    ?)
      echo "script usage: $(basename \$0) [-y] for yes to all" >&2
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"

echo "starting run, use 'q' to quit more" > tmp.out
echo "" > tmp.err

# install tensorflow
if [[ $(pip freeze | grep -E 'tensorflow-macos|tensorflow-metal' -c) -eq 2 ]]; then
  echo "pip tensorflow dep already installed" >> tmp.out
  if ! which -s conda; then
    echo "not checking tensorflow-deps for conda" >> tmp.out
  else
    if [[ $(conda list | grep tensorflow-deps -c) -eq 1 ]]; then
      echo "conda tensorflow-deps exists" >> tmp.out
    else
      install_tf="no"
      if [[ $YES_ALL -eq 0 ]]; then
        echo "----------------------------------------------------------------------"
        echo "----------------------------------------------------------------------"
        echo "Do you want to install tensorflow (yes/no) "
        read -r install_tf
      else
        install_tf="yes"
      fi
      if [[ $install_tf == "yes" ]]; then
        conda install -c apple tensorflow-deps
      else
        echo "not installing tensorflow" >> tmp.out
      fi
    fi
  fi
else
  install_tf="no"
  if [[ $YES_ALL -eq 0 ]]; then
    echo "----------------------------------------------------------------------"
    echo "----------------------------------------------------------------------"
    echo "Do you want to install tensorflow (yes/no) "
    read -r install_tf
  else
    install_tf="yes"
  fi
  if [[ $install_tf == "yes" ]]; then
    if ! which -s conda; then
      echo "not installing tensorflow-deps because no conda" >> tmp.err
    else
      if [[ $(conda list | grep tensorflow-deps -c) -eq 1 ]]; then
        echo "conda tensorflow-deps exists" >> tmp.out
      else
        conda install -c apple tensorflow-deps
      fi
    fi
    pip install tensorflow-macos
    pip install tensorflow-metal
  else
    echo "not installing tensorflow" >> tmp.out
  fi
fi

# install other python dependencies
install_libs="no"
if [[ $YES_ALL -eq 0 ]]; then
  echo "----------------------------------------------------------------------"
  echo "----------------------------------------------------------------------"
  echo "Do you want to install non tensorflow python dependencies (yes/no) "
  read -r install_libs
else
  install_libs="yes"
fi

if [[ $install_libs == "yes" ]]; then
  conda install -c conda-forge -y pandas jupyter
  conda install -c conda-forge mysql
  pip install -r requirements.txt
else
  echo "not installing python dependencies" >> tmp.out
fi

# install other python dependencies
local_db_install="no"
if [[ $YES_ALL -eq 0 ]]; then
  echo "----------------------------------------------------------------------"
  echo "----------------------------------------------------------------------"
  echo "Do you want to install the local database, will OVERWRITE if already exists (yes/no) "
  read -r local_db_install
else
  local_db_install="yes"
fi
if [[ $local_db_install == "yes" ]]; then
  export PYTHONPATH=$PYTHONPATH:.
  python scripts/generate_db.py || exit
else
  echo "not installing local db" >> tmp.out
fi

# building frontend and installing
cd project/frontend || exit
npm install i
echo "----------------------------------------------------------------------"
echo "----------------------------------------------------------------------"
echo "Asking for password to install pm2 globally, which will allow us to run react and flask in the same terminal"
sudo npm install pm2 -g
pm2 --name REACTSIDE start npm -- start

cd ../../ # go back to root

# trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
        pm2 delete REACTSIDE  # if we exit flask
}

flask run

