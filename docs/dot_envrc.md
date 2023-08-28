# Example `.envrc` for Jupyter Notebook Virtual Environment

```bash
echo '=================================='
echo 'loading .envrc virtual environment'
echo '=================================='

# this (more or less) creates a python 3 virtual environment
layout python3

# this and the similar bit below is my hacky way of trying to not re-update pip
# or reinstall packages
export pip_flagDir=${PWD}/.direnv/python-3.11.4/pip_flags
mkdir -p $pip_flagDir
echo "UPDATE PIP?"
# if the flag file is there, don't update
if [ ! -f "${pip_flagDir}/.pip_updated" ]; then
  echo 'updating pip now'
  pip install --upgrade pip
  pip_success=$?
  if (( pip_success < 1 )); then
    touch $pip_flagDir/.pip_updated
  fi
else
  echo 'pip previously updated'
fi

# declare an array of pip packages
pip_installs=(
              "ipython"
              "ipykernel"
              "numpy"
              )
# now loop through the above array and only install if there's no flag file
for pkg in "${pip_installs[@]}"
do
  echo "INSTALL ${pkg}?"
  if [ ! -f "${pip_flagDir}/.${pkg}_installed" ]; then
    echo "installing ${pkg} now"
    pip install $pkg
    pkg_installed=$?
    if (( pkg_installed < 1 )); then
      touch $pip_flagDir/.${pkg}_installed
    fi
  else
    echo "${pkg} previously installed"
  fi
done

# enables debug mode
# export DEBUG=1

# Set the ipython and jupyter notebook config directories to be local
export IPYTHONDIR=$PWD/.ipython
export JUPYTER_CONFIG_DIR=$PWD/.jupyter

echo '=================================='
echo '        environment loaded        '
echo '=================================='
```
