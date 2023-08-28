# Getting `clang-format` version 14

If you don't have the `mam4xx`-required version 14 of `clang-format` on your system and you're a Mac user, the likely easiest way to obtain it (because `clang-format@14` is not available on its own) is by using [`homebrew`](https://brew.sh) to install [`llvm@14`](https://formulae.brew.sh/formula/llvm@14#default)  and then adding it to your path. e.g.,
```bash
$ brew install llvm@14
$ export PATH="/opt/homebrew/opt/llvm@14/bin/clang-format:$PATH"
# Note: this is where it is for me on M1 mac--may be different on another system
# you can confirm where yours is with 'brew info llvm@14'
$ which clang-format
/opt/homebrew/opt/llvm@14/bin/clang-format
```
For non-mac users, I'd first check to see if `llvm v14` (or `clang-format v14`) is provided by your package manager. If not, you can always manually install and build `llvm v14`, though it's a slightly heavier lift than the above. First steps are:
```bash
git clone git@github.com:llvm/llvm-project.git
git checkout llvmorg-14.0.6 # version tag
```
Here is the [github repo](https://github.com/llvm/llvm-project/tree/llvmorg-14.0.6) for `llvm v14`, and it'll build faster/smaller if you use the flag `-DLLVM_ENABLE_PROJECTS="clang"` to only build `clang` and it's friends, rather than all of `llvm`. Other than that, the README offers solid guidance.

### Unnecessary but Convenient Workflow Customization

If you'd like to add a layer of automation/complexity to ensure you only use `clang-format v14` on `mam4xx` and/or want to use a newer version on the rest of your system, I'm a big fan of a terminal tool called [direnv](https://direnv.net/) (`brew install direnv`) that allows you to automatically load and unload environment variables based on `$PWD` using a `.envrc` file. As an example, here's my `.envrc` that adds `clang-format v14` to the path when I'm working on `mam4xx`.
```
PATH_add /opt/homebrew/opt/llvm@14/bin/clang-format

# also, since I often forget to load a python environment that's required for
# running ctest, this creates or loads a python 3 virtual environment with numpy
layout python3
pip install --upgrade pip
# the upgrade isn't strictly necessary but trades a little extra setup on the front end to avoid pip endlessly reminding you to update
pip install numpy
```
This file is placed in the top-level `mam4xx` directory and runs when I `cd mam4xx`, stays loaded in any subdirectories, and clears everything it does when exiting to a directory above or outside of `mam4xx`. Then run `direnv allow` to enable the functionality. For the `conda` people, I've also used this tool to auto-load a pre-configured `conda` environment since the `.envrc` is basically a bash script with a few bells and whistles added. In case anyone's curious, here's a [fancier version](dot_envrc.md) that I use to setup an environment for a jupyter notebook (no promises on robustness of this script 🙂).
