# 2048
2048 machine learning project for Stanford CS238: Decision Making Under Uncertainty

## Development
Create a virtual environment in the root directory of the project.
```
python -m venv .venv
```

Make sure to activate your virtual environment before developing.

Linux & macOS:
```
source .venv/bin/activate
```

Windows:
```
.venv/Scripts/activate
```

Make sure to install the required dependencies.
```
pip install -r requirements.txt
```

## Data generation
To build the C++ code, make a `./build` directory and `cd` into it.
```
mkdir build && cd build
```

If you've recently changed the `CMakeLists.txt` file or haven't built it before, make sure to configure.
```
cmake ..
```

If you've recently changed some code or haven't built before, build the program.
```
cmake --build .
```

To run the executable, use the following command.
```
cd Debug && ./Learning2048 && cd ..
```