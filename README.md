# Vortices Functional Renormalization Group 
An FRG project focusing on 2D bosons which provides a unified method dealing with the BKT physics.
### Installation
This project is organized by [uv](https://github.com/astral-sh/uv). 

Install uv with our standalone installers:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install with homebrew:
```bash
# With Homebrew
brew install uv
```

After the installation of uv, the project can be run using the following command,
```bash
uv sync
# 2D BEC-BCS crossover
uv run plotTc.py
# 2D BEC BKT transition
uv run plotTcBEC.py
```


