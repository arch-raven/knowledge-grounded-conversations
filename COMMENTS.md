## Tracking code base changes
- Most changes in the code can be tracked via git, install [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens) extension in VS Code. This tool helps to see the author of any line in codebase (author as well as commit history)

## Usage

```bash
conda activate multigen-new
CUDA_VISIBLE_DEVICES=4,5, python main.py
```

## Following README.md

The original instructions are present in the `README.md` file. 

- Original authors had`$DATA_TYPE` one of `anlg`, `eg`, `story`. Use `wizard` to for using Wizard of wikipedia
- For #Training section, the one change is that instead of specifying all arguments in command line, I specify them inside the `main.py` file. Changing `main(args.split())` -> `main()` will revert the change. 


## Conda environment

- `multigen-new` is the current working environment for the repo
- There is `environment.yml` & `requirements.txt` to reproduce `multigen-new`
- Conda-pack
    - Exactly replicating `multigen-new` can be hard even from the given environment files, so there is `multigen.tar.gz` which is a packed tar file of `/home1/deeksha/anaconda3/envs/multigen` folder.
    - Follow `On target machine` instructions from this [doc](https://conda.github.io/conda-pack/#commandline-usage)