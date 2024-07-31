# bioDigitalMatter Voxel Builder

## Setup

```bash
conda env update -f environment.yml
conda activate bdm_voxel_builder
pip install -e .
```

## Run main

```bash
python -m bdm_voxel_builder data/config.py
```

## Run tests

```bash
inv test
```

## Run lint

```bash
inv check
```

## Run format

```bash
inv format
```
