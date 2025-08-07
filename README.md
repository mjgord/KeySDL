# KeySDL

KeySDL is a pipeline to fit GLV or replicator models of microbial systems from observations assumed to be at steady state. The research paper detailing the motivation, use cases, and validation of KeySDL is available as a [preprint].

This repository contains the minimal standalone implementation of KeySDL.
For adaptation to more complex use cases, see the repository containing the full scripts used in the KeySDL paper [here].

## Usage:

A python3 environment with numpy, pandas, and (py)torch is required. For convenience, the environment used to test KeySDL.py is provided in KeySDL.yml and can be installed using conda as follows:

```
conda create -f KeySDL.yml
```

To use KeySDL:

```
conda activate KeySDL_env
python3 KeySDL.py test_data.csv
```

Running these commands should exactly replicate the contents of the `KeySDL_out` directory in this repository on most systems.

Intel-based MacOS computers cannot run the version of Torch included in KeySDL.py. Older versions may work, but the environment will need to be manually constructed.

## Inputs:

```
python3 KeySDL.py data.csv [out_dir] [compositional]
```

data.csv [string]: a csv file where each column is a microbe and each row is a steady-state observation.
The header row must contain microbe names, and there must not be an index column.

out_dir [string]: path to folder for output files. If not provided, KeySDL creates a folder called "KeySDL_out" in the current working directory.

compositional [boolean]: whether to model as compositional replicator or absolute GLV dynamics.
Default value of True is appropriate for all sequencing datasets without quantification

Output files:

A.csv: GLV/replicator interactions matrix A.
r.csv: Growth rates r (all ones for replicator system).
dropout_keystones.csv: Simulated impact on removal for each microbe.
simulated_abundance.csv: Simulated mean relative abundance of each microbe across 500 random steady states.

KeySDL.py will also print the self-consistency score S<sub>sc</sub>. Scores above 0.7 are consistent with but not indicative of a system governed by GLV dynamics, while scores below this range would indicate that GLV dynamics are not a primary driver of system hehavior. Full details of this metric and its characterization are available in the paper.

## Citing KeySDL

We ask that if you use KeySDL, you cite the manuscript describing KeySDL. This is currently available as a [preprint]. We are submitting the manuscript for publication and this section will be updated when there is a published version.
