# AutoMALA tests scripts

[Nextflow](https://www.nextflow.io/) scripts for the experiments in 

Biron-Lattes, M., Surjanovic, N.,  Syed, S., Campbell, T., & Bouchard-Côté, A. (2023). autoMALA: Locally adaptive Metropolis-adjusted Langevin algorithm. *arXiv preprint arXiv:2310.16782*.

## Usage

### Compute clusters

The `nextflow.config` file instructs Nextflow to use Apptainer with a custom Docker image, so the user only needs to run e.g.,
```bash
./nextflow run AM_scaling.nf -profile [CLUSTER_PROFILE]
```
where `[CLUSTER_PROFILE]` currently supports
- `sockeye`: UBC ARC sockeye cluster
- `CC`: Digital Research Alliance of Canada (formerly Compute Canada)

**Important**: remember to fill the `clusterOptions` variable in `nextflow.config` with your credentials.

### Local

If you have Docker installed locally, then the same Docker image can be used locally via
```bash
sudo ./nextflow run AM_scaling.nf -with-docker alexandrebouchardcote/default:0.1.4
```

Alternatively, if you have Julia and cmdstan installed locally, you can run
```bash
./nextflow run AM_scaling.nf
```
**Note**: this assumes that there is an environment variable `CMDSTAN` pointing to the directory where cmdstan lives. You can do this via
```bash
export CMDSTAN=/full/path/to/cmdstan-vx.xx
```

