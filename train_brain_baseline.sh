python -m sample_factory.runner.run \
--run=swarm_rl.runs.quad_multi_mix_baseline_attn_8 \
--runner=slurm \
--slurm_workdir=/home/zhaojing/slurm_output \
--experiment_suffix=slurm \
--pause_between=1 \
--slurm_gpus_per_job=1 \
--slurm_cpus_per_gpu=16 \
--slurm_sbatch_template=/home/zhaojing/sbatch_old.sh \
--slurm_print_only=False
