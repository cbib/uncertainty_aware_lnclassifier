rule gres_bug:
    output:
        touch("results/gres_bug_done")
    resources:
        slurm_partition="gpu",
        gres="gpu:nvidia_h100_nvl_1g.12gb:1",
    shell:
        """
        echo "Testing gres resource allocation"
        touch {output}
        """
