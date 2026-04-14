configfile: "config/config.yaml"

include: "workflow/rules/cv_orchestrator.smk"

rule all:
    input:
        rules.all_cv.input
