import os
import time
import subprocess
import wandb
import shutil
import argparse
import pandas as pd
from pathlib import Path

def run_command(command):
    """Utility function to run a shell command and print its output."""
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate() 
    if process.returncode != 0:
        raise Exception(f"Command failed with error: {stderr.decode('utf-8')}")
    print(stdout.decode('utf-8'))
    return stdout

def check_slurm_jobs_status(job_ids):
    # Convert list of job IDs to a comma-separated string
    job_ids_str = ','.join(map(str, job_ids))
    
    # Check running or queued jobs using squeue
    squeue_command = f"squeue --jobs={job_ids_str} --noheader --format=%T"
    squeue_output = subprocess.getoutput(squeue_command).splitlines()
    
    return [] if not squeue_output else [status for status in squeue_output]

def main(args):
    start_from_step = 1
    os.chdir(args.base_dir)
    
    # Check that the experiment folder exists
    if not os.path.exists(args.experiment_folder):
        raise ValueError(f"Experiment folder {args.experiment_folder} does not exist")
    
    # Step 1: Generate EHR embeddings
    if start_from_step <= 1:
        tasks_to_instructions = "" if args.task_to_instructions == "" else f"--task_to_instructions {args.task_to_instructions}"
        feature_command = f"""
        python {args.base_dir}/ehrshot/4_generate_llm_features.py \
        --path_to_database {args.path_to_database} \
        --path_to_labels_dir {args.path_to_labels_dir} \
        --num_threads {args.num_threads} \
        --is_force_refresh \
        --path_to_features_dir {args.experiment_folder} \
        --text_encoder {args.text_encoder} \
        --serialization_strategy {args.serialization_strategy} \
        {tasks_to_instructions}
        """
        run_command(feature_command)

    # Change into scripts directory
    os.chdir(f"{args.base_dir}/ehrshot/bash_scripts")

    # Step 2: Evaluate embeddings on different tasks
    if start_from_step <= 2:
        eval_script = f"{args.base_dir}/ehrshot/bash_scripts/7_eval.sh"
        eval_command = f"""bash {eval_script} \
        --is_use_slurm \
        --path_to_features_dir {args.experiment_folder} \
        --path_to_output_dir {args.experiment_folder}
        """
        stdout = run_command(eval_command)

        # Step 2.1: Check for job completion
        job_ids = [int(line.split()[-1]) for line in stdout.decode('utf-8').split("\n") if "Submitted batch job" in line]
        print(f"Manual kill command: scancel {' '.join(map(str, job_ids))}")
        status = check_slurm_jobs_status(job_ids)
        while status:
            print(f"Waiting for eval jobs to complete (current status: {[s[0:3] for s in status]})...")
            time.sleep(15)
            status = check_slurm_jobs_status(job_ids)
            
        # Ensure that all subfolder starting with "guo_", "new_", "lab_", "chexpert" have a all_results.csv file
        tasks = ["guo_", "new_", "lab_", "chexpert"]
        for subfolder in os.listdir(args.experiment_folder):
            if any([subfolder.startswith(task) for task in tasks]):
                results_file = os.path.join(args.experiment_folder, subfolder, 'all_results.csv')
                if not os.path.exists(results_file):
                    raise ValueError(f"Results file {results_file} does not exist")

    # Step 3: Calculate metrics
    if start_from_step <= 3:
        calculate_metrics_command = f"""
        python {args.base_dir}/ehrshot/10_cis.py \
        --path_to_results_dir {args.experiment_folder} \
        --path_to_output_file {os.path.join(args.experiment_folder, 'all_results.csv')}
        """
        run_command(calculate_metrics_command)

    # Step 4: Log results to wandb
    experiment_name = Path(args.experiment_folder).name.split("_202")[0]
    # Get experiment identifier from name of folder above
    experiment_id = Path(args.experiment_folder).parent.name
    wandb.init(project=f"ehrshot-{experiment_id}", name=experiment_name)
    results_path = os.path.join(args.experiment_folder, 'all_results.csv')
    # Read results and log to wandb
    results = pd.read_csv(results_path)
    # TODO: Use different logging for experiment settings (experimental_setup)
    experimental_setup = {
        "text_encoder": args.text_encoder,
        "serialization_strategy": args.serialization_strategy,
        "task_to_instructions": False if args.task_to_instructions == "" else True
    }
    performance_results = {f"{row['subtask']}_{row['score']}": row['est'] for _, row in results.iterrows()}
    wandb.log({**experimental_setup, **performance_results})
    
    # Upload snapshot of key files to wandb
    files_to_upload = [
        f'{args.base_dir}/ehrshot/bash_scripts/4_generate_llm_features.sh',
        f'{args.base_dir}/ehrshot/serialization/text_encoder.py',
        f'{args.base_dir}/ehrshot/serialization/ehr_serializer.py'
    ]
    if args.task_to_instructions:
        files_to_upload.append(args.task_to_instructions)
    for file_path in files_to_upload:
        shutil.copy(file_path, args.experiment_folder)
        wandb.save(os.path.join(args.experiment_folder, os.path.basename(file_path)))

    print("Experiment completed and results uploaded to wandb.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EHRShot experiments")
    parser.add_argument("--base_dir", required=True, help="Base directory")
    parser.add_argument("--experiment_folder", required=True, help="Path to the experiment folder")
    parser.add_argument("--path_to_database", required=True, help="Path to the database")
    parser.add_argument("--path_to_labels_dir", required=True, help="Path to the labels directory")
    parser.add_argument("--path_to_split_csv", required=True, help="Path to the CSV file containing splits")
    parser.add_argument("--num_threads", type=int, default=20, help="Number of threads")
    parser.add_argument("--text_encoder", required=True, help="Text encoder to be used")
    parser.add_argument("--serialization_strategy", required=True, help="Serialization strategy to be used")
    parser.add_argument( "--task_to_instructions", type=str, default="", help="Path to task to instructions file")
    
    args = parser.parse_args()
    main(args)