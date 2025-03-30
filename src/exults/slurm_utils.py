import re
import logging
import subprocess
import os
import json
from pathlib import Path
import pandas as pd


def read_job_id(job_id: str):
    """Reads formated job id and returns a tuple with format:
    (main_id, [array_index, [final_array_index])
    """
    pattern = r"(?P<main_id>\d+)_\[(?P<arrays>(\d+(-\d+)?(,)?)+)(\%\d+)?\]"
    match = re.search(pattern, job_id)
    if match is not None:
        main = match.group("main_id")
        array_ranges = match.group("arrays").split(",")
        return [tuple([main] + array_range.split("-")) for array_range in array_ranges]
    else:
        main_id, *array_id = job_id.split("_", 1)
        if not array_id:
            return [(main_id,)]
        # there is an array
        array_num = str(
            int(array_id[0])
        )  # trying to cast to int to make sure we understand
        return [(main_id, array_num)]


def read_info(string):
    """Reads the output of sacct and returns a dictionary containing main information"""
    if not isinstance(string, str):
        string = string.decode()
    lines = string.splitlines()
    if len(lines) < 2:
        return {}  # one job id does not exist (yet)
    names = lines[0].split("|")
    # read all lines
    all_stats = {}
    for line in lines[1:]:
        stats = {x: y.strip() for x, y in zip(names, line.split("|"))}
        job_id = stats["JobID"]
        if not job_id or "." in job_id:
            continue
        try:
            multi_split_job_id = read_job_id(job_id)
        except Exception as e:
            # Array id are sometimes displayed with weird chars
            logging.warn(
                f"Could not interpret {job_id} correctly (please open an issue):\n{e}",
                DeprecationWarning,
            )
            continue
        for split_job_id in multi_split_job_id:
            all_stats["_".join(split_job_id[:2])] = (
                stats  # this works for simple jobs, or job array unique instance
            )
            # then, deal with ranges:
            if len(split_job_id) >= 3:
                for index in range(int(split_job_id[1]), int(split_job_id[2]) + 1):
                    all_stats[f"{split_job_id[0]}_{index}"] = stats
    return all_stats


def get_job_info(job_ids):
    if len(job_ids) == 0:
        return {}
    command = ["sacct", "-o", "JobID,State,NodeList", "--parsable2"]
    for jid in job_ids:
        command.extend(["-j", str(jid)])
    output = subprocess.run(command, capture_output=True, check=True)
    job_info = read_info(output.stdout)
    return job_info


import threading
import time


class JobsWatcher:
    def fetch_job_pool(self):
        # read from file
        with open(self.data_path, "r") as f:
            pool = json.load(f)
        return pool

    def save_job_pool(self, pool):
        with open(self.data_path, "w") as f:
            json.dump(pool, f)

    def track_jobs(self, jobs):
        # add to file
        pool = self.fetch_job_pool()
        pool_job_ids = set([job["job_id"] for job in pool])
        for job in jobs:
            if job["job_id"] in pool_job_ids:
                print(f'Already tracking {job["job_id"]}')
            else:
                pool.append(job)
        self.save_job_pool(pool)

    def remove_jobs(self, jobs):
        pool = self.fetch_job_pool()
        jobs_to_remove = set(jobs)
        pool = [job for job in pool if job["job_id"] not in jobs_to_remove]
        self.save_job_pool(pool)

    def clear_all(self):
        self.save_job_pool([])

    def add_to_requeue_pool(self, jobs):
        pool = self.fetch_job_pool()
        pool_job_ids = {job["job_id"]: job for job in pool}
        for job in jobs:
            if job["job_id"] not in pool_job_ids:
                pool.append({**job, "requeue": True})
            elif not pool_job_ids[job["job_id"]].get("requeue", False):
                pool_job_ids[job["job_id"]]["requeue"] = True
        self.save_job_pool(pool)

    def remove_from_requeue_pool(self, jobs):
        pool = self.fetch_job_pool()
        pool_job_ids = {job["job_id"]: job for job in pool}
        for job in jobs:
            if job["job_id"] not in pool_job_ids:
                print(f'{job["job_id"]} not in pool')
            else:
                pool_job_ids[job["job_id"]]["requeue"] = False
        self.save_job_pool(pool)

    def update(self):
        # polls sacct
        # calls scontrol requeue if needed
        try:
            pool = self.fetch_job_pool()
        except Exception as e:
            print(f"Error reading pool: {e}")
            raise e

        job_ids = [job["job_id"] for job in pool]
        job_info = get_job_info(job_ids)
        augmented_job_info = []
        for job in pool:
            if (
                job.get("requeue", False)
                and job_info[job["job_id"]]["State"] == "PREEMPTED"
            ):
                try:
                    subprocess.check_call(
                        ["scontrol", "requeue", job["job_id"]], timeout=60
                    )
                    print(f'Requeued {job["job_id"]}')
                except Exception as e:
                    print(f'Error requeuing {job["job_id"]}')
            augmented_job_info.append({**job, **job_info[job["job_id"]]})
        return pd.DataFrame(augmented_job_info)

    def loop(self):
        while self.live_update:
            try:
                self.update()
            except Exception as e:
                print(f"Error updating: {e}")
            time.sleep(self.poll_interval)

    def kill_loop(self):
        self.live_update = False
        self.thread.join()

    def start_loop(self):
        self.live_update = True
        self.thread = threading.Thread(target=self.loop, args=())

    def __init__(self, *, poll_interval=60, data_path="jobs_watcher.json"):
        # inits data_path if not exists
        self.data_path = Path(data_path)
        self.poll_interval = poll_interval

        if not self.data_path.exists():
            self.save_job_pool([])

        self.start_loop()


import exults.run_manager as rm


def get_last_output(cfg_path, *, output_root="results", expts_root="experiments"):
    parent_dir = Path(rm.get_run_dir_parent(cfg_path, output_root, expts_root))
    dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(parent_dir / d)]
    success_dir = [d for d in dirs if "done.out" in os.listdir(parent_dir / d)]
    max_run = max(int(d) for d in dirs)
    max_success = max(int(d) for d in success_dir)
    if max_run != max_success:
        print(
            f"Warning: latest run {max_run} of {cfg_path} is not successful. Falling back to {max_success}"
        )
    return parent_dir / str(max_success)
