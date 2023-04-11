"""
Run PyTorch cpu benchmarking.
"""
import argparse
import itertools
import os
import subprocess
import sys
import yaml
import numpy

from typing import List, Tuple, Dict
from .cpu_utils import REPO_PATH, get_output_dir, get_output_json, dump_output, analyze
from ..utils import add_path

with add_path(REPO_PATH):
    from torchbenchmark.util.experiment.instantiator import list_models, TorchBenchModelConfig, \
                                                            list_devices, list_tests
    from torchbenchmark.util.experiment.metrics import TorchBenchModelMetrics

BM_NAME = 'cpu'
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def generate_model_configs(devices: List[str], tests: List[str], model_names: List[str], batch_size: int, jit: bool, extra_args: List[str]) -> List[TorchBenchModelConfig]:
    """Use the default batch size and default mode."""
    if not model_names:
        model_names = list_models()
    cfgs = itertools.product(*[devices, tests, model_names])
    result = [TorchBenchModelConfig(
        name=model_name,
        device=device,
        test=test,
        batch_size=batch_size,
        jit=jit,
        extra_args=extra_args,
        extra_env=None,
    ) for device, test, model_name in cfgs]
    return result

def dump_result_to_json(metrics, output_dir, fname):
    result = get_output_json(BM_NAME, metrics)
    dump_output(BM_NAME, result, output_dir, fname)

def validate(candidates: List[str], choices: List[str]) -> List[str]:
    """Validate the candidates provided by the user is valid"""
    for candidate in candidates:
        assert candidate in choices, f"Specified {candidate}, but not in available list: {choices}."
    return candidates

def generate_model_configs_from_yaml(yaml_file: str) -> List[TorchBenchModelConfig]:
    yaml_file_path = os.path.join(CURRENT_DIR, yaml_file)
    with open(yaml_file_path, "r") as yf:
        config_obj = yaml.safe_load(yf)
    devices = config_obj.keys()
    configs = []
    for device in devices:
        for c in config_obj[device]:
            if not c["stable"]:
                continue
            config = TorchBenchModelConfig(
                name=c["model"],
                device=device,
                test=c["test"],
                batch_size=c["batch_size"] if "batch_size" in c else None,
                jit=c["jit"] if "jit" in c else False,
                extra_args=[],
                extra_env=None,
            )
            configs.append(config)
    return configs

def parse_str_to_list(candidates):
    if isinstance(candidates, list):
        return candidates
    candidates = list(map(lambda x: x.strip(), candidates.split(",")))
    return candidates

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", default="cpu", help="Devices to run, splited by comma.")
    parser.add_argument("--test", "-t", default="eval", help="Tests to run, splited by comma.")
    parser.add_argument("--model", "-m", default=None, help="Only run the specifice models, splited by comma.")
    parser.add_argument("--batch-size", "-b", default=None, help="Run the specifice batch size.")
    parser.add_argument("--jit", action="store_true", help="Convert the models to jit mode.")
    parser.add_argument("--config", "-c", default=None, help="YAML config to specify tests to run.")
    parser.add_argument("--output", "-o", default=None, help="Output dir.")
    parser.add_argument("--launcher", action="store_true", help="Use torch.backends.xeon.run_cpu to get the peak performance on Intel(R) Xeon(R) Scalable Processors.")
    parser.add_argument("--launcher-args", default=None, help="Provide the args of torch.backends.xeon.run_cpu. See `python -m torch.backends.xeon.run_cpu --help`")
    parser.add_argument("--dryrun", action="store_true", help="Dryrun the command.")
    return parser.parse_known_args()

def run(args: List[str]):
    args, extra_args = parse_args(args)
    extra_args.remove(BM_NAME)
    if args.config:
        configs = generate_model_configs_from_yaml(args.config)
    else:
        # If not specified, use the entire model set
        if not args.model:
            args.model = list_models()
        devices = validate(parse_str_to_list(args.device), list_devices())
        tests = validate(parse_str_to_list(args.test), list_tests())
        models = validate(parse_str_to_list(args.model), list_models())
        configs = generate_model_configs(devices, tests, model_names=models, batch_size=args.batch_size, jit=args.jit, extra_args=extra_args)
    args.output = args.output if args.output else get_output_dir(BM_NAME)
    try:
        for config in configs:
            run_benchmark(config, args)
    except KeyboardInterrupt:
        print("User keyboard interrupted!")
    result_metrics = analyze(args.output)
    dump_result_to_json(result_metrics, args.output, "cpu_res.json")

def run_benchmark(config, args):
    benchmark_script = REPO_PATH.joinpath("userbenchmark", "cpu", "run_config.py")

    cmd = [sys.executable]
    if args.launcher:
        cmd.extend(["-m", "torch.backends.xeon.run_cpu"])
        if args.launcher_args:
            cmd.extend(args.launcher_args.split(" "))
    cmd.append(str(benchmark_script))
    if config.name:
        cmd.append("-m")
        cmd.append(config.name)
    if config.device:
        cmd.append("--device")
        cmd.append(config.device)
    if config.batch_size:
        cmd.append("-b")
        cmd.append(str(config.batch_size))
    if config.test:
        cmd.append("-t")
        cmd.append(config.test)    
    if config.jit:
        cmd.append("--jit")

    cmd.extend(config.extra_args)
    cmd.append("-o")
    cmd.append(args.output)
    
    print(f"Running benchmark: {cmd}")
    if not args.dryrun:
        subprocess.check_call(cmd, cwd=REPO_PATH)
