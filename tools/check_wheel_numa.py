# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check real NUMA subprocess startup after installing a wheel on Linux.

Run outside the source package with numactl on PATH. No model or CUDA allocation
is needed. Both EngineCore and worker contexts must execute the installed
wrapper; silently falling back to an unbound Python process is a failure.
"""

import json
import multiprocessing
import multiprocessing.spawn
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


def child_probe(connection):
    import torch

    import vllm

    connection.send(
        {
            "vllm_path": vllm.__file__,
            "python": sys.executable,
            "affinity": sorted(os.sched_getaffinity(0)),
            "numactl_args": os.environ["_VLLM_INTERNAL_NUMACTL_ARGS"],
            "numactl_show": subprocess.check_output(
                ["numactl", "--show"], text=True
            ).strip(),
            "cuda_initialized": torch.cuda.is_initialized(),
        }
    )
    connection.close()


def main() -> None:
    import torch

    import vllm
    from vllm.config import DeviceConfig, ParallelConfig, VllmConfig
    from vllm.utils import numa_utils

    site = Path(sysconfig.get_path("purelib")).resolve()
    assert Path(vllm.__file__).resolve().is_relative_to(site), vllm.__file__
    wrapper = Path(numa_utils.__file__).with_name("numa_wrapper.sh")
    assert wrapper.is_file(), f"Installed wheel is missing {wrapper}"
    assert os.access(wrapper, os.X_OK), f"Wrapper is not executable: {wrapper}"
    cpu = min(os.sched_getaffinity(0))
    node_path = next(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    node = int(node_path.name[4:])
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            distributed_executor_backend="mp",
            numa_bind=True,
            numa_bind_nodes=[node],
            numa_bind_cpus=[str(cpu)],
        ),
    )
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    context = multiprocessing.get_context("spawn")
    original_executable = multiprocessing.spawn.get_executable()
    results = {}
    for kind in ("EngineCore", "worker"):
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=child_probe, args=(sender,))
        try:
            with numa_utils.configure_subprocess(config, 0, 0, process_kind=kind):
                assert os.fsdecode(multiprocessing.spawn.get_executable()) == str(
                    wrapper
                )
                expected_args = os.environ["_VLLM_INTERNAL_NUMACTL_ARGS"]
                process.start()
            sender.close()
            process.join(timeout=60)
            assert process.exitcode == 0, (kind, process.exitcode)
            assert receiver.poll(), f"No child result for {kind}"
            result = receiver.recv()
            assert result["vllm_path"] == vllm.__file__, result
            assert result["python"] == sys.executable, result
            assert result["numactl_args"] == expected_args, result
            assert result["affinity"] and cpu in result["affinity"], result
            if kind == "worker":
                assert result["affinity"] == [cpu], result
            assert not result["cuda_initialized"], result
            assert multiprocessing.spawn.get_executable() == original_executable
            results[kind] = {"exitcode": process.exitcode, **result}
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            sender.close()
            receiver.close()
    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            {
                "wrapper": str(wrapper),
                "mode": oct(wrapper.stat().st_mode & 0o777),
                "processes": results,
                "cuda_initialized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
