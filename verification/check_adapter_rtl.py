"""Require exact RTL and floating-point IP parameters for the adapter profile."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--production', type=Path, required=True)
parser.add_argument('--adapter', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
production = args.production.resolve() / 'hls'
adapter = args.adapter.resolve() / 'hls'

def rtl_files(base):
    result = {}
    for language in ('verilog', 'vhdl'):
        for path in sorted((base / 'syn' / language).rglob('*')):
            if path.is_file():
                result[path.relative_to(base / 'syn').as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def fp_parameters(base):
    # Compare the complete native XCI files, not just selected parameters.
    paths = sorted((base / 'impl/ip/hdl/ip').rglob('*.xci'))
    if len({path.name for path in paths}) != len(paths):
        raise ValueError('duplicate XCI basenames prevent an unambiguous comparison')
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}

left, right = rtl_files(production), rtl_files(adapter)
differences = [name for name in sorted(left.keys() | right.keys()) if left.get(name) != right.get(name)]
fp_left, fp_right = fp_parameters(production), fp_parameters(adapter)
kernel_left = hashlib.sha256((production / 'kernel.xml').read_bytes()).hexdigest()
kernel_right = hashlib.sha256((adapter / 'kernel.xml').read_bytes()).hexdigest()
report = {
    'production': str(production), 'adapter': str(adapter),
    'rtl_files_production': len(left), 'rtl_files_adapter': len(right),
    'rtl_differences': differences, 'production_rtl_sha256': left,
    'adapter_rtl_sha256': right,
    'floating_point_parameter_sets': len(fp_left),
    'floating_point_parameters_identical': bool(fp_left) and fp_left == fp_right,
    'production_fp_parameters': fp_left, 'adapter_fp_parameters': fp_right,
    'kernel_abi_metadata_identical': kernel_left == kernel_right,
    'production_kernel_xml_sha256': kernel_left,
    'adapter_kernel_xml_sha256': kernel_right,
    'status': 'PASS' if left and left == right and fp_left and fp_left == fp_right
              and kernel_left == kernel_right else 'FAIL',
}
destination = args.report
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: report[key] for key in (
    'status', 'rtl_files_production', 'rtl_files_adapter', 'rtl_differences',
    'floating_point_parameter_sets', 'floating_point_parameters_identical',
    'kernel_abi_metadata_identical')}, indent=2))
raise SystemExit(0 if report['status'] == 'PASS' else 1)
