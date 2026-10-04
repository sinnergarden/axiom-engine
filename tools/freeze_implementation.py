"""Build-time source binding. This script is never invoked by Runtime or UI."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'src/axiom_engine/_implementation.py'


def source_binding():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'src/axiom_engine').rglob('*.py')) if p != OUTPUT}


def implementation_ref():
    return 'sha256:' + hashlib.sha256(json.dumps(source_binding(), sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    OUTPUT.write_text('"""Generated source identity; see tools/freeze_implementation.py."""\n'
                      'IMPLEMENTATION_REF = ' + repr(implementation_ref()) + '\n')
