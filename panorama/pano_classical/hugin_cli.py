import subprocess
from pathlib import Path

def stitch_with_hugin(project_file, output_prefix, hugin_cmd='pto2mk'):
    """Optional batch wrapper for calibrated Hugin/PTGui-compatible workflows.
    The project file must already contain the fisheye calibration and control points.
    """
    project_file=Path(project_file); output_prefix=Path(output_prefix)
    subprocess.run([hugin_cmd,'-o',str(output_prefix),str(project_file)],check=True)
    return output_prefix
