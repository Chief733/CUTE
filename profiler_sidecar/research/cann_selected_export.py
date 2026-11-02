"""Experiment: vendor parsing, exporting only task/communication/step timelines."""
from pathlib import Path
import sys

analysis=Path('/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis')
sys.path.insert(0,str(analysis))
from msinterface.msprof_entrance import MsprofEntrance
from msinterface.msprof_export import ExportCommand
from msinterface.msprof_timeline import MsprofTimeline
from common_func.msprof_common import MsProfCommonConstant
from profiling_bean.prof_enum.export_data_type import ExportDataType

selected={ExportDataType.TASK_TIME,ExportDataType.HCCL,ExportDataType.STEP_TRACE,ExportDataType.MSPROF}
ExportCommand.EXPORT_HANDLE_MAP[MsProfCommonConstant.TIMELINE]=[
    row for row in ExportCommand.EXPORT_HANDLE_MAP[MsProfCommonConstant.TIMELINE]
    if row['export_type'] in selected]
MsprofTimeline.CONNECT_LIST=[]
job=sys.argv[1]
sys.argv=['msprof.py','export','timeline','-dir',job]
MsprofEntrance().main()
