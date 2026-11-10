"""The sample information MassHunter writes into a .d run, AcqData/sample_info.xml: the sample's name, type and
position, the method, the acquisition time and the rest, as the worklist had them. No Qt."""

from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

SAMPLE_INFO = Path("AcqData") / "sample_info.xml"

# the fields a list of runs shows of each, by the names MassHunter gives them
SUMMARY_FIELDS = {"Sample Name": "sample_name", "Sample Type": "sample_type", "Sample Position": "position", "AcqTime": "acquired_at"}


def read_sample_info(run: str | Path) -> dict[str, str]:
    """Of each field of a .d run's sample information, its value, by name ("Sample Name", "AcqTime", ...), in the
    file's order and without the fields that have none; {} if the run has none or it cannot be read."""
    try:
        root = ElementTree.parse(Path(run) / SAMPLE_INFO).getroot()
    except (OSError, ElementTree.ParseError):
        return {}
    fields = {}
    for field in root.iter("Field"):
        name, value = (field.findtext("Name") or "").strip(), (field.findtext("Value") or "").strip()
        if name and value:
            fields[name] = value
    return fields


def summary(fields: dict[str, str]) -> dict[str, str]:
    """The fields a list of runs shows, by its names (sample_name, sample_type, position, acquired_at)."""
    return {key: fields[name] for name, key in SUMMARY_FIELDS.items() if name in fields}


def acquired_at(fields: dict[str, str]) -> datetime | None:
    """When the run was acquired (AcqTime, with its time zone), or None if not told or not readable."""
    try:
        return datetime.fromisoformat(fields["AcqTime"])
    except (KeyError, ValueError):
        return None
