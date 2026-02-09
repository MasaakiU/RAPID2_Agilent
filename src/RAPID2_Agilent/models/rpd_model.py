
import mmap
import pprint
import struct
import numpy as np
from lxml import etree
from pathlib import Path
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict, List, Tuple
from PyQt6.QtCore import pyqtSignal, QObject

from ._native.extract_raw_spectrum import decompress_intensity, build_checkpoints
from ._native.calibrate_mz import calibrate_mz

class RPDModelList(QObject):
    settings_status_changed = pyqtSignal(str, object)
    def __init__(self):
        super().__init__()
        self.rpd_models: List[RPDModel] = []  # List to hold multiple RPDModel instances

class RPDModel(QObject):
    settings_status_changed = pyqtSignal(str, object)
    def __init__(self):
        super().__init__()
        self.file_path: Path = None
        self.N_scans = 0
        self.ms_scan_info: MSScanInfo = None
        self.ms_calib_info: MSCalibInfo = None
        self.ms_profile: MSProfile = None
        self.RT_list: np.ndarray = None
        self.spectrum_param_index: int = 0  # profile or centroid, depending on data
    @property
    def AcqData_path(self) -> Path:
        return self.file_path / "AcqData"
    # load_file_related_methods
    def load_file(self, file_path: Path):
        self.file_path = file_path
        # parse MSTS
        for time_seg in etree.parse(self.AcqData_path / "MSTS.xml").getroot().findall("TimeSegment"):
            self.N_scans += int(time_seg.find("NumOfScans").text)
        # parse MSScan
        self.ms_scan_info = MSScanInfo(
            xsd_path=self.AcqData_path / "MSScan.xsd", 
            bin_path=self.AcqData_path / "MSScan.bin", 
            N_scans=self.N_scans
        )
        # parse_MSCalib
        self.ms_calib_info = MSCalibInfo(
            xml_path=self.AcqData_path / "DefaultMassCal.xml",
            bin_path=self.AcqData_path / "MSMassCal.bin", 
            N_scans=self.N_scans
        )
        # parse_MSProfile
        self.ms_profile = MSProfile(
            bin_path=self.AcqData_path / "MSProfile.bin",
        )
        self.ms_profile.set_checkpoints(
            [self.ms_scan_info.get_spectrum_info(i, self.spectrum_param_index, "SpectrumOffset") for i in range(self.N_scans)],
            [self.ms_scan_info.get_spectrum_info(i, self.spectrum_param_index, "ByteCount") for i in range(self.N_scans)]
        )
        # RT_list is extracted because it will be used frequently
        self.RT_list = np.array([
            self.ms_scan_info.get_scan_info(i, "ScanTime")
            for i in range(self.N_scans)
        ])
    ### extract data ###
    def extract_spectrum(self, scan_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract m/z and intensity arrays for a given scan index.
        """
        from time import time
        t0 = time()
        start_t, delta_t, inten_list = self.ms_profile.extract_raw_spectrum(
            offset = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "SpectrumOffset"),
            compressed_len = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "ByteCount"),
            # decompressed_len = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "UncompressedByteCount"),   # always 0
            # N_datapoints = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "PointCount"),                   # written at the head of each scan data
        )
        print(time() - t0)
        print(start_t, delta_t, len(inten_list))
        print(inten_list[:10])
        print(inten_list[-10:])
        quit()


        # t0 = time()
        # mz_list = self.ms_calib_info.calibrate_mz(
        #     scan_index, 
        #     N_datapoints = self.ms_scan_info.get_spectrum_info(scan_index, self.spectrum_param_index, "PointCount"), 
        #     calib_id = self.ms_scan_info.get_scan_info(scan_index, "CalibrationID"),
        #     start_t = start_t,
        #     delta_t = delta_t
        # )
        # print(time() - t0)
        # quit()
        # return mz_list, inten_list
    def extract_chromatogram(self, low_mz: float, high_mz: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

class MSScanInfo():
    def __init__(self, xsd_path: Path, bin_path: Path, N_scans: int):
        self.N_repeat_dict: Dict = {}
        self.scan_info: List = []
        # parse xsd file
        tree = etree.parse(xsd_path)
        namespace = tree.xpath('namespace-uri(.)')
        complextypes_dict = {}
        for complextype in tree.getroot().findall(f"{{{namespace}}}complexType"):
            complextypes_dict[complextype.get('name')] = [
                (element.get('name'), element.get('type')) 
                for element in complextype[0].findall(f"{{{namespace}}}element")
            ]
        # parse binary file
        with open(bin_path, "rb") as f:
            # fixed header
            fixed_header = struct.unpack("<22i", f.read(88))
            assert fixed_header[0] == 257, fixed_header[0]  # 257, unknown
            assert all(x == 0 for x in fixed_header[1:17])  # 16 fields of ScanRecordType from CalibrationID to NumOfActualsPerScan?
            assert fixed_header[17] == 2, fixed_header[17]  # ScanRecordType/DataDependentScanParamType?
            assert fixed_header[18] == 6, fixed_header[18]  # ScanRecordType/SpectrumParamValues?
            self.N_repeat_dict["SpectrumParamValues"] = fixed_header[19]
            assert fixed_header[20] == 2, fixed_header[20]  # ScanRecordType/DataDependentScanParamType?
            assert fixed_header[21] == 0, fixed_header[21]  # repeat count of DataDependentScanParamType?
            # skip variable header and jump to data section
            f.seek(struct.unpack("<i", f.read(4))[0])
            for _ in range(N_scans):
                self.scan_info.append(self.parse_complextype(f, complextypes_dict, "ScanRecordType"))
    def parse_complextype(self, f: BinaryIO, complextypes_dict: Dict[str, list], name: str) -> Dict[str, Any]:
        name = name.split(":", 1)[-1]  # remove namespace if exists
        out: Dict[str, Any] = {}
        for field_name, field_type in complextypes_dict[name]:
            N_repeat = self.N_repeat_dict.get(field_name, 1)
            if N_repeat == 1:
                out[field_name] = self.read_type(f, complextypes_dict, field_type)
            else:
                out[field_name] = [self.read_type(f, complextypes_dict, field_type) for _ in range(N_repeat)]
        return out
    def read_type(self, f: BinaryIO, complextypes_dict: Dict[str, list], field_type: str) -> Any:
        if field_type == 'xs:byte':
            return struct.unpack('c', f.read(1))[0]
        elif field_type == 'xs:short':
            return struct.unpack('<H', f.read(2))[0]
        elif field_type == 'xs:int':
            return struct.unpack('<I', f.read(4))[0]
        elif field_type == 'xs:long':
            return struct.unpack('<Q', f.read(8))[0]
        elif field_type == 'xs:float':
            return struct.unpack('<f', f.read(4))[0]
        elif field_type == 'xs:double':
            return struct.unpack('<d', f.read(8))[0]
        else:
            return self.parse_complextype(f, complextypes_dict, field_type)
    def __str__(self):
        return pprint.pformat(self.scan_info)
    def get_scan_info(self, scan_index: int, field: str) -> Any:
        return self.scan_info[scan_index][field]
    def get_spectrum_info(self, scan_index: int, spectrum_param_index: int, field: str) -> Any:
        return self.scan_info[scan_index]["SpectrumParamValues"][spectrum_param_index][field]

class MSCalibInfo():
    def __init__(self, xml_path: Path, bin_path: Path, N_scans: int):
        # load primary calibration info from binary file
        self.calib_info = [None] * N_scans
        with open(bin_path, "rb") as f:
            f.seek(0x48)
            length = struct.unpack("<I", f.read(4))[0]
            for i in range(N_scans):
                self.calib_info[i] = np.array(struct.unpack(f"<{length}d", f.read(length * 8)))
        # load polynomial calibration info from xml file
        self.default_calib_info = {}
        for calib in etree.parse(xml_path).getroot().findall("./DefaultCalibrations/DefaultCalibration"):
            calib_id = calib.get("DefaultCalibrationID")
            # Number="1" is for coeff and base, which we do not collect here because they varies per scan.
            # Get info for Step Number="2", which holds the polynomial calibration coefficients shared for all scans.
            step2 = calib.find("./Step[@Number='2']")
            if step2 is not None:
                assert (tech := step2.findtext("CalibrationTechnique")) == "ExternalReference", f"ID {calib_id}: Technique mismatch ({tech})"
                assert (formula := step2.findtext("CalibrationFormula")) == "Polynomial", f"ID {calib_id}: Formula mismatch ({formula})"
                self.default_calib_info[int(calib_id)] = {
                    "polynomial_flag": [char == '1' for char in reversed(format(int(step2.findtext("ValueUseFlags")), '08b'))],  # convert to binary and reverse to match the order of coefficients
                    "polynomial_coefficients":  [float(v.text) for v in step2.findall("./Values/Value")]
                }
    def calibrate_mz_python(self, scan_index: int, N_datapoints: int, calib_id: int, start_t: float, delta_t: float):
        # get variables
        coeff, base, left, right, a2, b2, c2, d2, e2, f2 = self.calib_info[scan_index]
        poly_flag = self.default_calib_info[calib_id]["polynomial_flag"]
        # t_list: linspaceより arange の方が軽いが、誤差が大きくなってしまう。
        # t_list = start_t + delta_t * np.arange(N_datapoints, dtype=np.float64)
        t_list = np.linspace(start_t, start_t + delta_t * (N_datapoints - 1), N_datapoints)
        # poly_coeffs[j] が t^j の係数（最大 7次まで）
        coefficients = (a2, b2, c2, d2, e2, f2)
        poly_coeffs = np.zeros(8, dtype=np.float64)
        coeff_idx = 0
        for j, F in enumerate(poly_flag):
            if F and coeff_idx < len(coefficients):
                poly_coeffs[j] = coefficients[coeff_idx]
                coeff_idx += 1
        t_list_clipped = np.clip(t_list, left, right)
        # Horner法で効率的に多項式計算する: 
        # poly = (((((c7*t) + c6)*t + c5)*t + c4)*t + ... + c0)
        poly = np.full_like(t_list_clipped, poly_coeffs[7])
        for k in range(6, -1, -1):
            poly = poly * t_list_clipped + poly_coeffs[k]
        return (coeff * (t_list - base))**2 - poly
    def calibrate_mz(self, scan_index: int, N_datapoints: int, calib_id: int, start_t: float, delta_t: float):
        coeff, base, left, right, a2, b2, c2, d2, e2, f2 = self.calib_info[scan_index]
        poly_flag = self.default_calib_info[calib_id]["polynomial_flag"]
        return calibrate_mz(
            coeff=coeff,
            base=base,
            left=left,
            right=right, 
            a2=a2,
            b2=b2,
            c2=c2,
            d2=d2,
            e2=e2,
            f2=f2,
            polynomial_flag=poly_flag,
            N_datapoints=N_datapoints,
            start_t=start_t,
            delta_t=delta_t
        )

@dataclass
class Checkpoint:
    """A restart state inside the token stream."""
    token_offset: int   # relative to payload start (payload = comp_block[16:])
    out_pos: int        # output bin index at this token start
    cur_size: int       # 1/2/4 (token width in bytes)

class MSProfile():
    UNPACKERS = {
        1: struct.Struct('<b'),
        2: struct.Struct('<h'),
        3: struct.Struct('<i'),
        # 4: struct.Struct('<q'),
        "dd": struct.Struct('<dd'),
    }
    SIZES = {1:1, 2:2, 3:4}#, 4:8}
    checkpoint_bin = 2048
    def __init__(self, bin_path: Path):
        """
        # 1. keep file handle open
        # 2. フィアルを下層メモリ空間にマップさせる
        #       fileno はOSが持つファイル識別子
        #       0はファイル全体をマップすることを意味する
        #       access=mmap.ACCESS_READ は読み取り専用でマップすることを意味する
        # 3. memory_viewを作成
        #       スライスがコピー無しで取れる
        #       Cython/C++で扱うときに便利
        """
        self._fh = open(bin_path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._mm_view = memoryview(self._mm)
    def set_checkpoints(self, offset_list: List[int], compressed_len_list: List[int]):
        for offset, compressed_len in zip(offset_list, compressed_len_list):
            a = self.build_checkpoints_python(offset, compressed_len, stride_bytes=2048)
            print(a)
            quit()
    def build_checkpoints_python(self, offset: int, compressed_len: int, stride_bytes=2048) -> List[Checkpoint]:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)
    def extract_raw_spectrum(self, offset: int, compressed_len: int) -> Tuple[float, float, np.ndarray]:
        return decompress_intensity(self._mm_view, offset, compressed_len)
    def decode_header(self, offset: int) -> Tuple[float, float, int, int]:
        start_t, delta_t = self.UNPACKERS["dd"].unpack_from(self._mm_view, offset)
        inten_header_4 = struct.unpack(f'<I', self._mm_view[offset+16:offset+20])[0]
        N_datapoints = inten_header_4 & 0x00FFFFFF
        assert inten_header_4 >> 24 == 144, inten_header_4 >> 24  # byte seems to be fixed to \x90 ...?
        init_zero_repeat = -self.UNPACKERS[3].unpack_from(self._mm_view, offset+20)[0]
        return start_t, delta_t, N_datapoints, init_zero_repeat
    def extract_raw_spectrum_python(self, offset: int, compressed_len: int) -> Tuple[float, float, np.ndarray]:
        start_t, delta_t, N_datapoints, cur_idx = self.decode_header(offset)    # cur_idx = init_zero_repeat
        # Initialize
        inten_list = [0] * N_datapoints
        size_switch_flag = 3
        cur_size = self.SIZES[size_switch_flag]
        cur_unpacker: struct.Struct = self.UNPACKERS[size_switch_flag]
        cur_offset = offset + 24
        while cur_offset < offset + compressed_len:
            # cur_bytes = cur_unpacker(self._mm_view[cur_offset:cur_offset+cur_size])[0]
            cur_bytes = cur_unpacker.unpack_from(self._mm_view, cur_offset)[0]
            cur_offset += cur_size
            # if positive, direct data
            if cur_bytes >= 0:
                # print(cur_bytes, cur_idx, cur_offset, len(comp_view), total_len)
                inten_list[cur_idx] = cur_bytes
                cur_idx += 1
            # if negative, zero run-length or size switch
            else:
                cur_bytes *= -1
                # zero-filling: 初期値が 0 なのでゼロ埋めは不要
                len_zero, size_switch_flag = divmod(cur_bytes, 4)
                cur_idx += len_zero
                # size switch
                cur_size = self.SIZES[size_switch_flag]
                cur_unpacker = self.UNPACKERS[size_switch_flag]
        assert cur_offset == offset + compressed_len, (cur_offset, offset, compressed_len)
        return start_t, delta_t, np.array(inten_list, dtype=np.uint32)    # max size = 4 bytes = uint32
    def close(self):
        try:
            self._mm_view.release()
        except Exception:
            pass
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()


# MSProfile らへんでチェックポイントを作って、m/z範囲指定に対して高速にアクセスできるようにしておこう。
# また、extract_raw_spectrum 自体も Cython化して高速化しよう。
