from enum import Enum
import os
import subprocess
from pathlib import Path
import argparse
import csv

try:
    # In python version 3.10 you should import Iterable from collections.abc instead:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

# [[Document Extractor]]
# 추출 대상 문서 확장자
TARGET_DOC_EXTS = ['hwp', 'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'msg']

# 추출할 최대 길이
MAX_EXTRACTION_TEXT_LEN = 1000


# 분류모델 학습 데이터로 사용될 컨텐트를 저장하는 tsv 파일 포맷
class TrainDataInputLabel(Enum):
    # name:내부변수,  value:레이블
    TEXT = '텍스트'
    LABEL = '레이블'


TRAIN_DATA_FIELD_LIST = [
    TrainDataInputLabel.TEXT.value, TrainDataInputLabel.LABEL.value
]

TRAIN_DATA_CONVERTERS = {
    TrainDataInputLabel.TEXT.value: str,
    TrainDataInputLabel.LABEL.value: str
}


def _tsv_write(filename: str, content: Iterable = None, mode: str = 'w', encoding: str = 'utf-8', newline: str = '',
               delimiter: str = '\t'):
    """tsv(또는 csv) 파일 쓰기

    Arguments:
        filename: filename
        content: content to write in the file. Should be an Iterable object.
        mode: an optional string that specifies the mode in which the file is opened
        encoding: the name of the encoding used to decode or encode the file.
        newline: determines how to parse newline characters from the stream.
        delimiter: delimiter. tab for tsv and comma for csv

    Returns:
    """
    with open(filename, mode=mode, encoding=encoding, newline=newline) as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar="", quoting=csv.QUOTE_NONE)
        writer.writerow(TRAIN_DATA_FIELD_LIST)
        writer.writerows(content)


def _get_synap_doc_filter_result(cmd: str, file_path: str, save_path: str) -> str:
    """사이냅 소프트 문서필터 실행 -> 문서 txt 추출

    Arguments:
        cmd: command
        file_path: 추출할 파일 경로
        save_path: 사이냅 문서필터 실행 결과 저장 경로 ( txt 파일 )

    Returns:
        retcode: return code from command.
    """
    save_dir = os.path.dirname(save_path)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f'cmd: {cmd}, file_path: {file_path}, save_path: {save_path}, save_dir: {save_dir}')
    return subprocess.call([
        cmd,
        file_path,
        save_path
    ])


def _find_rsrc_in_dir(rsrc_basename: str, base_resource_dir: str) -> str:
    """base_resource_dir 내부를 순회하면서 rsrc_basename 파일을 찾아 해당 파일의 full path 포함한 파일명을 리턴

    Arguments:
        rsrc_basename: resource file's base name. i.e. file name + extension only (e.g. test.jpg)
        base_resource_dir: base resource directory

    Returns:
        resourcefile: resource(source) file name with its absolute path if found. Otherwise, returns None.
    """
    for dirpath, dirnames, filenames in os.walk(base_resource_dir):
        for base_filename in filenames:
            if base_filename == rsrc_basename:
                resourcefile = os.path.join(dirpath, base_filename)
                # print(f'rsrc_basename[{rsrc_basename}] found in {base_resource_dir}! resourcefile: {resourcefile}')
                return resourcefile
    return None


def preprocess_training_data(abs_base_data_source_dir: str, args):
    """abs_base_data_source_dir 내부를 순회하면서 문서 내용 추출이 필요한 대상 파일의 내용을 추출하여 tsv 파일로 저장

    Arguments:
        abs_base_data_source_dir: absolute base data source directory
        args: additional arguments

    Returns:
    """
    training_data_list = list()
    abs_base_target_dir = os.path.abspath(args.base_target_dir)
    for dirpath, dirnames, filenames in os.walk(abs_base_data_source_dir):
        print(f'dirpath: {dirpath}, dirnames: {dirnames}, filenames: {filenames}')
        for base_filename in filenames:
            full_filename = os.path.join(dirpath, base_filename)
            full_filename_root, full_filename_ext_with_dot = os.path.splitext(full_filename)
            # print(f'full_filename_root: {full_filename_root}, full_filename_ext_with_dot: {full_filename_ext_with_dot}')
            if full_filename_ext_with_dot[1:] in TARGET_DOC_EXTS:
                # 추출할 문서(동일 이름으로 텍스트 파일 중복 생성을 방지하기 위해 원본 파일의 확장자를 타겟 파일명의 일부로 사용)
                target_file_root = full_filename_root.replace(abs_base_data_source_dir, abs_base_target_dir,
                                                              1) + full_filename_ext_with_dot.replace('.', '_', 1)
                print(f'target_file_root: {target_file_root}, abs_base_data_source_dir: {abs_base_data_source_dir}, '
                      f'abs_base_target_dir: {abs_base_target_dir}')
                target_file = target_file_root + '.txt'
                result = _get_synap_doc_filter_result(args.doc_extractor_cmd, full_filename, target_file)

                # TODO tsv에 일단 처리 결과만 저장 -> 이후에는 생성된 텍스트 파일에서 데이터 읽어 TrainDataInputLabel 형태로 처리
                training_data_list.append([target_file, result])

    print(f'len(index_info_list): {len(training_data_list)}')
    # training_data_list 확인시
    for index, training_data in enumerate(training_data_list, start=1):
        print(f'index: {index}, training_data: {training_data}')

    base_output_filename = 'output_4_' + args.label + '.tsv'
    output_file = os.path.join(abs_base_target_dir, base_output_filename)
    _tsv_write(output_file, training_data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracting index info in a source directory.")

    # 파라미터 처리
    # 필수
    parser.add_argument('--base_data_source_dir', type=str, required=True,
                        help='base data source directory where all files belonging to a certain label reside.')
    parser.add_argument('--label', type=str, required=True, help='target label')
    parser.add_argument('--base_target_dir', type=str, required=True,
                        help='base target directory which will save the text of each target file after extracting.')

    # 옵션
    parser.add_argument('--doc_extractor_cmd', type=str, default="home/workspace/venv/venv/bin/synap/v4/snf_exe",
                        help='the command which extracts the content of target files.')
    args = parser.parse_args()

    abs_base_data_source_dir = os.path.abspath(args.base_data_source_dir)
    preprocess_training_data(abs_base_data_source_dir, args)
