from enum import Enum
import os, re
import argparse
import csv

from preprocess_training_data import MAX_EXTRACTION_TEXT_LEN

try:
    # In python version 3.10 you should import Iterable from collections.abc instead:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


class TrainDataInputLabel(Enum):
    # name:내부변수,  value:레이블
    TEXT = "Data"
    LABEL = "정답"


TRAIN_DATA_FIELD_LIST = [TrainDataInputLabel.TEXT.value, TrainDataInputLabel.LABEL.value]


def _tsv_write(
    output_dir: str,
    content: Iterable = None,
    mode: str = "w",
    encoding: str = "utf-8",
    newline: str = "",
    delimiter: str = "\t",
):
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
    with open(str(output_dir + "/merge.tsv"), mode=mode, encoding=encoding, newline=newline) as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar="", quoting=csv.QUOTE_NONE)
        # writer = csv.writer(f)
        writer.writerow(TRAIN_DATA_FIELD_LIST)
        writer.writerows(content)
        f.close()


def _find_target_label(dirpath: str, label_tuple_list: list) -> str:
    """tsv(또는 csv) 파일 쓰기

    Arguments:
        dirpath: directory path
        label_tuple_list: list of label tuple (label base directory, label)

    Returns:
        target_label: target label for dirpath if found. Otherwise, returns None.
    """
    for label_base_dir, label in label_tuple_list:
        if dirpath.startswith(label_base_dir):
            return label
    return None


def merge_file(input_dir: str, args):

    training_data_list = list()
    for dirpath, dirnames, filenames in os.walk(input_dir):
        # print(f"dirpath: {dirpath}, dirnames: {dirnames}, filenames: {filenames}")
        # input_dir에 있는 폴더명을 레이블명으로 사용
        if dirpath == input_dir:
            label_tuple_list = [(os.path.join(input_dir, dirname), dirname)  for dirname in dirnames]

        for base_filename in filenames:
            full_filename = os.path.join(dirpath, base_filename)
            full_filename_root, full_filename_ext_with_dot = os.path.splitext(full_filename)
            # print(f'full_filename_root: {full_filename_root}, full_filename_ext_with_dot: {full_filename_ext_with_dot}')
            # txt 파일인지 확인
            if full_filename_ext_with_dot[1:] == "txt":
                # 사이냅 문서필터에서 저장되는 인코딩이 euc-kr
                with open(full_filename, "r", encoding="cp949") as f:
                    # 해당 txt파일의 전문을 text_file에 저장
                    text_file = f.read()
                    text_file = text_file.replace("\n", " ")
                target_label = _find_target_label(dirpath, label_tuple_list)
                training_data_list.append([text_file[0:100], target_label])

    _tsv_write(args.output_dir, training_data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marge text file in a source directory.")

    # 파라미터 처리
    # 필수
    parser.add_argument("--input_dir", type=str, required=True, help="base data source directory where all files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="base target directory which will save the text of each target file after marge.",
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    merge_file(input_dir, args)
