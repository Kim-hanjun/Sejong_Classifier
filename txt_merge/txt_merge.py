from enum import Enum
import os, re
import argparse
import csv

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
    with open(str(output_dir + "/marge.tsv"), mode=mode, encoding=encoding, newline=newline) as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar="", quoting=csv.QUOTE_NONE)
        # writer = csv.writer(f)
        writer.writerow(TRAIN_DATA_FIELD_LIST)
        writer.writerows(content)
        f.close()


def marge_file(input_dir: str, args):

    training_data_list = list()
    for dirpath, dirnames, filenames in os.walk(input_dir):
        print(f"dirpath: {dirpath}, dirnames: {dirnames}, filenames: {filenames}")
        for base_filename in filenames:
            full_filename = os.path.join(dirpath, base_filename)
            full_filename_root, full_filename_ext_with_dot = os.path.splitext(full_filename)
            # print(f'full_filename_root: {full_filename_root}, full_filename_ext_with_dot: {full_filename_ext_with_dot}')
            # txt 파일인지 확인
            if full_filename_ext_with_dot[1:] == "txt":
                with open(full_filename, "r", encoding="cp949") as f:
                    # 해당 txt파일의 전문을 text_file에 저장
                    text_file = f.read()
                    text_file = text_file.replace("\n", " ")
                # 정답 label을 만들기 위해 폴더명에 숫자가 있는 경우 숫자 제거
                result = re.sub(r"[0-9]", "", dirpath.split("/")[-1])

                # TODO tsv에 일단 처리 결과만 저장 -> 이후에는 생성된 텍스트 파일에서 데이터 읽어 TrainDataInputLabel 형태로 처리
                training_data_list.append([text_file, result])

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
    marge_file(input_dir, args)
