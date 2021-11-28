from constants import DataDir

from util import get_clean_dataframe_from_file


def main() -> None:
    df = get_clean_dataframe_from_file(DataDir.all_tables)


if __name__ == "__main__":
    main()
