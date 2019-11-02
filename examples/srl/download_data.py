import os

import texar.torch as tx

DATA_URL = "https://drive.google.com/file/d/14mcYZKOEdCnjzaxXh1bzg8aQuwncRgU-/view?usp=sharing"
EMBED_URL = [
    "http://nlp.stanford.edu/data/glove.840B.300d.zip",
    "https://dada.cs.washington.edu/qasrl/data/glove_50_300_2.zip",
]


def download(url, path, delete=True):
    result = tx.data.maybe_download(url, path, extract=True)
    if delete:
        os.remove(result)  # delete the archive


def format_file(from_path: str, to_path: str, has_extra_fields: bool = True):
    with open(from_path) as f_in, open(to_path, "w") as f_out:
        document_lines = []
        for line in f_in:
            parts = line.split()
            if len(parts) > 0:
                # If ``has_extra_fields`` is False (e.g., test files):
                #   postag, parse, entity, sense do not exist
                # Orig order:   word, postag, parse, entity, sense, lemma, *pred
                # Reorder into: word, lemma, *pred
                # IDs:          0, 5 (1), *
                if has_extra_fields:
                    parts = [parts[0], parts[5]] + parts[6:]
                else:
                    parts = [parts[0], parts[1]] + parts[2:]
                if parts[0] == "#":
                    parts[0] = "$$"  # should be dollar; # messes up our reader
                document_lines.append(parts)
                continue

            n_parts = len(document_lines[0])
            for idx in range(n_parts):
                strs = [parts[idx] for parts in document_lines]
                if idx > 0 and all(any(ch in s for ch in "()*-") for s in strs):
                    # Fields with brackets
                    max_len = max(len(s) if s.endswith(")") else len(s) + 1
                                  for s in strs)
                    for parts in document_lines:
                        if parts[idx].endswith(")"):
                            parts[idx] = parts[idx].rjust(max_len)
                        else:
                            parts[idx] = (parts[idx] + " ").rjust(max_len)
                else:
                    max_len = max(len(s) for s in strs)
                    for parts in document_lines:
                        parts[idx] = parts[idx].rjust(max_len)
            for parts in document_lines:
                f_out.write("   ".join(parts) + "\n")
            f_out.write("\n")
            document_lines = []
    os.remove(from_path)


def main():
    download(DATA_URL, "data/")
    for split in ["train", "dev", "test"]:
        # Make a separate folder for each dataset split.
        os.makedirs(f"data/{split}", exist_ok=True)
        for file in os.listdir("data"):
            file_path = os.path.join("data", file)
            if os.path.isfile(file_path) and file.startswith(split):
                # The original data files do not come with extensions,
                # but they must be ".gold_conll" to be recognized by readers.
                format_file(file_path, f"data/{split}/{file}.gold_conll",
                            split != "test")

    for url in EMBED_URL:
        download(url, "embeddings/")


if __name__ == "__main__":
    main()
