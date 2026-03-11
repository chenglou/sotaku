import contextlib
import os
import sys


class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_stdout_to_log(output_dir, model_path, suffix):
    if not output_dir:
        yield None
        return

    model_name = os.path.basename(model_path).replace(".pt", "")
    log_path = os.path.join(output_dir, f"{model_name}_{suffix}.log")

    with open(log_path, "w") as log_file:
        tee = TeeStdout(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee):
            yield log_path


def start_stdout_tee(output_dir, model_path, suffix):
    if not output_dir:
        return lambda: None

    model_name = os.path.basename(model_path).replace(".pt", "")
    log_path = os.path.join(output_dir, f"{model_name}_{suffix}.log")
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = TeeStdout(original_stdout, log_file)

    def restore():
        sys.stdout = original_stdout
        log_file.close()

    return restore
