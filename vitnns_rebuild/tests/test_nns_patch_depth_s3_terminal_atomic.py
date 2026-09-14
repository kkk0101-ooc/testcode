from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time

from vitnns import nns_patch_depth_s3 as s3


def test_atomic_hardlink_publication_never_exposes_partial_json() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);terminal=root/"unit.json";temporary=root/"unit.json.tmp"
        document=s3.signed("synthetic_terminal","VALID_FEATURE",{"value":"x"*100000},producer_commit="0"*40)
        errors=[];stop=False
        def reader()->None:
            while not stop:
                if terminal.exists():
                    try: json.loads(terminal.read_text())
                    except Exception as error: errors.append(error)
        thread=threading.Thread(target=reader);thread.start()
        s3.write_json_new(temporary,document);time.sleep(0.01);temporary.link_to(terminal);s3.fsync_directory(root);temporary.unlink();s3.fsync_directory(root)
        stop=True;thread.join();assert not errors;assert json.loads(terminal.read_text())["canonical_digest"]==document["canonical_digest"]
