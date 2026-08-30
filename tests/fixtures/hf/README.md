# Recorded Hugging Face metadata

`*.info.json` and `*.config.json` are **verbatim** responses from the public
Hugging Face API, recorded so the whole suite runs offline. Refresh them by
re-fetching the same two URLs for the pinned SHA:

    https://huggingface.co/api/models/<repo>/revision/<sha>
    https://huggingface.co/<repo>/resolve/<sha>/config.json

`synthetic_*.json` are **not** recorded. They are hand-written stand-ins for
repositories that are not real, used where a fixture is needed but no decision
has been made -- notably the calibration dataset, whose licence is still an
open blocking question.
