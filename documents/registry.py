from documents.processor import DocumentProcessor

_PROCESSORS = {}

def get_processor(session_id: str) -> DocumentProcessor:
    if session_id not in _PROCESSORS:
        _PROCESSORS[session_id] = DocumentProcessor(
            session_id=session_id
        )

    return _PROCESSORS[session_id]