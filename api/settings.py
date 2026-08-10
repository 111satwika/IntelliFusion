"""
Settings API. Every setting (top_k, use_vision, the three RAG-quality
toggles, and conversation memory) is per-session, stored via
session_store - mirroring how top_k/use_vision already lived in
st.session_state in the Streamlit build.

The RAG toggles used to be process-global module flags
(app.retrieval.crag / query_transform / app.generation.self_reflection
each held a plain module-level boolean read by every request
regardless of who set it) - fine for the old single-session Streamlit
app, but wrong once multiple browser sessions can hit this FastAPI
server concurrently: one session's toggle flip would silently change
what every OTHER session's next chat request does. They're now stored
per-session here instead, and the resolved values are passed explicitly
into cli.ask_with_vision_stream (see api/chat.py) rather than being
read back out of the old global flags.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_session_id
from session_store import get_settings, set_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_all(session_id: str = Depends(get_session_id)):
    return get_settings(session_id)


class SessionSettingsUpdate(BaseModel):
    top_k: int | None = None
    use_vision: bool | None = None
    crag_enabled: bool | None = None
    query_transform_enabled: bool | None = None
    self_rag_enabled: bool | None = None
    conversation_memory_enabled: bool | None = None


@router.post("")
def update_session_settings(body: SessionSettingsUpdate, session_id: str = Depends(get_session_id)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    return set_settings(session_id, **updates)
