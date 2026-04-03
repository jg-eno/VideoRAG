"""
Run Human Activity Recognition + summarization with fewer LLM stages.

Uses VideoRAG ``activity_summary`` mode:
  - Fixed embedding anchor for chunk / entity / visual retrieval (no query-rewrite LLMs)
  - No per-segment LLM filter, no keyword-extraction LLM
  - Fixed system/user prompts for HAR-style output
  - Still runs vision captioning on retrieved segments + one final LLM call

Same ``working_dir`` as ``construct_graph.py``. Optional: set
``param.activity_summary_max_segments = N`` to cap caption cost.
"""

import logging
import multiprocessing
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from VideoRAG_algorithm.videorag._llm import gemini_embed_and_chat_config
from VideoRAG_algorithm.videorag import VideoRAG, QueryParam


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    param = QueryParam(mode="activity_summary")
    param.response_type = "Multiple Paragraphs"
    # param.activity_summary_max_segments = 12  # optional cap

    videorag = VideoRAG(
        llm=gemini_embed_and_chat_config,
        working_dir="./videorag-workdir-gemini-embed",
    )
    videorag.load_caption_model(debug=False)

    # ``query`` is only used if chunk retrieval with the fixed anchor returns nothing
    # (fallback embedding query). Otherwise intent is driven by fixed prompts.
    response = videorag.query(
        query="activities and events in the video",
        param=param,
    )
    print(response)
