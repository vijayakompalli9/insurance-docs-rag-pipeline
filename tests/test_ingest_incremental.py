from rag_pipeline.embeddings import LocalHashingEmbedder
from rag_pipeline.ingest import run_ingest


class CountingEmbedder(LocalHashingEmbedder):
    def __init__(self) -> None:
        super().__init__(2048)
        self.calls: list[int] = []

    def embed(self, texts):
        self.calls.append(len(texts))
        return super().embed(texts)


def test_second_run_skips_all_unchanged_chunks(settings):
    emb = CountingEmbedder()
    first = run_ingest(settings, emb)
    assert first.embedded == first.chunks_total > 40
    second = run_ingest(settings, emb)
    assert second.embedded == 0 and second.reused == first.chunks_total
    assert emb.calls == [first.chunks_total]  # no embed call on the second run


def test_only_edited_chunk_is_reembedded_and_removed_doc_is_dropped(settings, corpus_copy):
    emb = CountingEmbedder()
    first = run_ingest(settings, emb)
    path = corpus_copy / "ho-liability.md"
    edited = path.read_text().replace("5,000 dollars per person", "6,000 dollars per person")
    path.write_text(edited)
    (corpus_copy / "uw-referral-authority.md").unlink()
    report = run_ingest(settings, emb)
    assert report.embedded == 1
    assert report.removed == 4
    assert report.chunks_total == first.chunks_total - 4


def test_embedder_change_forces_full_rebuild(settings):
    run_ingest(settings, CountingEmbedder())
    report = run_ingest(settings, LocalHashingEmbedder(4096))
    assert report.full_rebuild and report.reused == 0
