from fakes import DeterministicEmbeddings, FakeChatModel, FakeReranker


def test_embeddings_are_deterministic_and_fixed_width() -> None:
    embeddings = DeterministicEmbeddings(dimension=6)

    first = embeddings.embed_query("同一文本")
    second = embeddings.embed_query("同一文本")

    assert first == second
    assert len(first) == 6


def test_chat_model_returns_responses_in_order() -> None:
    model = FakeChatModel(["first", "second"])

    assert model.invoke("q1").content == "first"
    assert model.invoke("q2").content == "second"
    assert model.calls == ["q1", "q2"]


def test_reranker_uses_explicit_scores() -> None:
    reranker = FakeReranker({"low": 0.1, "high": 0.9})

    ranked = reranker.rank("query", ["low", "high"])

    assert [item["document"] for item in ranked] == ["high", "low"]
    assert [item["rank"] for item in ranked] == [1, 2]
