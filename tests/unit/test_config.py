from codexlens_search.config import Config


def test_config_instantiates_no_args():
    cfg = Config()
    assert cfg is not None


def test_defaults_hnsw_ef():
    cfg = Config.defaults()
    assert cfg.hnsw_ef == 150


def test_defaults_hnsw_M():
    cfg = Config.defaults()
    assert cfg.hnsw_M == 32


def test_small_hnsw_ef():
    cfg = Config.small()
    assert cfg.hnsw_ef == 50


def test_custom_instantiation():
    cfg = Config(hnsw_ef=100)
    assert cfg.hnsw_ef == 100


def test_fusion_weights_keys():
    cfg = Config()
    assert set(cfg.fusion_weights.keys()) == {"exact", "fuzzy", "vector", "graph", "symbol", "entity"}


def test_graph_weight_defaults():
    cfg = Config()
    assert cfg.graph_kind_weights["call"] == 1.5
    assert cfg.graph_kind_weights["inherit"] == 0.9
    assert cfg.graph_dir_weights["backward"] == 1.3


def test_symbol_search_enabled_default():
    cfg = Config()
    assert cfg.symbol_search_enabled is True


def test_entity_graph_enabled_default():
    cfg = Config()
    assert cfg.entity_graph_enabled is True


def test_agent_defaults():
    cfg = Config()
    assert cfg.agent_enabled is False
    assert cfg.agent_llm_model == "glm-5-turbo"
    assert cfg.agent_max_iterations == 5
