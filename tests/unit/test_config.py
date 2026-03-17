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
    assert set(cfg.fusion_weights.keys()) == {"exact", "fuzzy", "vector", "graph"}
