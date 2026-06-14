from trader_optimizer.regime_detector_specs import detector_spec_manifest


def test_detector_spec_manifest_links_corpus_wisdom_to_outputs() -> None:
    manifest = detector_spec_manifest()

    assert manifest["schema"] == "regime_detector_specs.v1"
    assert manifest["advancedParametersNote"].endswith("advanced_regime_parameters.md")
    specs = {spec["detectorId"]: spec for spec in manifest["specs"]}
    assert "direction_slope_persistence" in specs
    assert "changePointConfidence" in specs["momentum_distribution_change"]["outputFields"]
    assert (
        specs["options_surface"]["implementationStatus"]
        == "not_loaded_until_options_seed_set_and_data_are_ready"
    )
    assert any(
        "Regime Changes and Financial Markets" == paper["title"]
        for paper in specs["direction_slope_persistence"]["papers"]
    )
    assert all(spec["leakageGuardrail"] for spec in manifest["specs"])
