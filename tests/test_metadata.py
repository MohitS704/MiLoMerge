import MiLoMerge


def test_version():
    with open("docs/source/conf.py") as conf_file:
        version = conf_file.readlines()[24].split("=")[-1].strip().replace('"','')
    
    with open("src/MiLoMerge/__init__.py") as init_file:
        version_2 = init_file.readlines()[7].split("=")[-1].strip().replace('"','')
    
    version_3 = MiLoMerge.__version__
    
    assert version == version_2 == version_3

def test_changelog():
    version = MiLoMerge.__version__
    with open("docs/source/Changelog.rst") as changelog:
        full_changelog = changelog.read()
    
    assert version in full_changelog


def test_dunder():
    necessary_funcs = [
        "mlm",
        "MergerLocal",
        "MergerNonlocal",
        "ROC_curve",
        "LOC_curve",
        "place_array_nonlocal",
        "place_event_nonlocal",
        "place_local"
    ]
    assert MiLoMerge.__all__
    assert MiLoMerge.__name__ == "MiLoMerge"
    assert MiLoMerge.__version__
    assert MiLoMerge.__author__
    assert all([i in necessary_funcs for i in MiLoMerge.__all__])
