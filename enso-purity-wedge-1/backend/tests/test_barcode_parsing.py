import pytest

from enso_purity.data.tcga_barcode import parse_slide_barcode, parse_aliquot_barcode


def test_parse_slide_barcode():
    s = parse_slide_barcode("TCGA-CS-5394-01A-01-TS1")
    assert s.case_id == "TCGA-CS-5394"
    assert s.sample_vial == "TCGA-CS-5394-01A"
    assert s.portion == "01"
    assert s.section_type == "TS"
    assert s.is_dx is False

def test_parse_slide_dx():
    s = parse_slide_barcode("TCGA-CS-5394-01A-01-DX1")
    assert s.is_dx is True

def test_parse_aliquot_barcode():
    a = parse_aliquot_barcode("TCGA-CS-5394-01A-11D-1234-01")
    assert a.patient_id == "TCGA-CS-5394"
    assert a.sample_vial == "TCGA-CS-5394-01A"
    assert a.portion == "11"
    assert a.analyte == "D"

def test_parse_slide_bad():
    with pytest.raises(ValueError):
        parse_slide_barcode("NOT-A-TCGA")

