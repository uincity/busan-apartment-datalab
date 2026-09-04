from src.collect_rent import _xml_items


def test_xml_items_parses_rent_response():
    content = b"""<?xml version='1.0' encoding='UTF-8'?>
    <response><header><resultCode>000</resultCode><resultMsg>OK</resultMsg></header>
    <body><items><item><aptNm>APT</aptNm><deposit>30,000</deposit><monthlyRent>0</monthlyRent></item></items>
    <totalCount>1</totalCount></body></response>"""

    rows, total = _xml_items(content)

    assert rows == [{"aptNm": "APT", "deposit": "30,000", "monthlyRent": "0"}]
    assert total == 1
