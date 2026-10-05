"""Frozen research quotation grid from saved public exchange rule bytes."""
from ..core.contracts import Document, require

ETF_IDS = (
    "cn.etf.SSE.510300.20120528", "cn.etf.SSE.510500.20130315", "cn.etf.SSE.510880.20070118",
    "cn.etf.SSE.511010.20130325", "cn.etf.SSE.513100.20130515", "cn.etf.SSE.518880.20130729",
    "cn.etf.SZSE.159915.20111209")


def frozen_etf_grid():
    sources = [
        dict(source_key="SSE_2012", url="https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20121217_10785167.shtml",
             content_sha256="sha256:9577e9ccea663bad2071bb64fe1b8dc265aa17e91b070722a739b5c38eedb853", clause="3.4.10;3.4.11"),
        dict(source_key="SSE_2026", url="https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/10816492/files/704204728fe74fff89de4f16efda4791.docx",
             content_sha256="sha256:fc922c433438b2636cb631eab25cca405209712acbb6aaded768c45456ff8888", clause="3.3.10;3.3.11"),
        dict(source_key="SZSE_2006", url="https://www.szse.cn/disclosure/notice/general/t20060515_499577.html",
             content_sha256="sha256:8ac9e917ef0c70a9bacd2769f468bf84584f4c229790c469b47c73cba40cd31e", clause="3.3.11;3.3.12"),
        dict(source_key="SZSE_FUND_FAQ", url="https://investor.szse.cn/knowledge/fund/trade/t20171113_538865.html",
             content_sha256="sha256:11a37db782c30258adb8a4db7b3046cee1e3c218b5a260147e695476a7b613c5", clause="fund minimum quotation increment")]
    return dict(contract_version="etf_price_grid_v1", price_unit="CNY/fund unit",
        rules=[dict(security_id=s, tick_size="0.001", source_keys=
            ["SSE_2012", "SSE_2026"] if ".SSE." in s else ["SZSE_2006", "SZSE_FUND_FAQ"]) for s in ETF_IDS],
        sources=sources)


def validate_etf_grid(profile, universe, price_unit):
    grid = frozen_etf_grid()
    require(profile["price_grid"] == grid and profile["price_grid_ref"] == Document.from_dict(grid).identity,
            "unbound or changed frozen ETF price grid")
    require(price_unit == grid["price_unit"] and set(universe) <= set(ETF_IDS), "unsupported ETF unit or grid security")


def price_tick_for(profile, security):
    return next(r["tick_size"] for r in profile["price_grid"]["rules"] if r["security_id"] == security)
