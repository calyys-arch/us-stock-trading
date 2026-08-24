"""Render the entry-hypothesis results report as a printable A4 PDF."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import Align, TableCellFillMode, XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "entry_hypothesis_results_report.pdf"
FONT = Path("/Library/Fonts/Arial Unicode.ttf")
DECOMPOSITION_JSON = ROOT / "backtests/reports/slippage_decomposition.json"
GATE_JSON = ROOT / "backtests/reports/entry_hypothesis_gate_report.json"
SIZING_GLOB = "sizing_wfo_*.json"
BUILT_ON = date.today().isoformat()


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _money(value, spec: str = ",.2f") -> str:
    if value is None:
        return "—"
    amount = float(value)
    return f"−${abs(amount):{spec}}" if amount < 0 else f"${amount:{spec}}"

# A4: 210 x 297 mm. Tight but printable margins.
L, R, T, B = 12.0, 12.0, 16.0, 16.0


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("CJK", size=8)
        self.set_text_color(90, 90, 90)
        self.set_y(7)
        self.cell(0, 5, "15 個進場假設：獨立七閘門結果報告", align=Align.L)
        self.set_x(self.l_margin)
        self.cell(0, 5, f"A4 列印版  ·  {BUILT_ON}  ·  單閘門 PASS 不是上線", align=Align.R)
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.2)
        y = 13
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_text_color(0, 0, 0)
        self.set_y(15)

    def footer(self) -> None:
        self.set_y(-11)
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(-9)
        self.set_font("CJK", size=8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, "官方研究 GO 是六道硬閘門的 AND。0 個研究 GO。", align=Align.L)
        self.set_x(self.l_margin)
        self.cell(0, 5, f"{self.page_no()}/{{nb}}", align=Align.R)
        self.set_text_color(0, 0, 0)

    def _full(self) -> None:
        self.set_x(self.l_margin)

    def h1(self, text: str) -> None:
        self.ln(2)
        self._full()
        self.set_font("CJK", size=16)
        self.multi_cell(0, 8, text)
        self.ln(1)

    def h2(self, text: str) -> None:
        self._need(14)
        self.ln(3)
        self._full()
        self.set_font("CJK", size=12)
        self.multi_cell(0, 6.5, text)
        self.set_draw_color(40, 40, 40)
        self.set_line_width(0.35)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def h3(self, text: str) -> None:
        self._need(12)
        self.ln(2)
        self._full()
        self.set_font("CJK", size=10.5)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    # Justified alignment stretches inter-word gaps badly on mixed CJK/ASCII
    # lines, so prose is left-aligned throughout.
    def body(self, text: str) -> None:
        self._full()
        self.set_font("CJK", size=9)
        self.multi_cell(0, 4.6, text, align=Align.L)
        self.ln(1.2)

    def bullet(self, text: str) -> None:
        self._need(8)
        self.set_font("CJK", size=9)
        self.set_x(self.l_margin)
        self.cell(4, 4.6, "•")
        self.multi_cell(self.w - self.r_margin - self.get_x(), 4.6, text, align=Align.L)
        self.ln(0.4)

    def callout(self, title: str, body: str) -> None:
        self._need(22)
        w = self.w - self.l_margin - self.r_margin
        x, y = self.l_margin, self.get_y()
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(60, 60, 60)
        self.set_line_width(0.4)
        # height estimated after text
        self.set_font("CJK", size=9)
        title_h = 5.5
        self.set_xy(x + 3, y + 2)
        self.multi_cell(w - 6, title_h, title, align=Align.L)
        y2 = self.get_y()
        self.set_xy(x + 3, y2)
        self.multi_cell(w - 6, 4.4, body, align=Align.L)
        y3 = self.get_y() + 2
        self.rect(x, y, w, y3 - y, style="DF")
        # redraw text on top of fill
        self.set_xy(x + 3, y + 2)
        self.set_font("CJK", size=9)
        self.multi_cell(w - 6, title_h, title, align=Align.L)
        self.set_xy(x + 3, self.get_y())
        self.multi_cell(w - 6, 4.4, body, align=Align.L)
        self.set_y(y3 + 2)

    def _need(self, h: float) -> None:
        if self.get_y() + h > self.h - self.b_margin:
            self.add_page()

    def table(self, headers: list[str], rows: list[list[str]], col_weights: list[float] | None = None) -> None:
        usable = self.w - self.l_margin - self.r_margin
        n = len(headers)
        if col_weights is None:
            widths = [usable / n] * n
        else:
            s = sum(col_weights)
            widths = [usable * w / s for w in col_weights]
        self._need(16)
        self.set_font("CJK", size=7.2)
        with self._table_ctx(widths) as table:
            head = table.row()
            for cell in headers:
                head.cell(cell, align=Align.C)
            self.set_font("CJK", size=7.0)
            for row in rows:
                r = table.row()
                for i, cell in enumerate(row):
                    align = Align.L if i == 0 else Align.C
                    if i == len(row) - 1:
                        align = Align.L
                    r.cell(str(cell), align=align)
        self.ln(2)

    def _table_ctx(self, widths: list[float]):
        return super().table(
            col_widths=widths,
            line_height=4.15,
            markdown=False,
            text_align=Align.L,
            borders_layout="ALL",
            cell_fill_color=(248, 248, 248),
            cell_fill_mode=TableCellFillMode.ROWS,
            first_row_as_headings=True,
            padding=1.0,
            gutter_height=0,
            repeat_headings=1,
        )


def _section_decomposition(pdf: "ReportPDF") -> None:
    """Zero-slippage replay per cell, from scripts/run_slippage_decomposition.py.

    Rendered from the JSON rather than transcribed so the PDF cannot drift from
    the numbers it cites.
    """
    rows = [r for r in (_load(DECOMPOSITION_JSON).get("results") or []) if not r.get("error")]
    if not rows:
        return
    rows.sort(key=lambda r: (int(r.get("chart_minutes") or 0),
                             -(r.get("edge_cost_ratio") or -99)))
    slip = sum(float(r.get("slippage") or 0.0) for r in rows)
    comm = sum(float(r.get("commission") or 0.0) for r in rows)
    total = slip + comm
    flipped = sum(1 for r in rows
                  if (r.get("pf_pre_cost") or 0) >= 1.0 and (r.get("pf_normal") or 0) < 1.0)

    pdf.h2("9. 成本拆解：關掉滑價之後還剩什麼")
    pdf.body(
        "多格訊號當初以「profit_factor_gross 低於 1，沒有毛邊緣，成本工程救不了」為由退役。"
        "那個推論建立在誤讀上：IntradayTrade.gross_pnl 由已經過 _slippage_price 的成交價算出，"
        "costs 只含佣金，所以 profit_factor_gross 是 pre-commission，不是 pre-cost。"
        "下表用凍結參數、同一份棒資料各重播兩次（正常成本／完全零成本）量出真正的稅前邊緣。"
    )
    pdf.table(
        ["格子", "圖", "筆數", "成本後 PF", "稅前 PF", "每筆邊緣", "每筆成本", "倍數"],
        [
            [
                r["cell"], f"{r.get('chart_minutes')}m", str(r.get("n_trades_zero")),
                f"{float(r.get('pf_normal') or 0):.3f}",
                "—" if r.get("pf_pre_cost") is None else f"{float(r['pf_pre_cost']):.3f}",
                _money(r.get("edge_per_trade")), _money(r.get("cost_per_trade")),
                f"{float(r.get('edge_cost_ratio') or 0):.2f}",
            ]
            for r in rows
        ],
        [1.6, 0.5, 0.7, 0.9, 0.8, 0.9, 0.9, 0.6],
    )
    pdf.bullet(
        f"滑價佔總成本 {(slip / total if total else 0):.1%}"
        f"（滑價 {_money(slip, ',.0f')}／佣金 {_money(comm, ',.0f')}）。"
        "成本優化的槓桿幾乎全在成交價，調佣金方案沒有意義。"
    )
    pdf.bullet(
        f"{flipped}／{len(rows)} 格在關掉成本後 PF 翻正。"
        "「沒有毛邊緣」對這些格子是錯的退役理由，正確說法是「邊緣小於執行成本」。"
    )
    pdf.bullet(
        "但沒有一格因此得救：每筆成本穩定落在 $150–270、與週期幾乎無關，"
        "而 1m 訊號每筆邊緣只有 $3–27。要救 1m 得把成本壓到十分之一，那不是調參數的量級。"
    )
    pdf.bullet(
        "唯一倍數過 1 的是 auction_reclaim 5m（1.91），但它只有 41 筆，統計上不能證明什麼。"
        "vwap_band_fade 1m 與 obv_divergence 5m 連稅前都是負的——原退役理由對這兩格成立。"
    )
    pdf.body(
        "「倍數」= 每筆稅前邊緣 ÷ 每筆總成本。壓力閘門把滑價乘 1.5，"
        "倍數要在滑價那一塊上留餘裕才可能存活。逐格明細見 backtests/reports/slippage_decomposition.md。"
    )


def _section_superseded(pdf: "ReportPDF") -> None:
    """Cells whose imported numbers failed to reproduce and were re-run."""
    cells = _load(GATE_JSON).get("cells") or {}
    replaced = sorted(
        ((k, c["superseded"], c.get("full_window_metrics") or {})
         for k, c in cells.items() if c.get("superseded")),
        key=lambda t: t[0],
    )
    if not replaced:
        return

    pdf.h2("10. 重跑取代的匯入值")
    pdf.body(
        "下列格子原本沿用舊報告的數字，但那些數字無法用現行已進版控的程式碼重現，已用重跑結果取代。"
        "舊值保留在 entry_hypothesis_gate_report.json 的 superseded 欄位供稽核。"
    )
    pdf.table(
        ["格子", "舊來源", "舊筆數", "舊 PF", "舊淨額", "新筆數", "新 PF", "新淨額"],
        [
            [
                key, str(old.get("imported_from") or "本地"),
                str(old.get("n_trades") or "—"),
                "—" if old.get("profit_factor") is None else f"{float(old['profit_factor']):.3f}",
                _money(old.get("total_net_pnl"), ",.0f"),
                str(new.get("n_trades") or "—"),
                "—" if new.get("profit_factor") is None else f"{float(new['profit_factor']):.3f}",
                _money(new.get("total_net_pnl"), ",.0f"),
            ]
            for key, old, new in replaced
        ],
        [1.3, 2.2, 0.6, 0.6, 0.9, 0.6, 0.6, 0.9],
    )
    pdf.bullet(
        "兩格 ORB 的舊數字之所以樂觀，是因為它們沿用更舊的報告，而本該取代它們的重新驗證"
        "（2026-08-06 排入）中途死掉、從未產出。重跑後 WFO 由 3/8 掉到 0/8，"
        "OOS Sharpe 由 +0.96 / +1.41 變成 −6.24 / −5.40。"
    )
    pdf.bullet(
        "新數字有交叉驗證：orb_vwap 1m 的 WFO 獨立跑出 18,788 筆 / PF 0.636，"
        "而零成本拆解用凍結參數重播得到 18,792 筆 / PF 0.636。兩條獨立路徑收斂，離群的是舊值。"
    )
    pdf.bullet(
        "auction_reclaim 5m 的差異是程式碼漂移：訊號檔在釘版提交前未進版控，"
        "產生原數字的程式碼已不存在。重跑後 111 筆 / PF 1.198，仍卡在壓力 PF 0.970。"
    )


def _section_gate_tightening(pdf: "ReportPDF") -> None:
    """The gate set's own failure, and the fix. Documented here rather than
    left in a side report because it changes what a GO in this report means."""
    runs = []
    for path in sorted((ROOT / "backtests/reports").glob(SIZING_GLOB)):
        payload = _load(path)
        if payload.get("arms"):
            runs.append(payload)
    if not runs:
        return
    # The false positive that forced the change comes first; any later cell is
    # a control on it.
    runs.sort(key=lambda p: p.get("cell") != "auction_reclaim_5m")
    sizing = runs[0]
    base = sizing.get("baseline") or {}
    arm = (sizing.get("arms") or [])[0]

    pdf.h2("11. 閘門集合自己的失效，以及修法")
    pdf.body(
        "2026-08-22 把 wfo_go 與 monte_carlo_p5_sharpe 由 warning 升為硬閘門。"
        "起因不是理念，是一個具體的假陽性。"
    )
    pdf.body(
        f"把 {sizing.get('cell')} 的部位縮到 "
        f"{arm.get('size_factor', 0):.2f}x（risk 與名目上限同步縮放）之後，"
        "壓力閘門翻成 PASS，舊制四道硬閘門全過，管線判它 GO——"
        "而這是整場戰役第一個日內 GO。"
    )
    pdf.table(
        ["指標", f"{base.get('size_factor', 1.0):.2f}x（基準）",
         f"{arm.get('size_factor', 0):.2f}x", "性質"],
        [
            ["壓力 PF（1.5×）", f"{base.get('stress_profit_factor', 0):.3f}",
             f"{arm.get('stress_profit_factor', 0):.3f}", "樣本內"],
            ["全窗 PF", f"{base.get('profit_factor', 0):.3f}",
             f"{arm.get('profit_factor', 0):.3f}", "樣本內"],
            ["全窗淨額", _money(base.get("total_net_pnl"), ",.0f"),
             _money(arm.get("total_net_pnl"), ",.0f"), "樣本內"],
            ["WFO 折通過率", f"{base.get('wfo_pass_ratio', 0):.3f}",
             f"{arm.get('wfo_pass_ratio', 0):.3f}", "樣本外"],
            ["OOS Sharpe 平均", f"{base.get('oos_sharpe_mean', 0):+.2f}",
             f"{arm.get('oos_sharpe_mean', 0):+.2f}", "樣本外"],
            ["MC p5 Sharpe", f"{base.get('mc_p5_sharpe', 0):+.2f}",
             f"{arm.get('mc_p5_sharpe', 0):+.2f}", "樣本外"],
            ["官方決策", str(base.get("decision")), str(arm.get("decision")), "舊制"],
            ["同數字、新制", str(base.get("decision_under_current_gates")),
             str(arm.get("decision_under_current_gates")), "現行"],
        ],
        [1.3, 1.2, 1.2, 0.8],
    )
    pdf.bullet(
        "樣本內欄全部改善，樣本外欄一個都沒動：折通過率仍是 8 折過 1 折。"
        "縮小部位壓低的是期望值周圍的量測噪音，不會改變期望值的正負號。"
    )
    for other in runs[1:]:
        o_base = other.get("baseline") or {}
        o_arm = (other.get("arms") or [])[0]
        pdf.bullet(
            f"對照組 {other.get('cell')} @ {o_arm.get('size_factor', 0):.2f}x："
            f"壓力 PF {o_base.get('stress_profit_factor', 0):.3f} → "
            f"{o_arm.get('stress_profit_factor', 0):.3f}，"
            f"WFO 通過率 {o_base.get('wfo_pass_ratio', 0):.3f} → "
            f"{o_arm.get('wfo_pass_ratio', 0):.3f}，"
            f"MC p5 {o_base.get('mc_p5_sharpe', 0):+.2f} → "
            f"{o_arm.get('mc_p5_sharpe', 0):+.2f}；"
            f"舊制 {o_arm.get('decision')}、新制 "
            f"{o_arm.get('decision_under_current_gates')}。"
        )
    pdf.bullet(
        "舊制四道硬閘門裡，三道雖讀樣本外的折，但只測「活得下來」："
        "回撤只要在上限內、有成交只要大於零、彙總 PF 把八折併成一個池子讓少數幾筆大贏撐住。"
        "第四道（壓力）是唯一樣本內的，於是成為唯一可翻的搖擺票。"
    )
    pdf.bullet(
        "真正測「穩定」的 wfo_go 與 MC p5 當時都動不了決策——硬／軟的切分和資訊所在的位置是反的。"
        "升級後硬閘門六道，此變更只會讓 GO 變 NO-GO，不會反向。"
    )
    pdf.bullet(
        "本矩陣 16 格、112 個計分列的決策與閘門布林值完全未變："
        "它們原本就沒過舊制的硬 AND。這是預防性修補，不是重新評分。"
    )
    pdf.bullet(
        "同批發現的引擎錯誤：run_intraday_stress_test 逐欄搬運 engine config，"
        "把 sizing 與成本參數退回預設值，使第一次 0.75x 壓力測試實際跑在 1.00x 部位上、"
        "輸出位元級等於基準。已改為整份繼承並加上回歸測試。"
        "既有呼叫點只設定有被搬運的三個欄位，已發布數字均未受影響。"
    )


def build() -> Path:
    if not FONT.exists():
        raise FileNotFoundError(f"Missing CJK font: {FONT}")

    pdf = ReportPDF(format="A4", unit="mm", orientation="P")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=B)
    pdf.set_margins(L, T, R)
    pdf.add_font("CJK", fname=str(FONT))
    pdf.add_font("CJK", style="B", fname=str(FONT))
    pdf.set_fallback_fonts(["CJK"])
    pdf.add_page()

    # Title block
    pdf.set_font("CJK", size=8)
    pdf.set_text_color(90, 90, 90)
    pdf._full()
    pdf.cell(0, 4, "us-stock-trading  ·  研究報告  ·  不可當作 auto_execute 依據", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)
    pdf.set_font("CJK", size=18)
    pdf._full()
    pdf.multi_cell(0, 8.5, "15 個進場假設")
    pdf.set_font("CJK", size=14)
    pdf._full()
    pdf.multi_cell(0, 7, "獨立七閘門結果報告")
    pdf.ln(1)
    pdf.set_font("CJK", size=9)
    pdf.set_text_color(60, 60, 60)
    pdf._full()
    pdf.multi_cell(
        0,
        4.6,
        f"{BUILT_ON}  ·  A4 直式  ·  多數格子已有 WFO；部分自官方報告匯入，第 10 節列出被重跑取代者",
        align=Align.L,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)

    pdf.callout(
        "結論：0 個官方研究 GO。真正的瓶頸是每筆邊緣小於每筆執行成本。",
        "獨立拆開七道閘門後，活動閘門（有單、樣本、常常還有折回撤）幾乎全過；"
        "錢閘門（成本後 PF、壓力 PF、Monte Carlo p5）幾乎全不過。這就是 hard AND 一直是 NO-GO 的原因。"
        "零成本重播進一步指出死因：每筆總成本穩定落在 $150–270 且與週期幾乎無關，"
        "而 1m 訊號每筆稅前邊緣只有 $3–27。單閘門 PASS 不是上線，也不是 paper 開單依據。",
    )

    pdf.h2("0. 讀這份報告前先知道四件事")
    pdf.bullet(
        "15×7 矩陣已大致跑完，仍有 9 格沒有 WFO（absorption_breakout 5m/15m、auction_reclaim 1m、"
        "vsa_effort 1m/5m、vsa_no_demand 1m/15m、obv_divergence 1m/15m）。已完成的格子有些是新 WFO、"
        "有些是從官方報告匯入，第 10 節列出被重跑取代的那幾格。"
    )
    pdf.bullet(
        "「沒有毛邊緣」這個退役理由對多數 1m 訊號是錯的。profit_factor_gross 是 pre-commission 而非 "
        "pre-cost，真正關掉滑價後 9 格裡有 6 格 PF 翻正。但沒有一格因此得救——見第 9 節。"
    )
    pdf.bullet("閘門地板曾經改過。舊日內管線常用 PF ≥ 1.3、2× 滑價後淨損益 > 0、WFO ≥ 60%。現在獨立計分用：pooled PF ≥ 1.0、1.5× 成本後 PF ≥ 1、WFO ≥ 50%（日線 pairs 掃描仍用 60%）。舊報告若沒有 1.5× PF，標「以 2× 外推」。")
    pdf.bullet("不要把 demo / 合成跑次當證據。volume_route_strategies.json 目前視窗仍是 2025-01-02 .. 2025-05-22 合成資料、0 筆成交。vsa_effort 報告也是合成 demo。")

    pdf.h3("資料來源索引")
    pdf.table(
        ["假設", "權威報告", "視窗"],
        [
            ["vsa / obv 5m WFO", "volume_book_signals + gates_unbundled", "2025-08-01 .. 2026-07-01，top-20"],
            ["1 / 5 / 15 / 60 圖", "chart_minutes_1m_5m_15m_60m", "[2026-01-02, 2026-04-01) yaml 預設，非 WFO"],
            ["auction_reclaim 5m", "auction_reclaim_backtest_report", "2025-08-01 .. 2026-07-01"],
            ["pairs 掃描 ETF", "pairs_scan_report", "開發 2018-01-01..2024-01-01；HO 至 2026-08-01"],
            ["xsection", "strategy_review §2.2；alt_universe_frequency", "約 2018–2025；中盤 2016-11 起"],
            ["daily_range_breakout", "track2_daily_breakout_results", "WFO 21 折；HO 2025-02-01 .. 2026-07-31"],
            ["absorption_breakout", "absorption_breakout_investigation", "官方 A0 + TIGHT6 holdout"],
            ["七個已退役 1m", "strategy_review / new_signals / l2_validation", "2025-08-01 .. 2026-07-01"],
            ["狀態總表（略舊）", "signal_status.md（2026-08-13）", "之後 paper 協定以 yaml 為準"],
        ],
        [1.3, 2.2, 2.0],
    )

    pdf.h2("1. 結論（先講完）")
    pdf.body("沒有任何一個進場假設清過官方 hard AND。獨立過關數：7 道裡過幾道。壓力閘門若只有 2× 數字，2× PF < 1 就記 FAIL。")
    pdf.table(
        ["假設", "圖", "獨立", "官方", "死在哪"],
        [
            ["vsa_no_demand", "5m WFO", "4", "NO-GO", "壓力 PF 0.61；WFO 38%；MC −3.94。唯一真實 WFO 成本後 PF PASS"],
            ["obv_divergence", "5m WFO", "3", "NO-GO", "年路徑 PF 0.34、壓力 0.24、WFO 0/8、MC −22.5；年回撤 −61.6%"],
            ["auction_reclaim", "5m WFO", "3", "NO-GO", "WFO 25%；年路徑 PF 1.16 不能當 pooled；2× −$3,410 / PF 0.88"],
            ["pairs 掃描 ETF", "日線", "3", "NO-GO", "WFO 55.6%（差 60% 一線）；PF 0.73 / HO 0.86；MC −1.57"],
            ["pairs 單對 AMAT/LRCX", "日線", "2", "NO-GO", "7 年 8 筆；WFO 8%；MC −0.90；樣本 ≥40 FAIL"],
            ["xsection mega-cap", "日線", "3", "NO-GO", "WFO 31%；OOS Sharpe −0.492；MC −0.957"],
            ["xsection 中盤", "日線", "1", "NO-GO", "0/16 折；零成本 Sharpe 仍 −0.50；40bps 下近爆倉"],
            ["daily_range_breakout", "日線", "3", "NO-GO", "WFO 24%；pooled OOS −$212k；HO 正報酬是多頭 beta"],
            ["absorption A0 全宇宙", "1m", "2–3", "NO-GO", "WFO 0/7；PF 0.47；無毛利"],
            ["absorption TIGHT6 HO", "holdout", "5", "仍 NO-GO", "PF 1.53、2× +$24k，但 MC −0.73；前 5 筆吃掉淨利"],
            ["vsa_effort", "—", "0", "—", "磁碟報告是合成 demo，無真實證據"],
            ["七個已退役 1m", "1m", "大多 3", "RETIRED", "PF 0.20–0.97；WFO 多為 0%；2× 壓力大虧"],
        ],
        [1.6, 0.8, 0.5, 0.8, 2.8],
    )

    pdf.h3("決策圖（不是 WFO，三個月 yaml 預設）")
    pdf.bullet("vsa_no_demand：5m 最不差（PF 0.729）。1m 成交 4.79×、PF 掉到 0.445。15m 成交更少但 PF 更差（0.492）。")
    pdf.bullet("obv_divergence：15m 比 5m 好（PF 0.615 vs 0.397），成交剩 1/4。1m 災難（PF 0.166、−$1.92M）。")
    pdf.bullet("60m：兩邊都 0 筆。一天大約 6 根棒，訊號要 8 根才交易。結構裝不進去，不是「小時線沒邊緣」。")

    pdf.h2("2. 七道閘門各自在問什麼")
    pdf.table(
        ["閘門", "地板", "單獨在擋什麼", "官方"],
        [
            ["wfo_go", "折通過率 ≥ 50%（pairs 掃描 60%）", "走步最佳化本身有沒有穩定外推", "硬*"],
            ["oos_drawdown_within_limit", "每個 OOS 折 |DD| ≤ 25%", "有沒有單折爆倉", "硬"],
            ["has_oos_trades", "至少一個 OOS 折成交 > 0", "擋「全零報酬卻被標 GO」的空轉", "硬"],
            ["min_trades_per_oos_fold", "全部 OOS 折合計 ≥ 40", "樣本夠不夠談 PF / MC", "軟"],
            ["cost_adjusted_profit_factor", "pooled 成本後 PF ≥ 1.0", "扣完成本還有沒有正期望", "硬"],
            ["monte_carlo_p5_sharpe", "bootstrap p5 Sharpe ≥ 0", "排序／尾部一換，還是不是正的", "硬*"],
            ["stress_slippage_1.5x_pf_ge_1", "成本 1.5× 後 PF ≥ 1", "不是「淨損益 > 0」", "硬†"],
        ],
        [1.5, 1.5, 1.7, 0.5],
    )
    pdf.body("關掉 has_oos_trades 不會變出新的官方 GO：零成交仍會死在 PF 與壓力。")
    pdf.body(
        "* 2026-08-22 由 warning 升為硬閘門，理由見第 11 節。"
        "† 壓力閘門是六道硬閘門裡唯一樣本內的一道——它用 candidate_params "
        "在挑出這組參數的同一段全窗上重播。"
    )

    pdf.h2("3. 獨立七閘門總表")
    pdf.body("圖例：P = PASS，F = FAIL，— = 不是該假設的決策圖，? = 舊報告沒有用現在的地板單獨量過。壓力欄：VSA / OBV 是真正的 1.5× PF。其餘多半只有 2×；2× PF 已經 < 1 時記 F。")

    pdf.h3("3.1 日線（沒有 1m / 5m / 15m）")
    pdf.table(
        ["假設", "WFO", "DD", "有單", "樣本", "PF≥1", "MC p5", "壓力", "官方"],
        [
            ["pairs 掃描 ETF", "F 56%", "P", "P", "P 1079", "F 0.73", "F −1.57", "F 2× 0.68", "NO-GO"],
            ["pairs 單對", "F 8%", "P", "P", "F 8 筆", "F", "F −0.90", "?", "NO-GO"],
            ["pairs 低頻 z 3–4", "P 78% 開發", "P", "P", "P", "毛 1.11 / HO 0.75", "F −0.66", "HO 打臉", "NO-GO"],
            ["xsection mega", "F 31%", "P", "P", "P", "F −0.49", "F −0.96", "F", "NO-GO"],
            ["xsection 中盤", "F 0/16", "F ≈−100%", "P", "P 1772", "F", "F", "F 2×", "NO-GO"],
            ["daily_range_breakout", "F 24%", "P 11.5%", "P", "P 1989", "F −$212k", "F −1.08", "F†", "NO-GO"],
        ],
        [1.5, 1.0, 0.8, 0.5, 0.7, 1.1, 0.8, 0.8, 0.7],
    )
    pdf.body("＊低頻 pairs 的 WFO 過關是開發窗；holdout Sharpe −0.963，不能當 GO。† Holdout 2× 仍淨正（+$244k）但是 328 多 vs 102 空、半導體多頭窗，不當壓力 PASS。")

    pdf.h3("3.2 日內、決策圖可重採樣（官方數字幾乎都是 5m）")
    pdf.table(
        ["假設", "圖", "WFO", "DD", "有單", "樣本", "PF≥1", "MC", "壓力", "官方"],
        [
            ["vsa_no_demand", "5m WFO", "F 38%", "P −3.6%", "P", "P 245", "P pooled", "F −3.94", "F 1.5× 0.61", "NO-GO"],
            ["obv_divergence", "5m WFO", "F 0/8", "P 折 / 年 −62%", "P", "P 4120", "F 0.34", "F −22.5", "F 1.5× 0.24", "NO-GO"],
            ["auction_reclaim", "5m WFO", "F 25%", "P −0.9%", "P", "P* 50", "F（年 1.16）", "F −3.93", "F 2× 0.88", "NO-GO"],
            ["absorption A0", "全宇宙", "F 0/7", "?", "P", "P", "F 0.47", "F", "F 2×", "NO-GO"],
            ["absorption TIGHT6", "holdout", "—", "P", "P", "P 204", "P 1.53", "F −0.73", "P 2× +$24k", "仍 NO-GO"],
            ["vsa_effort", "任何", "—", "—", "—", "—", "—", "—", "—", "無真實 WFO"],
        ],
        [1.4, 0.8, 0.7, 1.1, 0.5, 0.7, 0.9, 0.7, 1.0, 0.8],
    )
    pdf.body("Auction 官方 min_trades 是 FAIL（當時較嚴／偏逐折）。若用現在 pooled ≥ 40，年路徑 50 筆會過；這不是 OOS 折加總的重算。1m / 15m 的獨立七閘門 WFO 還沒有。")

    pdf.h3("3.3 日內、1 分鐘原生（已退役）")
    pdf.table(
        ["假設", "WFO", "DD", "樣本", "PF", "MC", "2× 壓力", "官方"],
        [
            ["sweep_reclaim", "F 0%", "P", "P ~104k", "F 0.55", "F −31.5", "F −$53.3M", "RETIRED"],
            ["fvg_retest", "F 0%", "P", "P", "F 0.20", "F −20.5", "F −$4.35M", "RETIRED"],
            ["orb_vwap（2026-08-21 重跑）", "F 0/8 −5.40", "F −94.4%", "P 18788", "F 0.636", "F −10.3", "F 1.5×", "RETIRED"],
            ["orb_vwap_regime（重跑）", "F 0/8 −6.24", "F −90.2%", "P 15379", "F 0.643", "F −9.77", "F 1.5×", "RETIRED"],
            ["vwap_band_fade", "F 0/8", "P", "P 5452", "F 0.58–0.60", "F −12", "F −$1.7M", "RETIRED"],
            ["vp_breakout", "F 0%", "P", "P 1816", "F 0.46–0.52", "F −10", "F −$0.8M", "RETIRED"],
            ["l2_absorption", "F 0/7", "?", "P 3095", "F 0.38（毛 0.39）", "F", "F −$2.03M", "RETIRED"],
        ],
        [1.4, 1.0, 0.4, 0.8, 1.3, 0.7, 1.2, 0.8],
    )
    pdf.body(
        "orb_vwap 早期「OOS Sharpe +1.41、WFO 62%」是 gap-trap 停損反號 bug，作廢。"
        "兩格 ORB 於 2026-08-21 用已釘版的程式碼重跑，舊的匯入值已被取代，明細見第 10 節。"
    )

    pdf.h2("4. 各假設細讀")

    pdf.h3("4.1 vsa_no_demand（Williams / Coulling 無需求，5m 確認棒）")
    pdf.body("官方 5m WFO，2025-08-01 .. 2026-07-01，top-20，flat 2.0 bps。候選參數：spread_atr_max=0.4，stop_atr_mult=0.3，vol_lookback=2。")
    pdf.table(
        ["項目", "數字"],
        [
            ["WFO", "8 折，通過 3/8 = 38%，OOS Sharpe 均 −0.333"],
            ["年路徑", "245 筆，淨 −$22,309，PF 0.80，最大回撤 −3.6%"],
            ["1.5× 壓力", "245 筆，淨 −$54,193，PF 0.61"],
            ["MC p5 Sharpe", "−3.938"],
            ["硬閘門（六道）", "DD P、有單 P、成本後 PF P、壓力 F、WFO F、MC F"],
            ["軟閘門", "樣本 P、edge PF P"],
        ],
        [1.0, 3.2],
    )
    pdf.body("這是 15 個裡唯一成本後 PF 閘門為 PASS 的真實 WFO。它死在「加厚成本後期望仍負」以及「折不夠穩、尾部 bootstrap 為負」。若拿掉壓力閘門（costs_covered），會被蓋 YES——那是刻意變鬆，不建議當 paper 門檻。")

    pdf.h3("4.2 obv_divergence（Granville B-2 / S-2，session OBV）")
    pdf.table(
        ["項目", "數字"],
        [
            ["WFO", "8 折，0/8，OOS Sharpe 均 −19.156"],
            ["候選參數", "lookback_bars=10，obv_lag_frac=0.35，stop_atr_mult=0.3"],
            ["年路徑", "4120 筆，淨 −$966,938，PF 0.34，最大回撤 −61.6%"],
            ["1.5× 壓力", "淨 −$1,360,753，PF 0.24"],
            ["MC p5", "−22.539"],
            ["硬閘門（六道）", "DD P（折回撤；年路徑已破 25%）、有單 P、PF F、壓力 F、WFO F、MC F"],
        ],
        [1.0, 3.2],
    )
    pdf.body("成交密度大約是 VSA 的 17 倍，虧損密度也是。折回撤閘門 PASS、年路徑 −61.6%，代表「逐折看沒爆、整年路徑會爆」——獨立計分時不要只引用折閘門。")

    pdf.h3("4.3 auction_reclaim（Creamer 拍賣收回，5m proxy）")
    pdf.table(
        ["項目", "數字"],
        [
            ["WFO", "8 折，25%，OOS Sharpe 均 −4.167"],
            ["年路徑", "50 筆，淨 +$3,612，PF 1.16，Sharpe 0.80，回撤 −0.87%"],
            ["2× 壓力", "淨 −$3,410，PF 0.878"],
            ["MC p5", "−3.926"],
            ["官方七閘（舊壓力）", "DD P、有單 P；其餘 F（含 min_trades、PF、WFO、MC、2× 淨正）"],
        ],
        [1.2, 3.0],
    )
    pdf.body("少數年路徑賺錢的日內假設。官方仍 NO-GO，因為：(1) WFO 只有 1/4 折外推；(2) 成本後 PF 看的是 pooled OOS，不是最後一折參數的年路徑 1.16；(3) 2× 後期望翻負；(4) 50 筆對 4 個自由參數偏薄，MC p5 為負。1.5× PF 沒有單獨量——2× 已是 0.88，不能當成 PASS。")

    pdf.h3("4.4 vsa_effort")
    pdf.body("磁碟上的 vsa_effort_backtest_report 是合成 demo、0 成交。沒有真實證據可寫進總表。不要跟 vsa_no_demand 混為同一個假設。")

    pdf.h3("4.5 absorption_breakout（量能突破延續）")
    pdf.body("官方全宇宙 A0：WFO 0/7，OOS Sharpe −14.610，毛 PF 0.482，成本後 PF 0.470——跟退役的 l2_absorption 一樣是「沒有毛利」，不是成本吃掉薄邊緣。")
    pdf.body("救援（TIGHT6 + breakout_atr_mult=0.5）是整場研究最接近的一次：")
    pdf.bullet("Holdout：204 筆，成本後 PF 1.525，毛 1.578，淨 +$31,374")
    pdf.bullet("2× 滑價仍淨正 +$24,429（這場戰役唯一 holdout 壓力不過零）")
    pdf.bullet("MC p5 Sharpe −0.727：前 5 筆贏面 > 全部淨利；拿掉後 holdout 變 −$5,638")
    pdf.body("Round 3 宏觀對齊（QQQ/SPY/XLK）把開發窗毛 PF 從 0.966 拉到 1.044，但仍過不了開發閘門，且沒有新的乾淨 holdout。Paper 白名單可以繼續觀察；研究章仍是 NO-GO。")

    pdf.h3("4.6 pairs_trading")
    pdf.body("單對 AMAT/LRCX：7 年 8 筆，WFO 12 折通過率 8%，MC p5 −0.903。樣本閘門（≥40）本身就 FAIL。")
    pdf.body("掃描 66 檔 ETF、368 候選對（權威）：")
    pdf.bullet("開發：1079 筆，PF 0.734，淨 −$223,019，回撤 −22.2%，WFO 5/9 = 55.6%，MC p5 −1.575，獲利模擬機率 0.6%")
    pdf.bullet("2× 價差：PF 0.675，淨 −$283,411")
    pdf.bullet("Holdout：513 筆，PF 0.856，Sharpe −0.708，MC p5 −1.722")
    pdf.bullet("出場第二輪：動態半衰期把毛 PF 抬到 1.037，但每筆淨利約 +$31 對上不可壓縮成本約 $54")
    pdf.body("低頻 entry_z 3.0–4.0：開發 WFO 77.8%、毛 PF 1.106，holdout Sharpe −0.963、毛 PF 0.747。開發變好沒有外推。")

    pdf.h3("4.7 xsection_mean_reversion")
    pdf.body("Mega-cap 一日反轉：16 折通過 31%，OOS Sharpe −0.492（不是「正但不夠」，是平均在輸），MC p5 −0.957。1-day reversal 在這組流動性最高的名字裡已被套利掉。")
    pdf.body("中盤（S&P 150–220、假設 40bps）：0/16 折，OOS Sharpe −21.5，回撤近 −100%。零成本對照 Sharpe −0.499，與 mega-cap 基線無法區分。換宇宙沒找回邊緣，只讓高周轉更付不起價差。")

    pdf.h3("4.8 daily_range_breakout")
    pdf.bullet("WFO 5/21 = 23.8%，OOS Sharpe −0.427，折回撤 0 次違規（最差 11.5%）")
    pdf.bullet("Pooled OOS 1989 筆，淨 −$212,003")
    pdf.bullet("MC（1000 次）：p5 Sharpe −1.08，獲利機率 8.7%，PF 中位 0.92")
    pdf.bullet("Holdout 看起來很好：Sharpe +1.42，成本後倍率 18.4×，2× 仍 +$244k——但是 328 多賺 +$356k、102 空虧 −$101k。這是 2025–2026 半導體多頭，不是對稱區間突破。")

    pdf.h3("4.9 已退役的七個 1m 微結構")
    pdf.body("共同形狀：有單、樣本夠、折回撤通常過；WFO / PF / MC / 2× 壓力不過。")
    pdf.bullet("l2_absorption 最差：毛 PF 0.39，成本只解釋虧損的約 4%。方向反轉更差（0.37 → 0.35）。")
    pdf.bullet("orb_vwap 曾經看起來最好，是停損反號；修好後基線 Sharpe −6.73。")
    pdf.bullet("vp_breakout 成交最少（1816）一樣輸，低頻率沒買到品質。")
    pdf.body("這些假設不該再拿 5m/15m 重採樣當「同一假設」——or_minutes、1m 回看根數會變成另一個問題。")

    pdf.h2("5. 1m / 5m / 15m（以及為什麼 60m 是空的）")
    pdf.body("來源：chart_minutes_1m_5m_15m_60m.md。視窗 [2026-01-02, 2026-04-01)，top-20，strategy.yaml 預設，不是 WFO winner。時間停損 max(10, 2 × chart_minutes)。這不是研究 GO。")
    pdf.table(
        ["訊號", "圖", "停損", "成交", "淨 PnL", "PF 淨", "Sharpe", "最大回撤"],
        [
            ["vsa_no_demand", "1m", "10", "1691", "−$329,500", "0.445", "−16.14", "−28.2%"],
            ["vsa_no_demand", "5m", "10", "353", "−$39,063", "0.729", "−5.86", "−4.1%"],
            ["vsa_no_demand", "15m", "30", "235", "−$71,660", "0.492", "−8.75", "−6.6%"],
            ["vsa_no_demand", "60m", "120", "0", "$0", "n/a", "n/a", "結構"],
            ["obv_divergence", "1m", "10", "10633", "−$1,918,961", "0.166", "−68.85", "−85.2%"],
            ["obv_divergence", "5m", "10", "1886", "−$341,531", "0.397", "−22.65", "−28.6%"],
            ["obv_divergence", "15m", "30", "468", "−$71,131", "0.615", "−7.61", "−6.9%"],
            ["obv_divergence", "60m", "120", "0", "$0", "n/a", "n/a", "結構"],
        ],
        [1.4, 0.6, 0.5, 0.6, 1.1, 0.6, 0.7, 0.7],
    )
    pdf.h3("相對 5m")
    pdf.bullet("VSA 1m：成交 4.79×，虧損 8.44×，PF 更差。更細的圖把同一套「根數」壓成更短的牆鐘窗，訊號變吵。")
    pdf.bullet("VSA 15m：成交 0.67×，虧損反而更大，PF 更差。對 VSA 來說 5m 是三個可跑週期裡唯一較不差的。")
    pdf.bullet("OBV 1m：成交 5.64×，幾乎同比例放大虧損，PF 0.166。")
    pdf.bullet("OBV 15m：成交 0.25×，虧損 0.21×，PF 變好。OBV 的 session 累積在 15m 比較不像每根都觸發。")
    pdf.body("60m 0 筆：RTH ≈ 6.5 小時 → 一天約 6 根完整 60m 棒；_MIN_TRADE_BARS = 8，OBV 預設回看 8（WFO 還到 10），且只累今日 session。沒改訊號 API 之前，這格不該解釋成績效。")
    pdf.body("這張表不能替代 1m / 15m WFO。它只回答「同一套預設參數，換決策圖，路徑長什麼樣」。VSA 的官方 PF PASS 發生在 5m WFO 的 pooled OOS，不是這張三個月路徑（路徑 PF 0.729 < 1）。")

    pdf.h2("6. 官方 AND 對上獨立計分")
    pdf.body("以 VSA / OBV 5m 官方跑次為例（唯一完整的 1.5× PF 數字）：")
    pdf.table(
        ["組合", "含哪些閘", "VSA", "OBV"],
        [
            ["官方生存 AND", "DD + 有單 + PF + 1.5× PF", "NO（壓力）", "NO（PF + 壓力）"],
            ["只拿掉壓力", "DD + 有單 + PF", "YES", "NO"],
            ["只看活動", "有單 + 樣本", "YES", "YES"],
            ["只看回撤", "DD", "YES（折）", "YES（折）/ 年路徑 NO"],
            ["七閘全 AND", "上表七個", "NO", "NO"],
        ],
        [1.2, 1.8, 1.0, 1.4],
    )
    pdf.body("獨立計分的價值是看死在哪一道，不是找一條能蓋 GO 的子集。activity_only 幾乎永遠 YES，不能開單。")

    pdf.h2("7. 已匯入 vs 仍在跑")
    pdf.body("14 格已從官方報告匯入並獨立計分。其餘 11 格由 scripts/run_missing_entry_hypothesis_wfo.py 排隊：先 15m，再 5m，最後 1m。不會對合成視窗的 volume_route_strategies.json 做 --resume。官方 hard AND 不變。")
    pdf.table(
        ["狀態", "格子"],
        [
            ["已匯入（官方）", "日線三格；VSA/OBV/auction 5m；absorption/l2 與六個退役 1m"],
            ["排隊中", "vsa_effort 1/5/15；VSA/OBV 1m+15m；auction 1m+15m；absorption 5m+15m"],
            ["不要 --resume", "volume_route_strategies.json 仍是 2025-01-02 合成視窗"],
        ],
        [1.2, 3.0],
    )
    pdf.body("Paper / live 現況（signal_status.md 已過時）：absorption_breakout 與 pairs_trading 可在紙上 auto_execute，那是觀察協定，不是研究 GO。")

    pdf.h2("8. 一句話對照")
    pdf.bullet("錢閘門過、穩定性不過：只有 vsa_no_demand 5m（pooled PF），以及 absorption_breakout 的單次 TIGHT6 holdout（PF + 2×，但 MC 抓到尾部）。")
    pdf.bullet("年路徑賺錢、折不穩：auction_reclaim 5m（+$3.6k / PF 1.16 / WFO 25%）。")
    pdf.bullet("開發變好、holdout 打臉：低頻 pairs；orb_vwap 救援；daily_range_breakout 的 holdout 多頭。")
    pdf.bullet("換圖有訊息、仍全是負 PF：VSA 留在 5m；OBV 若還要研究，15m 比 5m/1m 不那麼糟。")
    pdf.bullet("官方要 GO：仍然需要 AND。現在沒有候選。")

    _section_decomposition(pdf)
    _section_superseded(pdf)
    _section_gate_tightening(pdf)

    pdf.ln(4)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("CJK", size=8)
    pdf.set_text_color(70, 70, 70)
    pdf._full()
    pdf.multi_cell(
        0,
        4.2,
        "本 PDF 由 scripts/render_entry_hypothesis_pdf.py 產生，頁面尺寸 ISO 216 A4（210 × 297 mm）。"
        "文字來源 backtests/reports/ 下的 entry_hypothesis_results_report.md、"
        "entry_hypothesis_gate_report.json、slippage_decomposition.json 與 sizing_wfo_*.json。"
        "列印建議：100% 實際大小、不縮放、雙面可選。",
        align=Align.L,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(path)
    print(f"bytes={path.stat().st_size}")
