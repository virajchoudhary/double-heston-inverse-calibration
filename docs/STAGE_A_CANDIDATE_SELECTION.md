# Stage A Sector-Candidate Selection

Status date: 10 August 2026

## 1. Purpose

This milestone evaluates one primary stock per sector for the later real-market representation and inverse-calibration work. The original three-date official-NSE Stage A evidence supports CIPLA, INFY, and HDFCBANK but leaves Power unresolved. The separately declared five-Wednesday Power tie-break extension resolves Power in favor of NTPC. Neither evidence block selects a final maturity or moneyness representation, changes the neural input contract, passes G2, generates final research data, trains an ANN/PINN, or performs market calibration.

## 2. Evidence dates and source replay

The evidence dates are 2026-07-01, 2026-07-15, and 2026-07-22. The inputs are the preserved official NSE CM and F&O UDiFF bhavcopy files already held under the ignored Stage A raw-data tree. No new date was downloaded and no raw observation was changed.

The existing offline Stage A command was replayed before this analysis. SHA-256 comparison covered all 20 preserved raw and existing derived files and returned `HASH_DIFF_COUNT=0`. The source screen and its eight prior derived CSVs were therefore byte-reproducible. NIFTY remained reference-only and was asserted absent from candidate evidence.

All eight candidates had one logical surface on each of the three dates. Every surface contained the same three actual stock-option expiries: 2026-07-28, 2026-08-25, and 2026-09-29, corresponding to DTE patterns 27/55/90, 13/41/76, and 6/34/69. Every expiry slice contained calls and puts, had a corresponding stock future, and passed the CM-spot/F&O-underlying consistency check.

## 3. Pairwise methodology

The comparison is pairwise within sector. It does not use a weighted composite score or count row wins mechanically. Evidence is considered in this order:

1. consistent observed central moneyness support;
2. active central-region support;
3. temporal consistency over all three dates;
4. trade-count-positive coverage;
5. OI-positive coverage;
6. last/settlement observation availability; and
7. general completeness and futures/spot eligibility.

For an expiry slice, a central node is supported when it lies inside the minimum-to-maximum observed `log(K/S)` domain for that slice. Active support applies the same domain test after retaining only rows for which the named binary activity condition is positive. This is a support test: no price, activity, strike, expiry, or node value is interpolated or extrapolated.

The five central nodes are -0.10, -0.05, 0.00, +0.05, and +0.10. Each candidate has nine expiry slices, so all-five coverage uses a denominator of 9 and node-slice coverage uses a denominator of 45. Symmetric reach is the smaller of the observed left and right log-moneyness reaches around ATM. The reported common interval is the intersection of observed support across all nine slices.

Price-positive percentages are separate from activity. Activity uses only `TtlTradgVol > 0`, `TtlNbOfTxsExctd > 0`, and `OpnIntrst > 0`. The magnitudes of `TtlTradgVol` and `TtlTrfVal` are not compared. NSE Total Traded Qty is not interpreted as contracts, lots, or shares.

## 4. Metric definitions and common eligibility evidence

- **Observed all-five:** percentage of expiry slices whose observed domain supports all five central nodes.
- **Active all-five:** percentage of expiry slices whose rows positive on at least one allowed activity flag span all five nodes.
- **Active node-slice:** percentage of the 45 central node-slice combinations supported by active rows.
- **Trade-count/OI/last rows:** overall positive-row percentage, followed by the minimum daily percentage and three-date range in percentage points.
- **Required reporting:** close, settlement, last, NSE Total Traded Qty, OI, and trade-count fields were reported on 100% of candidate rows. Positivity is reported separately.
- **Completeness:** every candidate had 3/3 surfaces, three actual expiries per surface, calls and puts in every slice, 100% option-to-futures alignment, positive CM spot, one consistent F&O underlying price, and no option/futures expiry-field mismatch.

Close was positive on 100% of rows for all eight candidates. In these files, the `TtlTradgVol > 0` and `TtlNbOfTxsExctd > 0` flags coincided row-for-row, so their reported percentages are numerically equal; this is an observation, not an assumed NSE identity.

The complete 128-row metric table, including per-node profiles and per-date values, is in `market_data_audit/stage_a/derived/candidate_pairwise_evidence.csv`.

## 5. Power: NTPC vs POWERGRID

| Metric | NTPC | POWERGRID | Better / tie | Rationale |
|---|---:|---:|---|---|
| Observed all-five slices | 100.0% | 100.0% | Tie | All nine slices support all five central nodes. |
| Common observed interval; symmetric reach | [-0.171778, +0.156796]; 0.156796 | [-0.139936, +0.131628]; 0.131628 | NTPC on reach | NTPC has broader worst-slice reach; POWERGRID has the smaller mean wing-gap, 0.025718 vs 0.042800. |
| Active all-five overall; by date | 55.6%; 33.3/66.7/66.7% | 55.6%; 33.3/66.7/66.7% | Tie | Combined active central support is identical. |
| Active node-slice | 62.2% | 62.2% | Tie | Aggregate active support at the five nodes is identical. |
| Trade-count all-five / node-slice | 55.6% / 62.2% | 55.6% / 57.8% | NTPC on node-slices | NTPC spans two additional central node-slice combinations. |
| Trade-count-positive rows: overall / daily min / range | 39.6% / 27.6% / 20.4 pp | 32.3% / 27.3% / 10.0 pp | NTPC on level and floor | POWERGRID has the smaller range, but at consistently lower coverage; range alone is not a robustness win. |
| OI-positive rows: overall / daily min / range | 55.7% / 36.7% / 33.3 pp | 47.7% / 38.1% / 20.1 pp | Mixed | NTPC has higher overall presence; POWERGRID's floor is 1.3 pp higher and its range smaller, but its later-date levels are lower. |
| Last-positive rows: overall / daily min / range | 56.3% / 37.2% / 34.1 pp | 48.5% / 38.6% / 20.1 pp | Mixed | NTPC has higher overall availability; POWERGRID's floor is 1.4 pp higher and its range smaller, but its later-date levels are lower. |
| Settlement-positive rows: overall / daily min | 94.9% / 90.7% | 96.0% / 91.5% | POWERGRID | POWERGRID is modestly higher. |
| Completeness / futures / spot integrity | 100% | 100% | Tie | Non-discriminating eligibility checks all pass. |

**Original three-date decision: UNRESOLVED.** The original POWERGRID tie-break is rejected. A smaller three-date range does not establish superior temporal robustness when the level is consistently lower: NTPC had higher trade-count-positive coverage on every original date and stronger overall trade-count, OI, and last-positive coverage, while POWERGRID had only small worst-date OI/last and settlement advantages. The higher-priority central evidence is tied or split. Neither candidate is selected from the original three-date evidence alone; the separately declared extension below supplies the additional evidence.

## 6. Healthcare/Pharma: SUNPHARMA vs CIPLA

| Metric | SUNPHARMA | CIPLA | Better / tie | Rationale |
|---|---:|---:|---|---|
| Observed all-five slices | 100.0% | 100.0% | Tie | All nine slices support all five central nodes. |
| Common observed interval; symmetric reach | [-0.153901, +0.101510]; 0.101510 | [-0.129314, +0.130644]; 0.129314 | CIPLA | CIPLA has the stronger and more symmetric common central domain; mean wing-gap is 0.037803 vs 0.080054. |
| Active all-five overall; by date | 55.6%; 33.3/66.7/66.7% | 55.6%; 33.3/66.7/66.7% | Tie | Full central-span frequency is identical. |
| Active node-slice | 60.0% | 64.4% | CIPLA | CIPLA has stronger active support at -0.10 and -0.05; +0.10 is tied. |
| Trade-count all-five / node-slice | 55.6% / 60.0% | 55.6% / 64.4% | CIPLA on node-slices | CIPLA spans more central node-slices. |
| Trade-count-positive rows: overall / daily min / range | 38.3% / 30.9% / 12.3 pp | 31.1% / 26.1% / 14.4 pp | SUNPHARMA | SUNPHARMA has broader and steadier row-level presence. |
| OI-positive rows: overall / daily min / range | 51.6% / 38.2% / 21.9 pp | 47.0% / 37.5% / 19.1 pp | Mixed | SUNPHARMA has higher coverage and floor; CIPLA has slightly lower variation. |
| Last-positive rows: overall / daily min / range | 52.8% / 39.0% / 22.5 pp | 48.0% / 38.5% / 19.5 pp | Mixed | SUNPHARMA has higher coverage and floor; CIPLA varies less. |
| Settlement-positive rows: overall / daily min | 96.1% / 92.8% | 97.9% / 95.1% | CIPLA | CIPLA is higher overall and on the worst date. |
| Completeness / futures / spot integrity | 100% | 100% | Tie | Non-discriminating eligibility checks all pass. |

**Decision: CIPLA. Confidence: moderate.** SUNPHARMA has more positive activity rows, but CIPLA wins the higher-priority central-shape evidence: its worst-slice observed domain is more symmetric and its active rows support a larger proportion of the central node-slice grid. The full active all-five rate and its daily pattern are tied. SUNPHARMA is the backup.

## 7. IT: INFY vs TCS

| Metric | INFY | TCS | Better / tie | Rationale |
|---|---:|---:|---|---|
| Observed all-five slices | 100.0% | 100.0% | Tie | All nine slices support all five central nodes. |
| Common observed interval; symmetric reach | [-0.113024, +0.173331]; 0.113024 | [-0.165615, +0.147784]; 0.147784 | TCS | TCS has stronger symmetric observed reach before activity is considered. |
| Active all-five overall; by date | 88.9%; 66.7/100.0/100.0% | 66.7%; 66.7/66.7/66.7% | INFY | INFY is never worse by date and is materially better on two dates. |
| Active node-slice | 88.9% | 84.4% | INFY | INFY retains +0.10 active support on more slices. |
| Trade-count all-five / node-slice | 77.8% / 86.7% | 66.7% / 82.2% | INFY | INFY has broader active central representation. |
| Trade-count-positive rows: overall / daily min / range | 69.5% / 58.4% / 22.2 pp | 60.1% / 49.3% / 16.4 pp | INFY on level and floor | TCS varies less, but INFY's worst date remains materially stronger. |
| OI-positive rows: overall / daily min / range | 87.0% / 74.1% / 21.4 pp | 76.1% / 61.6% / 23.3 pp | INFY | INFY is higher overall, higher on the worst date, and slightly steadier. |
| Last-positive rows: overall / daily min / range | 87.1% / 74.4% / 21.0 pp | 76.5% / 61.6% / 24.2 pp | INFY | INFY is stronger on all three summaries. |
| Settlement-positive rows: overall / daily min | 97.1% / 94.3% | 97.9% / 95.7% | TCS | TCS has a modest settlement advantage. |
| Completeness / futures / spot integrity | 100% | 100% | Tie | Non-discriminating eligibility checks all pass. |

**Decision: INFY. Confidence: high.** TCS has the more symmetric observed domain and a small settlement advantage, but INFY has materially stronger active all-five coverage, active node-slice support, trade-count coverage, OI coverage, last availability, and worst-date levels. TCS is the backup.

## 8. Financial/Banking: ICICIBANK vs HDFCBANK

| Metric | ICICIBANK | HDFCBANK | Better / tie | Rationale |
|---|---:|---:|---|---|
| Observed all-five slices | 100.0% | 100.0% | Tie | All nine slices support all five central nodes. |
| Common observed interval; symmetric reach | [-0.173519, +0.122747]; 0.122747 | [-0.157695, +0.142140]; 0.142140 | HDFCBANK | HDFCBANK has stronger symmetric observed reach and the smaller mean wing-gap. |
| Active all-five overall; by date | 77.8%; 66.7/66.7/100.0% | 88.9%; 66.7/100.0/100.0% | HDFCBANK | Same worst-date floor, but HDFCBANK is stronger on one additional date. |
| Active node-slice | 77.8% | 88.9% | HDFCBANK | HDFCBANK is uniformly stronger across the five nodes. |
| Trade-count all-five / node-slice | 55.6% / 71.1% | 77.8% / 84.4% | HDFCBANK | HDFCBANK has broader active central representation. |
| Trade-count-positive rows: overall / daily min / range | 44.0% / 32.7% / 22.2 pp | 70.7% / 61.6% / 19.1 pp | HDFCBANK | Higher overall, much higher floor, and lower variation. |
| OI-positive rows: overall / daily min / range | 57.4% / 40.7% / 31.7 pp | 82.8% / 71.5% / 17.8 pp | HDFCBANK | Higher overall, much higher floor, and lower variation. |
| Last-positive rows: overall / daily min / range | 57.6% / 40.7% / 32.1 pp | 83.2% / 72.7% / 16.6 pp | HDFCBANK | Higher overall, much higher floor, and lower variation. |
| Settlement-positive rows: overall / daily min | 95.0% / 91.5% | 97.9% / 95.9% | HDFCBANK | Higher overall and on the worst date. |
| Completeness / futures / spot integrity | 100% | 100% | Tie | Non-discriminating eligibility checks all pass. |

**Decision: HDFCBANK. Confidence: high.** HDFCBANK is stronger on symmetric observed reach, active central support, trade-count and OI coverage, price availability, and temporal consistency. ICICIBANK is the backup.

## 9. Power Tie-Break Extension

### 9.1 Why the original decision was rejected

The independent review accepted the arithmetic and provenance but rejected POWERGRID as the original tie-break winner. The smaller three-date range rewarded consistently lower activity rather than demonstrating a more robust usable surface. The correction was recorded as `POWER_SELECTION = UNRESOLVED` before acquiring or analyzing any new date.

### 9.2 Predeclared design and provenance

`POWER_CANDIDATE_TIEBREAK_EXTENSION` is separate from `ORIGINAL_STAGE_A`. It ranks only NTPC and POWERGRID over the complete five-Wednesday July 2026 panel:

- original Stage A dates: 2026-07-01, 2026-07-15, and 2026-07-22;
- added dates: 2026-07-08 and 2026-07-29; and
- no other date, August observation, Bloomberg field, NIFTY row, or other stock entered the tie-break.

The two added CM and F&O UDiFF archives came from `https://nsearchives.nseindia.com/content/`. They were preserved under the separate ignored `market_data_audit/stage_a/power_tiebreak/raw/nse` tree.

| Date / market | Archive SHA-256 | Extracted CSV SHA-256 |
|---|---|---|
| 2026-07-08 CM | `DF1B518CBE9FE9834CEA314ACA83EA1F01FBD50A101B1D7DE05E58D3A3BEA893` | `2032C64291DB018471B8808AF5795D5EFFF15383AE4B2308E0E750EE9F92AA3E` |
| 2026-07-08 FO | `4F674FA74C1F8531894AD6918A147A06A872188B360AD9EDBD0B1C710A41A950` | `53B74F8F82E0BF3A858C32E7CD6F2C31EF1990517DD175D1837D594967B41C3D` |
| 2026-07-29 CM | `48E878EB3E4FB1156947B126F39C32B8B5B3F332CA0DA6F40E78F51057EABC80` | `295216F074460BF05038036AA7FDFDA3FB07083F2568392D66F4AE6FFA784BFA` |
| 2026-07-29 FO | `AA7978C89D7BBF7BD7FBA9EF5AC59B450F310D42623E41DF1142024BE4C24C56` | `79B29FF8672A7C5E58402EB236BAF3FA66A9F01C5255162B2E0A0DAD36CE7E82` |

The 43-row deterministic evidence is `market_data_audit/stage_a/derived/power_candidate_tiebreak_evidence.csv`, SHA-256 `27C1F6A6C0E80809AB3A40BF7DE2B521BF540D341F779599A0BA2EE06893011F`. Consecutive offline replays produced the same evidence and acquisition-manifest hashes. An independent raw-CSV implementation reproduced all 86 candidate-value cells and their date-dominance fields with zero mismatches.

### 9.3 Eligibility and central support

Both candidates have 5/5 logical surfaces, three actual expiries on every surface, calls and puts in all 15 expiry slices, complete futures alignment, positive and consistent CM/F&O spot, complete option/futures expiry-field agreement, and 100% required-field reporting. Both have 100% observed support at each central node on every date.

| Five-date metric | NTPC | POWERGRID | Date dominance, NTPC/POWERGRID/tie |
|---|---:|---:|---:|
| Trade-count all-five active slices | 46.7% | 53.3% | 0/1/4 |
| Trade-count active central node-slices | 60.0%; floor 53.3% | 54.7%; floor 33.3% | 2/1/2 |
| OI all-five active slices | 46.7% | 53.3% | 0/1/4 |
| OI-active central node-slices | 60.0%; floor 53.3% | 58.7%; floor 40.0% | 1/1/3 |
| Trade-count-positive rows | 40.5%; floor 27.6% | 30.4%; floor 25.6% | 5/0/0 |
| OI-positive rows | 54.2%; floor 36.7% | 44.3%; floor 33.5% | 4/1/0 |
| Last-positive rows | 54.7%; floor 37.2% | 44.8%; floor 33.5% | 4/1/0 |
| Settlement-positive rows | 96.3%; floor 90.7% | 97.5%; floor 91.5% | 0/4/1 |
| Minimum symmetric observed reach | 0.150429 | 0.123456 | 4/1/0 |

The positive `TtlTradgVol` and positive trade-count flags coincide row-for-row in this panel. Only their binary status is reported; no traded-quantity or turnover magnitude is compared.

### 9.4 Date-by-date comparison

| Date | Trade-count rows: NTPC / POWERGRID | OI rows: NTPC / POWERGRID | Last rows: NTPC / POWERGRID | Trade-count active node-slices: NTPC / POWERGRID |
|---|---:|---:|---:|---:|
| 2026-07-01 | 27.6% / 27.3% | 36.7% / 38.1% | 37.2% / 38.6% | 53.3% / 40.0% |
| 2026-07-08 | 45.6% / 29.0% | 59.6% / 43.8% | 59.6% / 44.4% | 60.0% / 66.7% |
| 2026-07-15 | 47.9% / 32.2% | 66.4% / 46.9% | 66.4% / 48.0% | 66.7% / 66.7% |
| 2026-07-22 | 47.3% / 37.3% | 70.0% / 58.2% | 71.3% / 58.8% | 66.7% / 66.7% |
| 2026-07-29 | 38.2% / 25.6% | 40.9% / 33.5% | 41.8% / 33.5% | 53.3% / 33.3% |

POWERGRID's all-five active-slice advantage comes from one additional full central span on 08 July; the candidates tie on the other four dates. NTPC has the stronger aggregate active node support and materially higher worst-date active node floor, so the decision does not reward range in isolation.

### 9.5 Final Power decision

**POWER_SELECTION = NTPC. Confidence: moderate.** Observed central support and eligibility tie. POWERGRID retains a narrow advantage in full all-five active-slice frequency and settlement availability, but NTPC has the defensible overall surface-suitability advantage: stronger active central node coverage and floor, trade-count-positive coverage on all five dates, OI and last-positive coverage on four of five dates, and broader symmetric observed reach on four of five dates. POWERGRID remains the Power backup.

### 9.6 Extension limitations

- The extension contains five weekly Wednesday observations from one month and is not a timeless liquidity ranking.
- Support brackets a node within an observed strike range; it does not create or claim an exact-node quote.
- Free bhavcopy still lacks bid, ask, and size, so executable quote quality remains unassessed.
- Positive last and settlement values measure observation availability, not freshness or model suitability.
- The extension resolves only candidate membership. It does not select maturity or moneyness nodes.

## 10. Selected primary underlyings

| Sector | Primary | Confidence | Decisive reason |
|---|---|---|---|
| Power | NTPC | Moderate | Five-Wednesday evidence shows stronger active central floor, date-by-date activity coverage, and observed reach. |
| Healthcare/Pharma | CIPLA | Moderate | Stronger symmetric common domain and active central node-slice support. |
| IT | INFY | High | Stronger active central, trade-count, OI, last-price, and worst-date coverage. |
| Financial/Banking | HDFCBANK | High | Broad and consistent dominance on active central and downstream observation coverage. |

## 11. Retained alternatives

| Sector | Backup | Reason retained |
|---|---|---|
| Power | POWERGRID | Complete and eligible, with more all-five active slices and slightly better settlement availability, but weaker active node floor and row-level coverage. |
| Healthcare/Pharma | SUNPHARMA | Higher row-level activity, OI, and last-positive coverage despite weaker central-shape evidence. |
| IT | TCS | Strong symmetric observed support and slightly better settlement availability. |
| Financial/Banking | ICICIBANK | Fully eligible but materially weaker than HDFCBANK on active and temporal evidence. |

## 12. Limitations

- The original eight-stock evidence contains only three official-NSE dates; the Power extension contains five July Wednesdays. The selections are Stage A universe decisions, not timeless liquidity rankings.
- Free NSE bhavcopy has no historical bid, ask, bid size, or ask size. Spread quality and executable quote quality were not assessed or inferred.
- Support means an observed range brackets a node; it does not claim an observation exists exactly at that node and does not create an interpolated quote.
- Positive activity is a binary presence test. No absolute traded-quantity, turnover, lot, share, or contract comparison was made.
- Close, settlement, and last positivity measure observation availability, not price quality or option-model suitability by themselves.
- No implied volatility, ANN/PINN performance, parameter recovery, calibration accuracy, market capitalization, reputation, or subjective sector leadership entered the decision.
- The Power decision depends on the separately declared five-Wednesday extension; the original three-date evidence remains unresolved by itself.

## 13. Bloomberg note

Bloomberg was not used. No Bloomberg bid/ask field, entitlement, export, or observation entered the evidence or decision. A later quote-quality layer may use separately authorized Bloomberg evidence, but it does not replace the official NSE Stage A source and is not required for this milestone.

## 14. Downstream consequences

The selected primary set is NTPC, CIPLA, INFY, and HDFCBANK. The next milestone is deterministic common-support analysis across only these four primaries to determine a defensible maturity/moneyness representation for G2.

That next analysis is identified but not executed here. This milestone does not freeze a maturity grid, moneyness grid, feature count, normalization, or replacement neural input representation.

## 15. Exact gate status

```text
STAGE_A_CANDIDATE_SELECTION = COMPLETE
POWER_SELECTION = NTPC
SELECTED_PRIMARY_SET = NTPC | CIPLA | INFY | HDFCBANK
BACKUP_SET = POWERGRID | SUNPHARMA | TCS | ICICIBANK
REPLACEMENT_REPRESENTATION = OPEN
G2 = NOT_PASSED
FINAL_SYNTHETIC_RESEARCH_DATA = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
FINAL_MARKET_CALIBRATION = NOT_PERFORMED
```

Recommended next decision only: authorize common-support analysis across NTPC, CIPLA, INFY, and HDFCBANK. Do not decide the representation before that evidence is complete.
