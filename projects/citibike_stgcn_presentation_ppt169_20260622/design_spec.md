# citibike_stgcn_presentation - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| Project Name | citibike_stgcn_presentation |
| Canvas Format | PPT 16:9 |
| Page Count | 10 |
| Design Style | data-journalism, dense chart-led briefing |
| Target Audience | coursework / project presentation audience |
| Use Case | experiment result presentation |
| Created Date | 2026-06-22 |

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| Format | PPT 16:9 |
| Dimensions | 1280 x 720 |
| viewBox | `0 0 1280 720` |
| Margins | 42 px outer rail, 72 px content inset |
| Content Area | chart-dominant layout with explanatory sidebars |

## III. Visual Theme

### Theme Style

- Mode: briefing
- Visual style: data-journalism
- Theme: light publication theme
- Tone: dense, clear, chart-first, suitable for oral reporting

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| Background | `#F8FAFC` | slide background |
| Surface | `#FFFFFF` | chart panels and sidebars |
| Primary | `#0C6DB8` | STGCN and shortage emphasis |
| Accent | `#28C0CD` | model / performance accent |
| Secondary accent | `#E06A36` | overflow and risk emphasis |
| Body text | `#1D2433` | main text |
| Secondary text | `#586070` | captions |
| Border/divider | `#D8DEE5` | thin dividers |
| Taupe | `#948080` | supporting bars |
| Peach | `#E99E80` | supporting bars |
| Teal deep | `#557F7C` | spatial feature emphasis |
| Cream | `#EBC9B9` | soft contrast |

## IV. Typography System

### Font Plan

Typography direction: CJK-first sans, compact and presentation-safe.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| Title | Microsoft YaHei | Arial | sans-serif |
| Body | Microsoft YaHei | Arial | sans-serif |
| Emphasis | Microsoft YaHei | Arial | sans-serif |
| Code | Consolas | Courier New | monospace |

Per-role font stacks:

- Title: `"Microsoft YaHei", Arial, sans-serif`
- Body: `"Microsoft YaHei", Arial, sans-serif`
- Emphasis: `"Microsoft YaHei", Arial, sans-serif`
- Code: `Consolas, "Courier New", monospace`

Baseline body font size: 18 px. Page titles use 28-34 px. Chart captions use 13-15 px.

## V. Layout Principles

The deck uses chart-led pages: each page places one large chart or two paired charts, with nearby explanatory text boxes. It avoids fixed top title bars and avoids empty space. Left vertical rails carry compact page labels; charts and explanation blocks fill the canvas.

## VI. Icon Usage Specification

No icon dependency is required. The deck uses color rails, labels, and charts instead of decorative icons.

## VII. Visualization Reference List

All visuals are pre-rendered chart images generated from experiment outputs. They are inserted as no-crop image assets and explained with native PPT/SVG text.

## VIII. Image Resource List

| Filename | Purpose | Type | Acquire Via | Status | page_role |
| -------- | ------- | ---- | ----------- | ------ | --------- |
| ppt_fig_01_model_rmse.png | model RMSE comparison | chart | user | ready | local |
| ppt_fig_02_model_r2.png | model R2 comparison | chart | user | ready | local |
| ppt_fig_03_feature_ablation.png | feature group ablation | chart | user | ready | local |
| ppt_fig_04_feature_importance.png | feature importance | chart | user | ready | local |
| ppt_fig_05_training_loss.png | STGCN training loss | chart | user | ready | local |
| ppt_fig_06_observed_predicted.png | observed vs predicted density | chart | user | ready | local |
| ppt_fig_07_hourly_error.png | hourly absolute error | chart | user | ready | local |
| ppt_fig_08_shortage_risk.png | shortage risk ranking | chart | user | ready | local |
| ppt_fig_09_overflow_risk.png | overflow risk ranking | chart | user | ready | local |
| ppt_fig_10_risk_locations.png | risk location map | chart | user | ready | local |

## IX. Content Outline

| Page | Title | Core message | Visual |
| ---- | ----- | ------------ | ------ |
| P01 | 区域净流量预测与调度风险识别 | The deck introduces a regional STGCN workflow and the final metrics. | metric cards |
| P02 | 任务升级 | The target changes from citywide demand to regional imbalance. | explanatory blocks |
| P03 | 模型效果 | STGCN has the lowest RMSE and highest R2. | RMSE + R2 charts |
| P04 | 空间特征 | Spatial features help but map-only features are not enough. | ablation chart |
| P05 | 关键变量 | current net flow, hour cycle, and lag features dominate. | importance chart |
| P06 | 训练诊断 | training and validation losses converge steadily. | loss chart |
| P07 | 预测贴合 | predicted net flow tracks observed net flow. | hexbin chart |
| P08 | 时段误差 | commute hours are harder to predict. | hourly error chart |
| P09 | 风险排序 | shortage and overflow risks map to different actions. | two risk charts |
| P10 | 空间落点 | highest risks concentrate around the same central grid. | risk map |

## X. Speaker Notes Plan

Each page note gives the exact reporting point: what the chart shows, why it matters, and how to transition to the next page.

## XI. Technical Constraints

SVG pages use only PPT-compatible elements: inline attributes, no CSS classes, no foreignObject, no masks, and chart images inserted with `preserveAspectRatio="xMidYMid meet"`.
