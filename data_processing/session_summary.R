### ========================================================================================================================================
## Name:        Saputra Rizky Johan (사푸트라)
## Institution: Seoul National University, South Korea
## Purpose:     Undergraduate Thesis
## Department:  Computer Science and Engineering
## College:     Engineering
## Advisor:     Prof. Lee Changun (이창건 교수)
## Title:       The Application of Facial Emotion Recognition (FER) in the Detection and Measurement of Burnout and Prolonged Stress Levels
## Module:      session_summary.r (A small test module to generate the table at section 5.2 of the thesis)
### ========================================================================================================================================

# =============================================================================================
# SETUP
# =============================================================================================
# install.packages(c("gt", "dplyr", "stringr"))  # <- run once
library(gt)
library(dplyr)
library(stringr)

# =============================================================================================
# IMPLEMENTATION
# =============================================================================================
# Data initialization
df <- tibble::tribble(
  ~Participant, ~AvgSystem, ~Min, ~Max, ~StdDev, ~Survey, ~AbsDiff,
  "user001",    0.400,      0.340, 0.780, 0.090,   0.600,    0.200,
  "user002",    0.372,      0.340, 0.640, 0.043,   0.600,    0.228,
  "user003",    0.717,      0.650, 0.750, 0.021,   0.733,    0.016,
  "user004",    0.714,      0.680, 0.730, 0.015,   0.867,    0.153,
  "user005",    0.706,      0.690, 0.720, 0.006,   0.867,    0.161,
)

# Construct the Mean +/- SD display row
fmt_pm <- function(x) sprintf("%.3f ± %.3f", mean(x), sd(x))
mean_row <- tibble::tibble(
  Participant = "Mean ± SD",
  AvgSystem   = fmt_pm(df$AvgSystem),
  Min         = "—",
  Max         = "—",
  StdDev      = "—",
  Survey      = fmt_pm(df$Survey),
  AbsDiff     = fmt_pm(df$AbsDiff)
)

display_tbl <- df %>%
  mutate(
    AvgSystem = sprintf("%.3f", AvgSystem),
    Min       = sprintf("%.3f", Min),
    Max       = sprintf("%.3f", Max),
    StdDev    = sprintf("%.3f", StdDev),
    Survey    = sprintf("%.3f", Survey),
    AbsDiff   = sprintf("%.3f", AbsDiff)
  ) %>%
  bind_rows(mean_row)

# Render the data with gt and load as a PNG file
note_text <- paste0(
  "Note. The Pearson correlation coefficient, denoted as r, indicates the relationship between
   the measured system and survey based burnout scoring results across the five participants.",
  "Hereby, the total Pearson correlation coefficient is computed as r ≈ 0.904, which indicates",
  "a strong positively linear relationship between the two measurements (refer to section 5.3)"
)

gt_tbl <- display_tbl |>
  gt() |>
  tab_header(
    title = md("**Table 1. FER Burnout System Session Based Quantitative Metrics**")
  ) |>
  cols_label(
    Participant = md("**Participant**"),
    AvgSystem   = md("**Avg. System Burnout**"),
    Min         = md("**Min**"),
    Max         = md("**Max**"),
    StdDev      = md("**Std. Dev**"),
    Survey      = md("**Survey Burnout**"),
    AbsDiff     = md("**|System – Survey|**")
  ) |>
  tab_style(
    style = list(cell_text(weight = "bold")),
    locations = cells_body(rows = Participant == "Mean ± SD")
  ) |>
  tab_source_note(md(paste0("*", note_text, "*"))) |>
  tab_options(
    table.align = "left",
    data_row.padding = px(6),
    column_labels.font.weight = "bold"
  )

# Save as high-DPI PNG ("session_summary.png") for documentation
gtsave(gt_tbl, filename = "session_summary.png", expand = 10)

### ===================================================================================================================================
## END: Add implementations if necessary
### ===================================================================================================================================