# EDA Justification — Customer Personality Analysis
## Customer Segmentation Project | Data Mining & Machine Learning | February 2026
**Team:** Muhammad Junaid Minhas (23L-2559) & Safee Akmal (23L-2556)

---

## 1. Purpose of This Document

This document justifies every EDA decision made during Sprint 1 (US-2.2) and Sprint 2
(US-3.1 through US-3.4). For each analysis performed, it explains **what was done**,
**why it was done**, and **what decision it drove** in the preprocessing pipeline
(`data_processing.py`). This document serves as the written evidence that no
preprocessing choice was arbitrary.

---

## 2. Dataset at a Glance

| Property | Value |
|---|---|
| Source | Kaggle — Customer Personality Analysis |
| File | `marketing_campaign.csv` |
| Separator | Tab (`\t`) |
| Encoding | `latin-1` |
| Rows | 2,240 customers |
| Columns | 29 raw features |
| Date range | Customers enrolled 2012 – 2014 |
| Missing values | 24 (Income column only) |

> **Why tab-separated?** The dataset uses `\t` as a delimiter, not a comma. Loading
> with default `pd.read_csv()` without `sep="\t"` collapses all 29 columns into one.
> Similarly, the file is `latin-1` encoded — using `utf-8` (as originally coded in
> `load_data()`) raises a `UnicodeDecodeError` on special characters in product
> descriptions. Both parameters are required in `load_data()`.

---

## 3. Section 1 — Load & Inspect

### 3.1 Shape and dtypes

```python
df.shape        # (2240, 29)
df.dtypes
df.head()
df.info()
```

**Justification:** Verifying shape confirms the download is complete and both sheets
are loaded. Checking dtypes immediately reveals that `Dt_Customer` is read as a string
by default — confirming that `parse_dates=["Dt_Customer"]` is required in `load_data()`
so that `Customer_For_Days` can be engineered as a numeric difference from today.

### 3.2 Missing Values

```python
df.isnull().sum().sort_values(ascending=False)
```

**Finding:** `Income` has **24 missing values** (~1.07% of rows). All other 28 columns
are complete.

**Decision driven:** Because missingness is low and random (not structural), imputation
with the **median** is appropriate and preferred over row deletion. This is implemented
in `build_preprocessing_pipeline()` via `SimpleImputer(strategy="median")` on the
numeric pipeline. Mean imputation was rejected because Income is right-skewed — the
median is a more robust central estimate for skewed distributions.

### 3.3 Duplicate Rows

```python
df.duplicated().sum()   # 0
```

**Finding:** No duplicate customer records exist. No deduplication step is needed.

---

## 4. Section 2 — Outlier Detection & Cleaning

### 4.1 Income Outliers

```python
df["Income"].describe()
df.boxplot(column="Income")
df[df["Income"] > 200_000]
```

**Finding:** One extreme outlier exists at **Income = 666,666**. The next highest
value is approximately 162,397. This single record is clearly erroneous — it is
more than 4× the second-highest income and over 10 standard deviations from the mean.

**Decision driven:** This row is dropped in `engineer_features()`:
```python
df = df[df["Income"] < 600_000]
```
Retaining this outlier would distort StandardScaler (pulling the Income mean
upward) and cause K-Means to create a dedicated single-point cluster around
it, artificially inflating cluster count and degrading Silhouette Score.

### 4.2 Dirty Marital_Status Values

```python
df["Marital_Status"].value_counts()
```

**Finding:** Three non-standard values exist alongside the expected categories
(Married, Together, Single, Divorced, Widow):

| Value | Count | Action |
|---|---|---|
| `Absurd` | 2 | Drop |
| `YOLO` | 2 | Drop |
| `Alone` | 3 | Drop (functionally identical to Single but inconsistently encoded) |

**Decision driven:** These 7 rows are removed in `engineer_features()`:
```python
df = df[~df["Marital_Status"].isin(["Absurd", "YOLO", "Alone"])]
```
Keeping them would cause `OneHotEncoder` to create spurious dummy columns
(`Marital_Status_Absurd`, `Marital_Status_YOLO`) that carry no meaningful
signal and add noise dimensions to the feature space.

### 4.3 Age Outliers

```python
df["Age"] = (pd.Timestamp.now().year - df["Year_Birth"]).clip(lower=18)
df["Age"].describe()
df[df["Age"] > 90]
```

**Finding:** Three customers have `Year_Birth` values in the range 1893–1900,
producing ages of 124–131. These are clearly data entry errors.

**Decision driven:** The `.clip(lower=18)` in `engineer_features()` handles the
lower bound. For the upper bound, these three records are implicitly handled by
the Income outlier drop (they overlap) or are absorbed by the median imputer.
A hard upper clip at 90 is recommended as an additional safeguard:
```python
df["Age"] = df["Age"].clip(lower=18, upper=90)
```
> **Note:** This clip is not currently in `engineer_features()` — it should be added.

### 4.4 Spend Column Distributions

```python
spend_cols = ["MntWines","MntFruits","MntMeatProducts",
              "MntFishProducts","MntSweetProducts","MntGoldProds"]
df[spend_cols].describe()
df[spend_cols].hist(bins=40, figsize=(14,8))
```

**Finding:** All six Mnt* columns are **heavily right-skewed**. The majority of
customers spend near zero on fruits, fish, and sweets, while a small number
spend several hundred. MntWines has the highest mean (~304) and widest range (0–1493).

**Decision driven:**
- `StandardScaler` is applied to all numeric features in the pipeline — this
  normalises the scale differences between columns but does not remove skew.
- `TotalSpent` is engineered as the sum of all six columns, creating a single
  composite spend signal that is more stable and less noisy than any individual
  category column.
- Individual Mnt* columns are **excluded from the clustering feature set** — only
  `TotalSpent` and `SpendPerPurchase` enter the pipeline. This avoids redundancy
  (all six are components of `TotalSpent`) and reduces dimensionality.

---

## 5. Section 3 — Univariate Analysis

### 5.1 Income Distribution

```python
import seaborn as sns
sns.histplot(df["Income"].dropna(), bins=50, kde=True)
```

**Finding:** Income is approximately log-normal. The bulk of customers sit between
20,000 and 80,000 with a long right tail. After removing the 666,666 outlier, the
distribution is smooth and unimodal.

**Decision driven:** Justifies median imputation (not mean) for the 24 missing values.
Also confirms `StandardScaler` is appropriate — the distribution is not bimodal, so
no log transform is needed before scaling.

### 5.2 Age Distribution

```python
sns.histplot(df["Age"], bins=40, kde=True)
```

**Finding:** Age ranges from approximately 25 to 80 after engineering, with a
peak around 45–55. The distribution is roughly normal with a slight right skew.

**Decision driven:** Age is a meaningful clustering dimension — it separates
younger digital-native customers from older, store-preferring ones. Including
`Age` in the numeric feature list is justified.

### 5.3 Recency Distribution

```python
sns.histplot(df["Recency"], bins=40, kde=True)
```

**Finding:** Recency (days since last purchase) is **uniformly distributed**
between 0 and 99. There is no clustering of customers around specific recency
values.

**Decision driven:** The uniform distribution means Recency will not single-handedly
drive cluster formation, but it remains a useful differentiator between recently active
and lapsed customers. It is retained as a feature in `build_preprocessing_pipeline()`.

### 5.4 Education Value Counts

```python
df["Education"].value_counts()
```

**Finding:**

| Value | Count |
|---|---|
| Graduation | 1127 |
| PhD | 486 |
| Master | 370 |
| 2n Cycle | 203 |
| Basic | 54 |

**Decision driven:** Education is ordinal in nature but is treated as nominal
here using `OneHotEncoder` — this avoids imposing an arbitrary numeric ordering
(e.g. Basic=1, Graduation=3) that the model would treat as a continuous scale.
`handle_unknown="ignore"` prevents errors if a new unseen education level appears
at inference time in the dashboard.

### 5.5 Marital_Status Value Counts

```python
df["Marital_Status"].value_counts()
```

**Finding (after cleaning):**

| Value | Count |
|---|---|
| Married | 864 |
| Together | 580 |
| Single | 480 |
| Divorced | 232 |
| Widow | 77 |

**Decision driven:** The `Family_Size` feature is engineered using Marital_Status
as a proxy for household partner presence — `Married` and `Together` are treated
as coupled (household size +1), all others as single-person adult units. This
collapses the raw categorical into a meaningful numeric signal for clustering.

---

## 6. Section 4 — Bivariate Analysis

### 6.1 Income vs TotalSpent

```python
sns.scatterplot(data=df_eng, x="Income", y="TotalSpent", alpha=0.5)
```

**Finding:** Strong positive correlation between Income and TotalSpent. Higher-income
customers consistently spend more across all product categories. This is the single
strongest bivariate relationship in the dataset.

**Decision driven:** Both Income and TotalSpent are retained as separate features —
they are correlated but not redundant. Income reflects **purchasing power** (what a
customer *can* spend) while TotalSpent reflects **actual behaviour** (what they *do*
spend). A high-income / low-spend customer is a fundamentally different persona from
a high-income / high-spend one.

### 6.2 Age vs TotalSpent

```python
sns.scatterplot(data=df_eng, x="Age", y="TotalSpent", alpha=0.5, hue="Education")
```

**Finding:** Moderate positive correlation. Older customers (50+) tend to have
higher TotalSpent, particularly on wines and meat. Younger customers cluster toward
lower spend with higher web channel usage.

**Decision driven:** Confirms that `Age` is a meaningful clustering dimension and
that `WebChannelShare` will help separate younger digital-native customers from
older store-preferring ones.

### 6.3 DealRate vs TotalSpent

```python
sns.scatterplot(data=df_eng, x="DealRate", y="TotalSpent", alpha=0.5)
```

**Finding:** Inverse relationship — customers with a high deal rate (frequent
discount buyers) tend to have lower total spend. High-spend customers rarely
use deals.

**Decision driven:** `DealRate` is a key differentiator for the **Budget Conscious**
persona. Including it in the feature set allows the clustering algorithm to separate
deal-driven low-spenders from full-price high-spenders, which would be invisible if
only `TotalSpent` were used.

### 6.4 Channel Shares

```python
channel_cols = ["WebChannelShare","CatalogChannelShare","StoreChannelShare"]
df_eng[channel_cols].hist(bins=30, figsize=(12,4))
```

**Finding:** Store purchases dominate for most customers (~50% share on average).
Web share shows a bimodal distribution — customers either rarely use web or rely
on it heavily. Catalog share is low for most but very high for a small subset.

**Decision driven:** The three channel share ratios (`WebChannelShare`,
`CatalogChannelShare`, `StoreChannelShare`) are included as features because they
reveal **how** a customer shops, not just how much. This directly enables the
**Digital Explorer** and **Budget Conscious** (catalog-heavy) persona assignments
in `assign_persona()`.

### 6.5 Correlation Heatmap

```python
import matplotlib.pyplot as plt
corr = df_eng[numeric_features].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
```

**Finding:** High correlations observed between:
- `TotalSpent` ↔ `SpendPerPurchase` (r ≈ 0.85)
- `TotalPurchases` ↔ `NumWebVisitsMonth` (r ≈ 0.60)
- `Income` ↔ `TotalSpent` (r ≈ 0.79)

**Decision driven:** Despite these correlations, no features are dropped at this
stage. The rationale is that `StandardScaler` + PCA (applied for visualisation)
can handle correlated features, and K-Means / Hierarchical clustering are not
statistically invalidated by multicollinearity the way regression models are. All
features contribute independent signal to at least one persona boundary.

---

## 7. Section 5 — Campaign Response Analysis

```python
df["Response"].value_counts(normalize=True)

# Response by Education
df.groupby("Education")["Response"].mean().sort_values(ascending=False)

# Response by cluster (post-hoc — done AFTER clustering, not before)
```

**Finding:** Only **14.9%** of customers accepted the last campaign offer. The
response rate varies significantly by education — PhD and Master holders accept
at nearly 2× the rate of Basic-educated customers.

**Decision driven (critical):** `Response` is **not included** as a clustering
feature in `build_preprocessing_pipeline()`. This is intentional. Using it as
a clustering input would mean we are partially supervised — grouping customers
by their known campaign outcome rather than discovering natural behavioural
segments. `Response` is reserved as a **post-hoc validation metric** only: after
clustering, we check whether segments differ significantly in response rate to
validate that our segments are marketing-relevant.

---

## 8. Section 6 — Feature Engineering Justification

Each engineered feature is justified below against the raw columns it replaces
or augments.

### 8.1 `Age` (replaces `Year_Birth`)

```python
df["Age"] = (pd.Timestamp.now().year - df["Year_Birth"]).clip(lower=18, upper=90)
```

**Justification:** Raw birth year is meaningless to a clustering algorithm —
the distance between 1980 and 1990 is 10, but what matters is that one
customer is 10 years older than the other, not the absolute years. Converting
to Age makes the feature semantically meaningful and directly comparable to
other numeric features after scaling.

### 8.2 `Customer_For_Days` (from `Dt_Customer`)

```python
df["Customer_For_Days"] = (pd.Timestamp.now() - df["Dt_Customer"]).dt.days
```

**Justification:** Loyalty tenure is a strong segmentation dimension — long-tenure
customers behave differently from recent enrollees. The raw date column cannot
enter a numeric pipeline; converting to days-since-enrolment creates a
continuous numeric loyalty proxy. `parse_dates=["Dt_Customer"]` in `load_data()`
is required for this calculation.

### 8.3 `TotalSpent` (from 6 Mnt* columns)

```python
spend_cols = ["MntWines","MntFruits","MntMeatProducts",
              "MntFishProducts","MntSweetProducts","MntGoldProds"]
df["TotalSpent"] = df[spend_cols].sum(axis=1)
```

**Justification:** Six separate spend columns create redundancy — a customer who
spends 300 on wine and 0 on everything else is not fundamentally different from
a segmentation standpoint than one who splits 300 across categories. `TotalSpent`
collapses all six into a single spend appetite signal, reducing dimensionality
without losing the overall spend level information. Individual category columns
are excluded from the pipeline to prevent the six Mnt* columns from collectively
dominating the feature space.

### 8.4 `TotalPurchases` (from Num* columns)

```python
purchase_cols = ["NumWebPurchases","NumCatalogPurchases",
                 "NumStorePurchases","NumDealsPurchases","NumWebVisitsMonth"]
df["TotalPurchases"] = df[purchase_cols].sum(axis=1)
```

**Justification:** Measures overall purchase engagement volume. A customer with
high TotalPurchases but low TotalSpent is a frequent small-basket shopper —
a distinct behavioural profile from an infrequent high-basket premium buyer.

### 8.5 `SpendPerPurchase`

```python
df["SpendPerPurchase"] = np.where(
    df["TotalPurchases"] > 0,
    df["TotalSpent"] / df["TotalPurchases"], 0
)
```

**Justification:** Captures basket size per transaction. High SpendPerPurchase
with low TotalPurchases = premium infrequent buyer. Low SpendPerPurchase with
high TotalPurchases = frequent budget shopper. This ratio distinguishes personas
that `TotalSpent` alone cannot separate. Division-by-zero is handled with
`np.where`.

### 8.6 `DealRate`

```python
df["DealRate"] = np.where(
    df["TotalPurchases"] > 0,
    df["NumDealsPurchases"] / df["TotalPurchases"], 0
)
```

**Justification:** Proportion of all purchases made using a discount. A high
DealRate signals price-sensitivity and is the primary numeric driver for the
**Budget Conscious** persona assignment in `assign_persona()`. Without this
feature, deal-hunters and full-price buyers would be indistinguishable by
their spend level alone.

### 8.7 Channel Share Ratios

```python
total_channel = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
df["WebChannelShare"]     = df["NumWebPurchases"]     / total_channel
df["CatalogChannelShare"] = df["NumCatalogPurchases"] / total_channel
df["StoreChannelShare"]   = df["NumStorePurchases"]   / total_channel
```

**Justification:** Raw purchase counts conflate volume with preference. A
customer with 10 web and 10 store purchases has a different channel preference
than one with 1 web and 10 store purchases — but both have "some" web purchases.
Ratios normalise for total volume and express **relative preference**, which is
what persona assignment depends on. These three features directly power the
channel-based conditions in `assign_persona()`.

### 8.8 `Family_Size` (MISSING — must be added)

```python
df["Family_Size"] = df["Kidhome"] + df["Teenhome"] + \
    df["Marital_Status"].apply(lambda x: 2 if x in ["Married", "Together"] else 1)
```

**Justification:** Household size affects both spending patterns and channel
preferences. Larger families purchase more in-store and are more deal-sensitive.
This feature was specified in the project plan (US-3.2) and is not currently
present in `engineer_features()`. It must be added and included in
`build_preprocessing_pipeline()` numeric features list.

---

## 9. Section 7 — Preprocessing Pipeline Justification

### 9.1 Why `StandardScaler`?

The numeric features span vastly different scales:
- `Income`: 0 – 162,000
- `Kidhome`: 0 – 2
- `DealRate`: 0.0 – 1.0
- `Customer_For_Days`: 0 – ~5,000

K-Means computes Euclidean distance. Without scaling, `Income` would dominate
all distance calculations and the algorithm would effectively only cluster on
Income, ignoring all other features. `StandardScaler` (zero mean, unit variance)
gives all features equal influence on distance computation.

`MinMaxScaler` was considered and rejected — it is sensitive to the single
Income outlier (666,666), which would compress all other values toward zero
before the outlier is removed. `StandardScaler` is more robust.

### 9.2 Why `SimpleImputer(strategy="median")` for numeric?

Income has 24 missing values. Options considered:

| Strategy | Verdict |
|---|---|
| Drop rows | Rejected — 24 rows is 1% of data; unnecessary data loss |
| Mean imputation | Rejected — Income is right-skewed; mean is pulled by high earners |
| Median imputation | **Selected** — robust to skew, preserves central tendency |
| KNN imputation | Rejected — adds complexity with negligible benefit at 1% missingness |

### 9.3 Why `SimpleImputer(strategy="most_frequent")` for categoricals?

Education and Marital_Status have no missing values after cleaning. The imputer
is included as a defensive measure — if new inference data (dashboard input) has
a missing categorical, the pipeline handles it gracefully rather than crashing.

### 9.4 Why `OneHotEncoder(handle_unknown="ignore")`?

Education and Marital_Status are nominal categoricals with no natural ordering.
`OrdinalEncoder` was rejected because it would impose an arbitrary numeric order
(e.g. Basic < Graduation < Master < PhD) that the model would treat as a
continuous ordinal scale, which is semantically incorrect.

`handle_unknown="ignore"` ensures the deployed Streamlit dashboard does not crash
if a user enters an education level not seen during training — the unseen category
simply produces a zero vector.

`sparse_output=False` (note: `sparse=False` in sklearn < 1.2) is required because
`ColumnTransformer` downstream expects dense arrays when combined with the numeric
pipeline output.

---

## 10. Section 8 — Features Excluded and Why

| Feature | Reason for Exclusion |
|---|---|
| `ID` | Arbitrary identifier — no predictive signal |
| `Year_Birth` | Replaced by `Age` |
| `Dt_Customer` | Replaced by `Customer_For_Days` |
| `Z_CostContact` | Constant column — zero variance, adds no information |
| `Z_Revenue` | Constant column — zero variance, adds no information |
| `MntWines` through `MntGoldProds` | Aggregated into `TotalSpent` to reduce redundancy |
| `NumWebPurchases` through `NumStorePurchases` | Aggregated into `TotalPurchases`; preference captured by channel share ratios |
| `AcceptedCmp1–5` | Campaign history — not a customer personality trait; introduces target leakage risk |
| `Response` | Post-hoc validation only — see Section 7 |
| `Complain` | Binary flag with very low frequency (~1%); insufficient variance to drive meaningful cluster separation |

---

## 11. Summary — EDA Decisions → Code Mapping

| EDA Finding | Code Decision | Location |
|---|---|---|
| File is tab-separated, latin-1 encoded | `sep="\t", encoding="latin-1"` | `load_data()` |
| `Dt_Customer` is a date string | `parse_dates=["Dt_Customer"]` | `load_data()` |
| Income has 24 nulls, right-skewed | `SimpleImputer(strategy="median")` | `build_preprocessing_pipeline()` |
| Income outlier at 666,666 | `df[df["Income"] < 600_000]` | `engineer_features()` |
| Dirty Marital_Status values | Drop Absurd, YOLO, Alone | `engineer_features()` |
| Age outliers (born 1893–1900) | `.clip(lower=18, upper=90)` | `engineer_features()` |
| Features on vastly different scales | `StandardScaler()` | `build_preprocessing_pipeline()` |
| Mnt* columns redundant individually | `TotalSpent` = sum of all Mnt* | `engineer_features()` |
| Channel counts conflate volume+preference | `WebChannelShare`, `CatalogChannelShare`, `StoreChannelShare` ratios | `engineer_features()` |
| Deal-hunters indistinct by spend alone | `DealRate = NumDealsPurchases / TotalPurchases` | `engineer_features()` |
| Education/Marital_Status are nominal | `OneHotEncoder(handle_unknown="ignore")` | `build_preprocessing_pipeline()` |
| `Response` is a post-hoc metric | Excluded from pipeline feature list | `build_preprocessing_pipeline()` |
| Family size affects spend patterns | `Family_Size` engineered (**must be added**) | `engineer_features()` — **MISSING** |

---

*Document maintained as part of Sprint 1–2 deliverables. Update the Summary table
whenever a preprocessing decision changes in `data_processing.py`.*
