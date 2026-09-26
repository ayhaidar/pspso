# Bundled example datasets

These compact datasets are bundled so dashboard and CLI examples remain usable
without an internet connection. The CSV files are preserved from their cited
sources; PSPSO applies only the documented loading transformations.

| File | Source | License | Loading transformation |
| --- | --- | --- | --- |
| `banknote_authentication.csv` | [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication), downloaded 8 September 2026 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | None |
| `auto_mpg.csv` | [UCI Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg), revised from CMU StatLib and downloaded 8 September 2026 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Drops the `car_name` identifier and labels origin codes as USA, Europe, or Japan |
| `palmer_penguins.csv` | [palmerpenguins](https://allisonhorst.github.io/palmerpenguins/), downloaded 8 September 2026 | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Parses the source's `NA` markers as missing values |

SHA-256 checksums of the downloaded CSV snapshots:

- `auto_mpg.csv`: `240f4657d747fe105f3c992251daba65f926ef54c4dec3cde34e398e8cf5da9c`
- `banknote_authentication.csv`: `0787e50299c198e919276660edf98dd5dcb94951ad81435b3ab61afb666936c9`
- `palmer_penguins.csv`: `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`

Dataset citations:

- Lohweg, V. (2012). *Banknote Authentication*. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C55P57>
- Quinlan, R. (1993). *Auto MPG*. UCI Machine Learning Repository.
  <https://doi.org/10.24432/C5859H>
- Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer
  Archipelago (Antarctica) penguin data*. <https://doi.org/10.5281/zenodo.3960218>
