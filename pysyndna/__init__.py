from pysyndna.src.fit_syndna_models import fit_linear_regression_models, \
    fit_linear_regression_models_for_qiita, LIN_REGRESS_RESULT_KEY, \
    FIT_SYNDNA_MODELS_LOG_KEY
from pysyndna.src.calc_cell_counts import calc_ogu_cell_counts_biom, \
    calc_ogu_cell_counts_per_g_of_sample_for_qiita, \
    OGU_CELLS_PER_G_OF_GDNA_KEY, OGU_CELLS_PER_G_OF_SAMPLE_KEY, \
    OGU_ID_KEY, OGU_LEN_IN_BP_KEY, OGU_PERCENT_COVERAGE_KEY, \
    CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY
from pysyndna.src.quant_orfs import \
    read_ogu_orf_coords_to_df, validate_and_cast_ogu_orf_coords_df, \
    calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs, \
    calc_copies_of_ogu_orf_ssrna_per_g_sample, \
    calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita, OGU_ORF_ID_KEY
from pysyndna.src.util import SAMPLE_ID_KEY

__all__ = ['fit_linear_regression_models',
           'fit_linear_regression_models_for_qiita',
           'calc_ogu_cell_counts_biom',
           'calc_ogu_cell_counts_per_g_of_sample_for_qiita',
           'read_ogu_orf_coords_to_df',
           'validate_and_cast_ogu_orf_coords_df',
           'calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs',
           'calc_copies_of_ogu_orf_ssrna_per_g_sample',
           'calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita',
           'OGU_CELLS_PER_G_OF_GDNA_KEY',
           'OGU_CELLS_PER_G_OF_SAMPLE_KEY',
           'SAMPLE_ID_KEY', 'OGU_ID_KEY', 'OGU_LEN_IN_BP_KEY',
           'OGU_ORF_ID_KEY', 'OGU_PERCENT_COVERAGE_KEY',
           'LIN_REGRESS_RESULT_KEY', 'FIT_SYNDNA_MODELS_LOG_KEY',
           'CELL_COUNT_RESULT_KEY', 'CELL_COUNT_LOG_KEY']

from . import _version
__version__ = _version.get_versions()['version']
