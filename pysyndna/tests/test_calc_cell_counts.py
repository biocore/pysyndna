import biom.table
import copy
import numpy as np
import pandas as pd
from pandas.arrays import SparseArray
from pandas.testing import assert_series_equal
import os
from unittest import TestCase
from pysyndna import calc_ogu_cell_counts_biom, \
    calc_ogu_cell_counts_per_g_of_sample_for_qiita, \
    calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input, \
    calc_ogu_cell_counts_per_cm2_of_sample_for_qiita, \
    calc_ogu_cell_counts_per_ul_of_sample_for_qiita
from pysyndna.src.calc_cell_counts import \
    calc_ogu_cell_counts_per_cm2_of_sample_for_qiita_split_input, \
    calc_ogu_cell_counts_per_ul_of_sample_for_qiita_split_input
from pysyndna.src.util import OGU_ID_KEY
from pysyndna.src.fit_syndna_models import SAMPLE_TOTAL_READS_KEY
from pysyndna.src.calc_cell_counts import SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY, \
    OGU_READ_COUNT_KEY, \
    OGU_LEN_IN_BP_KEY, OGU_GDNA_MASS_NG_KEY, \
    SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY, OGU_GENOMES_PER_G_OF_GDNA_KEY, \
    OGU_CELLS_PER_G_OF_GDNA_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY, \
    GDNA_CONCENTRATION_NG_UL_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY, \
    GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY, OGU_CELLS_PER_G_OF_SAMPLE_KEY, \
    SAMPLE_VOLUME_UL_KEY, \
    GDNA_MASS_TO_SAMPLE_VOL_RATIO_KEY, OGU_CELLS_PER_UL_OF_SAMPLE_KEY, \
    SAMPLE_SURFACE_AREA_CM2_KEY, GDNA_MASS_TO_SAMPLE_SURFACE_AREA_RATIO_KEY, \
    OGU_CELLS_PER_CM2_OF_SAMPLE_KEY, \
    OGU_PERCENT_COVERAGE_KEY, OGU_AGNOSTIC_COVERAGE_KEY, \
    CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY, \
    _calc_long_format_ogu_cell_counts_df, \
    _prepare_cell_counts_calc_df, \
    _calc_ogu_cell_counts_df_for_sample, \
    _calc_ogu_gdna_mass_ng_series_for_sample, \
    _calc_ogu_genomes_per_g_of_gdna_series_for_sample, \
    _calc_ogu_genomes_series_for_sample
from pysyndna.tests.test_util import Testers


def _remove_filtered_entries(
        input_list, zero_based_positions_to_remove=None):
    if zero_based_positions_to_remove is None:
        zero_based_positions_to_remove = [7, 11]
    return [x for i, x in enumerate(input_list) if
            i not in zero_based_positions_to_remove]


class TestCalcCellCountsData:
    # Throughout below, example1 is the "Sample A" data from
    # test_fit_syndna_models.py, paired with "standard" elute volume and
    # gdna density values. example2 is the "Sample B" data from
    # test_fit_syndna_models.py, paired with a smaller elute volume and
    # and a lower gdna density value (which leadds to a smaller amount of
    # syndna pool and thus a bunch of interesting calculation knock-ons).

    # The example1 (Sample A) and example2 (Sample B) values are taken from
    # FitSyndnaModelsTest.lingress_results, which gets them from those
    # calculated in Excel (see results for full data
    # on "linear regressions counts" sheet of "absolute_quant_example.xlsx").
    # Note that these do not and *should* NOT be expected to match any results
    # in Zaramela's linear models; see FitSyndnaModelsTest.lingress_results
    # comments for details.
    linregresses_dict = {
        'example1': {
            "slope": 1.24487652379132,
            "intercept": -7.35593916054843,
            "rvalue": 0.986503097515657,
            "pvalue": 1.42844356065977E-07,
            "stderr": 0.0730540855033502,
            "intercept_stderr": 0.271274537363401},
        'example2': {
            "slope": 1.24675913604407,
            "intercept": -7.45004083037736,
            "rvalue": 0.986324179735633,
            "pvalue": 1.5053811468097E-07,
            "stderr": 0.073657952553024,
            "intercept_stderr": 0.2729411999326}
    }

    # Values from "absolute_quant_example.xlsx"
    sample_and_prep_input_dict = {
        SAMPLE_ID_KEY: ["example1", "example2"],
        SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.027829017, 0.029491697],
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY: [5, 4.76],
        SAMPLE_VOLUME_UL_KEY: [200, 150],
        SAMPLE_SURFACE_AREA_CM2_KEY: [4, 6],
        GDNA_CONCENTRATION_NG_UL_KEY: [2, 1.4],
        ELUTE_VOL_UL_KEY: [100, 100],
        INPUT_SYNDNA_POOL_MASS_NG_KEY: [0.25, 0.238],
    }

    # Values from "absolute_quant_example.xlsx" EXCEPT for the
    # SAMPLE_TOTAL_READS_KEY values, which come from summing
    # the OGU_READ_COUNT_KEY values for each sample
    mass_and_totals_dict = {
        SAMPLE_ID_KEY: ["example1", "example2"],
        # Note that total reads aren't actually needed anymore, but they're
        # here to allow testing that passing in MORE columns is ok.
        SAMPLE_TOTAL_READS_KEY: [3216923, 611913],
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY: [5, 4.76],
        GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY: [7.1867431342E-06,
                                             4.7470988923E-06],
        GDNA_MASS_TO_SAMPLE_VOL_RATIO_KEY: [1.0000000000E-09,
                                            9.3333333333E-10],
        GDNA_MASS_TO_SAMPLE_SURFACE_AREA_RATIO_KEY: [5.0000000000E-08,
                                                     2.3333333333E-08]
    }

    # These values are taken from cell directly under the
    # header "Applying the linear models to sequencing data" of the
    # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
    # notebook by capturing the value of the dataframe when i=1.
    # The "GenomeLenght" [sic] column is the OGU_LEN_IN_BP_KEY and the
    # "Species" column is OGU_ID_KEY.
    ogu_lengths_dict = {
        OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                     "Escherichia coli", "Tyzzerella nexilis",
                     "Prevotella sp. oral taxon 299",
                     "Streptococcus mitis", "Leptolyngbya valderiana",
                     "Neisseria subflava", "Neisseria flavescens",
                     "Fusobacterium periodonticum",
                     "Streptococcus pneumoniae", "Haemophilus influenzae",
                     "Veillonella dispar"],
        OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                            2453028, 2031251, 89264, 2292986, 2204851,
                            2484878.333, 2058778.25, 1680673.6, 2116567]
    }

    ogu_percent_coverage_dict = {
        OGU_ID_KEY: ogu_lengths_dict[OGU_ID_KEY],
        # The percent values here are completely made up; I modified the
        # original coverage values in the tests (which were extracted from the
        # Zaramela SynDNA_saliva_samples_analysis.ipynb notebook), changing
        # them to be in the 0 to 100 range and making sure that the
        # Neisseria subflava and Haemophilus influenzae were lower than all
        # the rest so they could continue to be the filtering tests.
        OGU_PERCENT_COVERAGE_KEY: [92.597514334489, 91.032039014754,
                                   65.862497316419, 61.537729965377,
                                   19.692890582578, 18.090986293668,
                                   33.880713389496, 7.934318831428,
                                   15.731341482939, 11.420333792332,
                                   10.484891342717, 1.394039151921,
                                   10.533221485547]
    }

    ogu_percent_coverage_per_sample_df = \
        pd.DataFrame(ogu_percent_coverage_dict)
    ogu_percent_coverage_per_sample_df["example1"] = \
        ogu_percent_coverage_per_sample_df[OGU_PERCENT_COVERAGE_KEY]
    ogu_percent_coverage_per_sample_df["example2"] = \
        ogu_percent_coverage_per_sample_df[OGU_PERCENT_COVERAGE_KEY]
    ogu_percent_coverage_per_sample_df.drop(
        columns=[OGU_PERCENT_COVERAGE_KEY], inplace=True)

    # This dict contains counts (from Zaramela, see above) for example 1
    example1_ogu_full_inputs_dict = ogu_lengths_dict.copy()
    # These values are taken from cell directly under the
    # header "Applying the linear models to sequencing data" of the
    # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
    # notebook by capturing the value of the dataframe when i=1.
    # The RawCounts column is OGU_READ_COUNT_KEY.
    example1_ogu_full_inputs_dict.update({
        OGU_READ_COUNT_KEY: [79950, 93024, 86188, 45441, 31185, 24929,
                             1975, 26130, 22303, 19783, 14478, 12145,
                             14609]})

    # This dict contains counts (NOT directly from Zaramela, see above) for
    # example 2
    example2_ogu_full_inputs_dict = ogu_lengths_dict.copy()
    example2_ogu_full_inputs_dict.update({
        # Example 2 counts are slightly manually modified from example 1,
        # differing by 1 count at position 1 (1-based, for L. gasseri),
        # 26130 counts at position 8 (for N. subflava, bc example 2 has
        # zero counts here), by a factor of 10 at position 10
        # (for F. periodonticum, bc example 2 has 10x the counts of example 1
        # at this position), and by 12045 at position 12 (for H. influenzae,
        # bc example 2 has 100 counts here instead of 12145).
        # All other positions are identical.
        OGU_READ_COUNT_KEY: [79951, 93024, 86188, 45441, 31185, 24929,
                             1975, 0, 22303, 197830, 14478, 100, 14609]
    })

    # This dict contains the gdna mass from Zaramela and the results of
    # calculations done using the truncated Avogadro's number
    # (6.022e23) used in the SynDNA notebooks (instead of the full value).
    example1_ogu_full_outputs_short_avogadro_dict = (
        example1_ogu_full_inputs_dict.copy())
    example1_ogu_full_outputs_short_avogadro_dict.update({
        # These values are taken from cell directly under the
        # header "Applying the linear models to sequencing data" of the
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values).
        OGU_GDNA_MASS_NG_KEY: [0.54168919427718800000,
                               0.65408448128256300000,
                               0.59479652205780900000,
                               0.26809831701850500000,
                               0.16778538250235500000,
                               0.12697002941967100000,
                               0.00540655563393439000,
                               0.13462933928829500000,
                               0.11054062877622500000,
                               0.09521377825242420000,
                               0.06455278517415270000,
                               0.05187010729728740000,
                               0.06528070535624650000],
        # for the arrays below, the 1st, 2nd, and 7th values (those for
        # L. gasseri, R. albus, and L. valderiana) match those worked out in
        # detail for example 1 on the "full_calcs on correct masses" sheet in
        # the "absolute_quant_example.xlsx" spreadsheet (in the section using
        # the truncated Avogadro's #, as Zaramela did).
        # The remainder were not worked out individually and come from the code
        # calculations.
        OGU_GENOMES_PER_G_OF_GDNA_KEY: [5.26939603e+13, 2.77101748e+13,
                                        2.18971931e+13, 1.28661875e+13,
                                        1.26738630e+13, 1.15823056e+13,
                                        1.12228129e+13, 1.08791685e+13,
                                        9.28966547e+12, 7.09989719e+12,
                                        5.80982169e+12, 5.71861894e+12,
                                        5.71492069e+12],
        OGU_CELLS_PER_G_OF_GDNA_KEY: [5.26939603e+13, 2.77101748e+13,
                                      2.18971931e+13, 1.28661875e+13,
                                      1.26738630e+13, 1.15823056e+13,
                                      1.12228129e+13, 1.08791685e+13,
                                      9.28966547e+12, 7.09989719e+12,
                                      5.80982169e+12, 5.71861894e+12,
                                      5.71492069e+12],
        OGU_CELLS_PER_G_OF_SAMPLE_KEY: [378697957.6284817,
                                        199145908.712499,
                                        157369502.5167467,
                                        92465984.77627191,
                                        91083797.64630365,
                                        83239054.98940048,
                                        80655473.50500171,
                                        78185789.2024791,
                                        66762439.53068219,
                                        51025137.39109518,
                                        41753696.145493336,
                                        41098245.40604969,
                                        41071667.042116195]
    })

    # This dict contains the gdna mass from the
    # "PredictedOguMass" column of the
    # "Mass results for sample A regression for full data on log10_read_count"
    # table on the "linear regressions counts" sheet of
    # "absolute_quant_example.xlsx", and
    # the results of calculations done on example 1
    # using the full Avogadro's number instead of the truncated one.
    example1_ogu_full_outputs_full_avogadro_dict = (
        example1_ogu_full_inputs_dict.copy())
    example1_ogu_full_outputs_full_avogadro_dict.update({
        OGU_GDNA_MASS_NG_KEY: [0.055906776,
                               0.067506894,
                               0.061387889,
                               0.027669949,
                               0.01731683,
                               0.01310435,
                               0.000558001,
                               0.013894854,
                               0.011408701,
                               0.009826844,
                               0.006662378,
                               0.005353421,
                               0.006737505],
        OGU_AGNOSTIC_COVERAGE_KEY:
            ogu_percent_coverage_dict[OGU_PERCENT_COVERAGE_KEY],
        # for the arrays below, the 1st, 2nd, and 7th values (those for
        # L. gasseri, R. albus, and L. valderiana) match those worked out in
        # detail for example1 in the "full_calcs on correct masses" sheet of
        # the "absolute_quant_example.xlsx" spreadsheet
        # (in the section using the FULL Avogadro's #, NOT matching Zaramela).
        # The remainder were not worked out individually and come from the code
        # calculations.
        OGU_GENOMES_PER_G_OF_GDNA_KEY: [5438576851832.26,
                                        2859984606316.35,
                                        2260023103719.98,
                                        1327927321117.10,
                                        1308077383248.35,
                                        1195417056032.46,
                                        1158313590928.33,
                                        1122845831970.39,
                                        958792227127.95,
                                        732784863204.92,
                                        599635357838.64,
                                        590222264508.60,
                                        589840565922.86],
        OGU_CELLS_PER_G_OF_GDNA_KEY: [5438576851832.26,
                                      2859984606316.35,
                                      2260023103719.98,
                                      1327927321117.10,
                                      1308077383248.35,
                                      1195417056032.46,
                                      1158313590928.33,
                                      1122845831970.39,
                                      958792227127.95,
                                      732784863204.92,
                                      599635357838.64,
                                      590222264508.60,
                                      589840565922.86],
        OGU_CELLS_PER_G_OF_SAMPLE_KEY: [39085654.85,
                                        20553974.73,
                                        16242205.52,
                                        9543472.56,
                                        9400816.15,
                                        8591155.32,
                                        8324502.25,
                                        8069604.57,
                                        6890593.46,
                                        5266336.58,
                                        4309425.29,
                                        4241775.81,
                                        4239032.64],
        OGU_CELLS_PER_UL_OF_SAMPLE_KEY: [5438.576852,
                                         2859.984606,
                                         2260.023104,
                                         1327.927321,
                                         1308.077383,
                                         1195.417056,
                                         1158.313591,
                                         1122.845832,
                                         958.792227,
                                         732.784863,
                                         599.635358,
                                         590.222265,
                                         589.840566],
        OGU_CELLS_PER_CM2_OF_SAMPLE_KEY: [271928.842592,
                                          142999.230316,
                                          113001.155186,
                                          66396.366056,
                                          65403.869162,
                                          59770.852802,
                                          57915.679546,
                                          56142.291599,
                                          47939.611356,
                                          36639.243160,
                                          29981.767892,
                                          29511.113225,
                                          29492.028296]
    })

    # NB: the reason there is no "example2_ogu_full_<etc>" is that
    # example 2 isn't being used to test generating an unfiltered dataset;
    # all those tests are being done with example 1.

    example1_ogu_filtered_outputs_full_avogadro_dict = {
        OGU_ID_KEY: _remove_filtered_entries(ogu_lengths_dict[OGU_ID_KEY]),
        OGU_LEN_IN_BP_KEY:
            _remove_filtered_entries(ogu_lengths_dict[OGU_LEN_IN_BP_KEY]),
        OGU_AGNOSTIC_COVERAGE_KEY:
            _remove_filtered_entries(
                ogu_percent_coverage_dict[OGU_PERCENT_COVERAGE_KEY]),
        OGU_READ_COUNT_KEY:
            _remove_filtered_entries(
                example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        OGU_GDNA_MASS_NG_KEY:
            _remove_filtered_entries(
                example1_ogu_full_outputs_full_avogadro_dict[
                    OGU_GDNA_MASS_NG_KEY]),
        OGU_GENOMES_PER_G_OF_GDNA_KEY:
            _remove_filtered_entries(
                example1_ogu_full_outputs_full_avogadro_dict[
                    OGU_GENOMES_PER_G_OF_GDNA_KEY]),
        OGU_CELLS_PER_G_OF_GDNA_KEY:
            _remove_filtered_entries(
                example1_ogu_full_outputs_full_avogadro_dict[
                    OGU_CELLS_PER_G_OF_GDNA_KEY]),
        OGU_CELLS_PER_G_OF_SAMPLE_KEY:
            _remove_filtered_entries(
                example1_ogu_full_outputs_full_avogadro_dict[
                    OGU_CELLS_PER_G_OF_SAMPLE_KEY]),
        OGU_CELLS_PER_UL_OF_SAMPLE_KEY:
            _remove_filtered_entries(
                example1_ogu_full_outputs_full_avogadro_dict[
                    OGU_CELLS_PER_UL_OF_SAMPLE_KEY]),
        OGU_CELLS_PER_CM2_OF_SAMPLE_KEY:
            _remove_filtered_entries(
                example1_ogu_full_outputs_full_avogadro_dict[
                    OGU_CELLS_PER_CM2_OF_SAMPLE_KEY])
    }

    # This dict contains the results of calculations done on *filtered*
    # example2 data using the full Avogadro's number instead of the truncated
    # one. The ogu id, ogu length, and ogu read counts are exactly the same as
    # in the full data *except* that Neisseria subflava and
    # Haemophilus influenzae have been removed (for falling below the
    # min_coverage threshold, which is the default min_coverage value in
    # our analysis system).
    example2_ogu_filtered_inputs_outputs_full_avogadro_dict = {
        OGU_ID_KEY: _remove_filtered_entries(ogu_lengths_dict[OGU_ID_KEY]),
        OGU_LEN_IN_BP_KEY:
            _remove_filtered_entries(ogu_lengths_dict[OGU_LEN_IN_BP_KEY]),
        OGU_AGNOSTIC_COVERAGE_KEY:
            _remove_filtered_entries(
                ogu_percent_coverage_dict[OGU_PERCENT_COVERAGE_KEY]),
        OGU_READ_COUNT_KEY:
            _remove_filtered_entries(
                example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        # NB: unlike the analogous example1 dictionary, this one does NOT have
        # an entry for OGU_GDNA_MASS_NG_KEY; the mass values are intermediate
        # values, not final deliverables, and the code generates them is
        # tested with example1 alone.

        # for the arrays below, the 1st, 2nd, and 7th values (those for
        # L. gasseri, R. albus, and L. valderiana) match those worked out in
        # detail for example 2 in the "full_calcs on correct masses" sheet of
        # the "absolute_quant_example.xlsx" spreadsheet
        # (in the section using the FULL Avogadro's #, NOT matching Zaramela).
        # The remainder were not worked out individually and come from the code
        # calculations.
        OGU_GENOMES_PER_G_OF_GDNA_KEY: [4.698763e+12,
                                        2.471605e+12,
                                        1.952836e+12,
                                        1.146051e+12,
                                        1.128120e+12,
                                        1.030524e+12,
                                        9.937836e+11,
                                        8.263655e+11,
                                        1.114513e+13,
                                        5.163945e+11,
                                        5.079680e+11],
        OGU_CELLS_PER_G_OF_GDNA_KEY: [4.698763e+12,
                                      2.471605e+12,
                                      1.952836e+12,
                                      1.146051e+12,
                                      1.128120e+12,
                                      1.030524e+12,
                                      9.937836e+11,
                                      8.263655e+11,
                                      1.114513e+13,
                                      5.163945e+11,
                                      5.079680e+11],
        OGU_CELLS_PER_G_OF_SAMPLE_KEY: [2.230549e+07,
                                        1.173295e+07,
                                        9.270306e+06,
                                        5.440416e+06,
                                        5.355296e+06,
                                        4.891999e+06,
                                        4.717589e+06,
                                        3.922839e+06,
                                        5.290705e+07,
                                        2.451376e+06,
                                        2.411375e+06],
        OGU_CELLS_PER_UL_OF_SAMPLE_KEY: [4385.512032,
                                         2306.831068,
                                         1822.647020,
                                         1069.647383,
                                         1052.911702,
                                         961.822421,
                                         927.531325,
                                         771.274450,
                                         10402.124237,
                                         481.968209,
                                         474.103500],
        OGU_CELLS_PER_CM2_OF_SAMPLE_KEY: [109637.800794,
                                          57670.776705,
                                          45566.175504,
                                          26741.184585,
                                          26322.792552,
                                          24045.560534,
                                          23188.283114,
                                          19281.861261,
                                          260053.105920,
                                          12049.205223,
                                          11852.587503]
    }

    # This dict contains the results of calculations done on *filtered*
    # example1 and example2 data, in the order that they appear in biom
    # outputs, which are alphabetized by ogu.  Note that data for N. subflava
    # and H. influenzae are commented out since they are filtered out by the
    # min_coverage threshold.
    reordered_results_dict = {
        SAMPLE_ID_KEY: ["example1", "example2"],
        # in a biom table, results are ordered alphabetically
        OGU_ID_KEY: [
            'Escherichia coli', 'Fusobacterium periodonticum',
            # 'Haemophilus influenzae',
            'Lactobacillus gasseri',
            'Leptolyngbya valderiana', 'Neisseria flavescens',
            # 'Neisseria subflava',
            'Prevotella sp. oral taxon 299',
            'Ruminococcus albus', 'Streptococcus mitis',
            'Streptococcus pneumoniae', 'Tyzzerella nexilis',
            'Veillonella dispar'],
        # The values in the first position of each sub-array below is that from
        # that ogu in self.example1_ogu_filtered_outputs_full_avogadro_dict,
        # while the value in the second position of each sub-array below is
        # that from that ogu in
        # self.example2_ogu_filtered_inputs_outputs_full_avogadro_dict.  Note
        # that with reordering, the 4th sub-array is the one for L. gasseri,
        # the 5th is for L. valderiana, and the 9th is for R. albus.
        # The commented out values are for N. subflava and H. influenzae,
        # which are removed due to low coverage.
        OGU_CELLS_PER_G_OF_GDNA_KEY: [
            [2260023103719.98, 1952836093054.1],
            [732784863204.92, 11145133111026.7],
            # [590222264508.6, 0],
            [5438576851832.26, 4698762891240.72],
            [1158313590928.33, 993783562051.94],
            [958792227127.95, 826365482621.33],
            # [1122845831970.39, 0],
            [1308077383248.35, 1128119680829.86],
            [2859984606316.35, 2471604715978.24],
            [1195417056032.46, 1030524022882.84],
            [599635357838.64, 516394509546.61],
            [1327927321117.1, 1146050767964.49],
            [589840565922.86, 507968035834.47]
        ]
    }

    example1_example4_results_dict = {
        OGU_ID_KEY: reordered_results_dict[OGU_ID_KEY],
        # NB: The test values for example1 here are *slightly* different than
        # those in self.example1_ogu_filtered_outputs_full_avogadro_dict bc
        # the gdna-to-sample mass ratio calculated internally during this
        # soup-to-nuts function has more digits past the decimal than does the
        # example1 entry in the manually-populated self.mass_and_totals_dict.
        # Since we are multiplying/dividing by large numbers like e.g., 10^9
        # (to change ng to g), this ends up making a slight difference in the
        # end product: for example, for L.gasseri,
        # 3908565*5.46* cells instead of 3908565*4.85* cells,
        # 8324502.*38* instead of 8324502.*25* for L. valderiana,
        # and 2055397*5.06* instead of 2055397*4.73* for R. albus.
        # Remember, with reordering, the 4th sub-array is for L. gasseri,
        # the 5th is for L. valderiana, and the 9th is for R. albus.
        # The commented out values are for N. subflava and H. influenzae,
        # which are removed due to low coverage.
        OGU_CELLS_PER_G_OF_SAMPLE_KEY: [
            [16242205.78, 6489214.14],
            [5266336.67, 37034933.76],
            # [4241775.87, 0],
            [39085655.46, 15613844.24],
            [8324502.38, 3302312.14],
            [6890593.56, 2745987.02],
            # [8069604.7, 0],
            [9400816.3, 3748706.92],
            [20553975.06, 8213066.28],
            [8591155.45, 3424399.56],
            [4309425.36, 1715963.04],
            [9543472.71, 3808291.37],
            [4239032.7, 1687962.12]
        ],
        OGU_CELLS_PER_CM2_OF_SAMPLE_KEY: [
            [113001.155186, 31896.322853],
            [36639.24316, 182037.174147],
            #
            [271928.842592, 76746.460557],
            [57915.679546, 16231.79818],
            [47939.611356, 13497.302883],
            #
            [65403.869162, 18425.954787],
            [142999.230316, 40369.543694],
            [59770.852802, 16831.892374],
            [29981.767892, 8434.443656],
            [66396.366056, 18718.82921],
            [29492.028296, 8296.811252],
        ],
        OGU_CELLS_PER_UL_OF_SAMPLE_KEY: [
            [2260.023104, 1275.852914],
            [732.784863, 7281.486966],
            #
            [5438.576852, 3069.858422],
            [1158.313591, 649.271927],
            [958.792227, 539.892115],
            #
            [1308.077383, 737.038191],
            [2859.984606, 1614.781748],
            [1195.417056, 673.275695],
            [599.635358, 337.377746],
            [1327.927321, 748.753168],
            [589.840566, 331.87245]
        ]}

    @classmethod
    def combine_inputs(cls):
        sample_names = cls.generate_sample_names_list(use_filtered=False)
        col_list = [OGU_ID_KEY, OGU_READ_COUNT_KEY, OGU_LEN_IN_BP_KEY]
        parallel_dicts_list = [TestCalcCellCountsData.example1_ogu_full_inputs_dict,
                               TestCalcCellCountsData.example2_ogu_full_inputs_dict]
        input_dict = {SAMPLE_ID_KEY: sample_names}
        for col_name in col_list:
            curr_col_list = []
            for curr_dict in parallel_dicts_list:
                curr_col_list.extend(curr_dict[col_name].copy())
            input_dict[col_name] = curr_col_list

        return input_dict

    @classmethod
    def combine_filtered_out(cls, col_name):
        example1_copy = (
            TestCalcCellCountsData.example1_ogu_filtered_outputs_full_avogadro_dict[col_name].copy())
        example2_copy = (
            TestCalcCellCountsData.example2_ogu_filtered_inputs_outputs_full_avogadro_dict[
                col_name].copy())
        example1_copy.extend(example2_copy)
        return example1_copy

    @classmethod
    def make_combined_counts_np_array(cls):
        # combine each item in the OGU_READ_COUNT_KEY array for
        # self.example1_ogu_full_inputs_dict with the analogous item in
        # self.example2_ogu_full_inputs_dict to make an array of two-item
        # arrays, and turn this into an np.array
        counts_array = np.array(
            [list(x) for x in zip(
                cls.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY],
                cls.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY])])
        return counts_array

    @classmethod
    def generate_sample_names_list(cls, use_filtered=True):
        sample_names = ["example1", "example2"]
        parallel_dicts_list = []
        if use_filtered:
            parallel_dicts_list = [
                cls.example1_ogu_filtered_outputs_full_avogadro_dict,
                cls.example2_ogu_filtered_inputs_outputs_full_avogadro_dict]
        else:
            parallel_dicts_list = [
                cls.example1_ogu_full_inputs_dict,
                cls.example2_ogu_full_inputs_dict]

        output = []
        for curr_index in range(len(sample_names)):
            curr_sample_name = sample_names[curr_index]
            curr_parallel_dict = parallel_dicts_list[curr_index]
            curr_names_list = [curr_sample_name for _ in
                               range(len(curr_parallel_dict[OGU_ID_KEY]))]
            output.extend(curr_names_list)
        return output


class TestCalcCellCounts(TestCase):
    _DEFAULT_SAMPLE_PLUS_PREP_COLS = [
        SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
        GDNA_CONCENTRATION_NG_UL_KEY, ELUTE_VOL_UL_KEY,
        INPUT_SYNDNA_POOL_MASS_NG_KEY]

    @classmethod
    def make_sample_plus_prep_input_dict(cls, cols_to_include=None):
        if cols_to_include is None:
            cols_to_include = cls._DEFAULT_SAMPLE_PLUS_PREP_COLS
        sample_plus_prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k].copy()
            for k in cols_to_include}
        return sample_plus_prep_info_dict

    def setUp(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita(self):
        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70

        prep_info_dict = self.make_sample_plus_prep_input_dict()
        # NOTE: this column is not needed anymore. It is left in this test
        # just to show that the code can deal with extra columns (it just
        # ignores them).
        prep_info_dict[SAMPLE_TOTAL_READS_KEY] = \
            TestCalcCellCountsData.mass_and_totals_dict[SAMPLE_TOTAL_READS_KEY]

        # reset the sample ids and elute volume for example4
        sample_ids = ["example1", "example4"]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_G_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.example1_example4_results_dict[OGU_ID_KEY],
            sample_ids)

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita(
            prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_w_casts(self):
        # inputs are the same as in
        # test_calc_ogu_cell_counts_per_g_of_sample_for_qiita EXCEPT that
        # all the inputs are strings, including ones that must be ints/floats.
        # These are automatically cast to what they need to be by the function.

        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70

        prep_info_dict = {
            k: [str(x) for x in v] for k, v in
            self.make_sample_plus_prep_input_dict().items()}
        sample_ids = ["example1", "example4"]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = str(example4_elute_vol)

        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict).astype(str)

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_G_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.example1_example4_results_dict[OGU_ID_KEY],
            sample_ids)

        # pass in strings for the numeric values to ensure they get cast
        min_coverage = "10"
        min_rsquared = "0.8"

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita(
            prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_w_negs(self):
        # inputs are the same as in
        # test_calc_ogu_cell_counts_per_g_of_sample_for_qiita EXCEPT that
        # the "example4" sample has a negative aliquot mass and thus its
        # results are removed from the output biom table

        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70

        prep_info_dict = self.make_sample_plus_prep_input_dict()
        sample_ids = ["example1", "example4"]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol
        prep_info_dict[SAMPLE_IN_ALIQUOT_MASS_G_KEY][1] = \
            -1 * prep_info_dict[SAMPLE_IN_ALIQUOT_MASS_G_KEY][1]

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        # Results are returned only for example 1 because example 4 has a
        # negative aliquot mass
        ogu_cell_counts_per_g_sample = np.array(
            [[x[0]] for x in
             TestCalcCellCountsData.example1_example4_results_dict[
                 OGU_CELLS_PER_G_OF_SAMPLE_KEY]]
        )

        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            ogu_cell_counts_per_g_sample,
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            [sample_ids[0]])

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita(
            prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "Dropping samples with negative values in necessary "
            "prep/sample column(s): example4\nThe following items have "
            "coverage lower than the minimum of 10.0: ['Neisseria subflava', "
            "'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_w_prep_err(self):
        # missing required columns--deliberately not using helper since this
        # test needs an incomplete dict
        prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k]
            for k in [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            prep_info_dict[SAMPLE_ID_KEY])
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        min_coverage = 1
        min_rsquared = 0.8

        err_msg = r"prep info is missing required column\(s\): " \
                  r"\['calc_mass_sample_aliquot_input_g'\]"
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_per_g_of_sample_for_qiita(
                prep_info_df, models_fp, counts_biom,
                coverages_df, lengths_fp, min_coverage, min_rsquared)

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_w_ids_err(self):
        prep_info_dict = self.make_sample_plus_prep_input_dict()

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        # remove one of the sample ids from the prep info; this will cause
        # an error because the biom table has a sample id not in prep info
        prep_info_df = pd.DataFrame(prep_info_dict)
        prep_info_df.drop(index=0, axis=0, inplace=True)

        sample_ids = TestCalcCellCountsData.sample_and_prep_input_dict[
            SAMPLE_ID_KEY]
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        min_coverage = 10
        min_rsquared = 0.8

        err_msg = (r"Found sample ids in OGU counts data that were not in"
                   r" sample info: \{'example1'\}")
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_per_g_of_sample_for_qiita(
                prep_info_df, models_fp, counts_biom,
                coverages_df, lengths_fp, min_coverage, min_rsquared)

    def test_calc_ogu_cell_counts_per_cm2_of_sample_for_qiita(self):
        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70

        prep_info_dict = self.make_sample_plus_prep_input_dict(
            [SAMPLE_SURFACE_AREA_CM2_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
             ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY])
        # NOTE: this column is not needed anymore. It is left in this test
        # just to show that the code can deal with extra columns (it just
        # ignores them).
        prep_info_dict[SAMPLE_TOTAL_READS_KEY] = \
            TestCalcCellCountsData.mass_and_totals_dict[SAMPLE_TOTAL_READS_KEY]
        sample_ids = ["example1", "example4"]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_CM2_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            sample_ids)

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_cm2_of_sample_for_qiita(
            prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_ul_of_sample_for_qiita(self):
        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70

        prep_info_dict = self.make_sample_plus_prep_input_dict(
            [SAMPLE_VOLUME_UL_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
             ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY])
        # NOTE: this column is not needed anymore. It is left in this test
        # just to show that the code can deal with extra columns (it just
        # ignores them).
        prep_info_dict[SAMPLE_TOTAL_READS_KEY] = \
            TestCalcCellCountsData.mass_and_totals_dict[SAMPLE_TOTAL_READS_KEY]
        sample_ids = ["example1", "example4"]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_UL_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            sample_ids)

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_ul_of_sample_for_qiita(
            prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input(self):
        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70
        sample_ids = ["example1", "example4"]
        sample_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k].copy()
            for k in [SAMPLE_IN_ALIQUOT_MASS_G_KEY]}
        sample_info_dict[SAMPLE_ID_KEY] = sample_ids

        prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k].copy()
            for k in [GDNA_CONCENTRATION_NG_UL_KEY,
                      ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY]}

        # NOTE: this column is not needed anymore. It is left in this test
        # just to show that the code can deal with extra columns (it just
        # ignores them).
        prep_info_dict[SAMPLE_TOTAL_READS_KEY] = \
            TestCalcCellCountsData.mass_and_totals_dict[SAMPLE_TOTAL_READS_KEY]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_G_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.example1_example4_results_dict[OGU_ID_KEY],
            sample_ids)

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input(
            sample_info_df, prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input_w_casts(self):
        # inputs are the same as in
        # test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input
        # EXCEPT that all the inputs are strings, including ones that must be
        # ints/floats. These are automatically cast to what they need to be.

        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70
        sample_ids = ["example1", "example4"]
        sample_info_dict = {
            k: [str(x) for x in
                TestCalcCellCountsData.sample_and_prep_input_dict[k]]
            for k in [SAMPLE_IN_ALIQUOT_MASS_G_KEY]}
        sample_info_dict[SAMPLE_ID_KEY] = sample_ids

        prep_info_dict = {
            k: [str(x) for x in
                TestCalcCellCountsData.sample_and_prep_input_dict[k]]
            for k in [GDNA_CONCENTRATION_NG_UL_KEY,
                      ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY]}
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = str(example4_elute_vol)

        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict).astype(str)

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_G_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.example1_example4_results_dict[OGU_ID_KEY],
            sample_ids)

        # pass in strings for the numeric values to ensure they get cast
        min_coverage = "10"
        min_rsquared = "0.8"

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input(
            sample_info_df, prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input_w_negs(self):
        # inputs are the same as in
        # test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input EXCEPT that
        # the "example4" sample has a negative aliquot mass and thus its
        # results are removed from the output biom table

        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70
        sample_ids = ["example1", "example4"]
        sample_info_dict = \
            {k: [x for x in TestCalcCellCountsData.sample_and_prep_input_dict[k]] for
             k in [SAMPLE_IN_ALIQUOT_MASS_G_KEY]}
        sample_info_dict[SAMPLE_ID_KEY] = sample_ids
        sample_info_dict[SAMPLE_IN_ALIQUOT_MASS_G_KEY][1] = \
            -1 * sample_info_dict[SAMPLE_IN_ALIQUOT_MASS_G_KEY][1]

        prep_info_dict = {
            k: [x for x in
                TestCalcCellCountsData.sample_and_prep_input_dict[k]]
            for k in [GDNA_CONCENTRATION_NG_UL_KEY,
                      ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY]}
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        # Results are returned only for example 1 because example 4 has a
        # negative aliquot mass
        ogu_cell_counts_per_g_sample = np.array(
            [[x[0]] for x in
             TestCalcCellCountsData.example1_example4_results_dict[
                 OGU_CELLS_PER_G_OF_SAMPLE_KEY]]
        )

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            ogu_cell_counts_per_g_sample,
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            [sample_ids[0]])

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input(
            sample_info_df, prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "Dropping samples with negative values in necessary "
            "prep/sample column(s): example4\nThe following items have "
            "coverage lower than the minimum of 10.0: ['Neisseria subflava', "
            "'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input_w_sample_err(self):  # noqa: E501
        # missing a required column column
        sample_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict
            for k in [SAMPLE_ID_KEY]}

        prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k]
            for k in [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
                      ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()
        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            prep_info_dict[SAMPLE_ID_KEY])
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        min_coverage = 10
        min_rsquared = 0.8

        err_msg = r"sample info is missing required column\(s\): " \
                  r"\['calc_mass_sample_aliquot_input_g'\]"
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input(
                sample_info_df, prep_info_df, models_fp, counts_biom,
                coverages_df, lengths_fp, min_coverage, min_rsquared)

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input_w_prep_err(self):  # noqa: E501
        sample_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k]
            for k in [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]}

        # missing required columns
        prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k]
            for k in [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            prep_info_dict[SAMPLE_ID_KEY])
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        min_coverage = 1
        min_rsquared = 0.8

        err_msg = r"prep info is missing required column\(s\): " \
                  r"\[\'mass_syndna_input_ng'\, 'vol_extracted_elution_ul'\]"
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input(
                sample_info_df, prep_info_df, models_fp, counts_biom,
                coverages_df, lengths_fp, min_coverage, min_rsquared)

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input_w_ids_err(self):
        sample_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k]
            for k in [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]}

        prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k]
            for k in [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
                      ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        # remove one of the sample ids from the sample info; this will cause
        # an error (whereas the reverse--sample id in sample info but not in
        # prep info--will NOT)
        sample_info_df = pd.DataFrame(sample_info_dict)
        sample_info_df.drop(index=0, axis=0, inplace=True)

        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            prep_info_dict[SAMPLE_ID_KEY])
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        min_coverage = 10
        min_rsquared = 0.8

        err_msg = (r"Found sample ids in prep info that were not in"
                   r" sample info: \{'example1'\}")
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_per_g_of_sample_for_qiita_split_input(
                sample_info_df, prep_info_df, models_fp, counts_biom,
                coverages_df, lengths_fp, min_coverage, min_rsquared)

    def test_calc_ogu_cell_counts_per_cm2_of_sample_for_qiita_split_input(self):
        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70
        sample_ids = ["example1", "example4"]
        sample_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k].copy() for
            k in [SAMPLE_SURFACE_AREA_CM2_KEY]}
        sample_info_dict[SAMPLE_ID_KEY] = sample_ids

        prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k].copy() for k in
            [GDNA_CONCENTRATION_NG_UL_KEY,
             ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY]}

        # NOTE: this column is not needed anymore. It is left in this test
        # just to show that the code can deal with extra columns (it just
        # ignores them).
        prep_info_dict[SAMPLE_TOTAL_READS_KEY] = \
            TestCalcCellCountsData.mass_and_totals_dict[SAMPLE_TOTAL_READS_KEY]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_CM2_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            sample_ids)

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_cm2_of_sample_for_qiita_split_input(
            sample_info_df, prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_ul_of_sample_for_qiita_split_input(self):
        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70
        sample_ids = ["example1", "example4"]
        sample_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k].copy() for
            k in [SAMPLE_VOLUME_UL_KEY]}
        sample_info_dict[SAMPLE_ID_KEY] = sample_ids

        prep_info_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k].copy() for k in
            [GDNA_CONCENTRATION_NG_UL_KEY,
             ELUTE_VOL_UL_KEY, INPUT_SYNDNA_POOL_MASS_NG_KEY]}

        # NOTE: this column is not needed anymore. It is left in this test
        # just to show that the code can deal with extra columns (it just
        # ignores them).
        prep_info_dict[SAMPLE_TOTAL_READS_KEY] = \
            TestCalcCellCountsData.mass_and_totals_dict[SAMPLE_TOTAL_READS_KEY]
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.example1_example4_results_dict[
                OGU_CELLS_PER_UL_OF_SAMPLE_KEY]),
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            sample_ids)

        min_coverage = 10
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_ul_of_sample_for_qiita_split_input(
            sample_info_df, prep_info_df, models_fp, counts_biom,
            coverages_df, lengths_fp, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})

        a_tester = Testers()
        a_tester.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY],
            decimal_precision=1)
        self.assertEqual(
            "The following items have coverage lower than the minimum of "
            "10.0: ['Neisseria subflava', 'Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_biom(self):
        params_dict = {k: TestCalcCellCountsData.sample_and_prep_input_dict[k] for k in
                       [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
                        SAMPLE_VOLUME_UL_KEY, SAMPLE_SURFACE_AREA_CM2_KEY,
                        GDNA_CONCENTRATION_NG_UL_KEY, ELUTE_VOL_UL_KEY,
                        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        params_df = pd.DataFrame(params_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            params_dict[SAMPLE_ID_KEY])
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.reordered_results_dict[OGU_CELLS_PER_G_OF_GDNA_KEY]),
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            TestCalcCellCountsData.reordered_results_dict[SAMPLE_ID_KEY])

        min_coverage = 10
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        # Note: 1) this is outputting the ogu_cell_counts_per_g_gdna, not the
        # ogu_cell_counts_per_g_sample (which is what is output by the qiita
        # version of this function) because I want to check that I really can
        # choose to get something else, and 2) this is using the full version
        # of Avogadro's #, not the truncated version that was used in the
        # notebook, so the results are slightly different (but more realistic)
        output_biom, output_msgs = calc_ogu_cell_counts_biom(
            params_df, TestCalcCellCountsData.linregresses_dict, counts_biom,
            coverages_df, lengths_df, min_coverage, min_rsquared,
            output_metric)

        # NB: only checking results to 1 decimal because Ubuntu and Mac
        # differ past that point. Not that it matters much since the decimal
        # portion of values this huge is not very important.
        a_tester = Testers()
        a_tester.assert_biom_tables_equal(expected_out_biom, output_biom,
                                          decimal_precision=1)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of "
             "10.0: ['Neisseria subflava', 'Haemophilus influenzae']"],
            output_msgs)

    def test_calc_ogu_cell_counts_biom_w_per_sample_coverages(self):
        params_dict = {k: TestCalcCellCountsData.sample_and_prep_input_dict[k] for k in
                       [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
                        SAMPLE_VOLUME_UL_KEY, SAMPLE_SURFACE_AREA_CM2_KEY,
                        GDNA_CONCENTRATION_NG_UL_KEY, ELUTE_VOL_UL_KEY,
                        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        params_df = pd.DataFrame(params_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            params_dict[SAMPLE_ID_KEY])
        coverages_per_sample_df = \
            TestCalcCellCountsData.ogu_percent_coverage_per_sample_df.copy()
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.reordered_results_dict[OGU_CELLS_PER_G_OF_GDNA_KEY]),
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            TestCalcCellCountsData.reordered_results_dict[SAMPLE_ID_KEY])

        min_coverage = 10
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        # Note: 1) this is outputting the ogu_cell_counts_per_g_gdna, not the
        # ogu_cell_counts_per_g_sample (which is what is output by the qiita
        # version of this function) because I want to check that I really can
        # choose to get something else, and 2) this is using the full version
        # of Avogadro's #, not the truncated version that was used in the
        # notebook, so the results are slightly different (but more realistic)
        output_biom, output_msgs = calc_ogu_cell_counts_biom(
            params_df, TestCalcCellCountsData.linregresses_dict, counts_biom,
            coverages_per_sample_df, lengths_df, min_coverage, min_rsquared,
            output_metric)

        # NB: only checking results to 1 decimal because Ubuntu and Mac
        # differ past that point. Not that it matters much since the decimal
        # portion of values this huge is not very important.
        a_tester = Testers()
        a_tester.assert_biom_tables_equal(expected_out_biom, output_biom,
                                          decimal_precision=1)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of "
             "10.0: ['Neisseria subflava', 'Haemophilus influenzae']"],
            output_msgs)

    def test_calc_ogu_cell_counts_biom_w_col_err(self):
        # missing SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY col
        params_dict = {k: TestCalcCellCountsData.sample_and_prep_input_dict[k] for k in
                       [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
                        GDNA_CONCENTRATION_NG_UL_KEY, ELUTE_VOL_UL_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        params_df = pd.DataFrame(params_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            params_dict[SAMPLE_ID_KEY])
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)

        min_coverage = 10
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        err_msg = r"sample info is missing required column\(s\): " \
                  r"\['sequenced_sample_gdna_mass_ng'\]"
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_biom(
                params_df, TestCalcCellCountsData.linregresses_dict,
                counts_biom, coverages_df, lengths_df,
                min_coverage, min_rsquared, output_metric)

    def _help_test_calc_ogu_cell_counts_biom_w_id_err(
            self, err_msg, params_dict=None, params_df=None,
            count_ogu_ids=None, count_sample_ids=None, coverages_df=None,
            lengths_df=None):
        if params_dict is None:
            params_dict = {
                k: TestCalcCellCountsData.sample_and_prep_input_dict[k] for
                k in [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
                      GDNA_CONCENTRATION_NG_UL_KEY, ELUTE_VOL_UL_KEY,
                      SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY]}

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()
        if count_ogu_ids is None:
            count_ogu_ids = TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY]
        if count_sample_ids is None:
            count_sample_ids = params_dict[SAMPLE_ID_KEY]

        counts_biom = biom.table.Table(
            counts_vals, count_ogu_ids, count_sample_ids)

        if params_df is None:
            params_df = pd.DataFrame(params_dict)
        if coverages_df is None:
            coverages_df = pd.DataFrame(
                TestCalcCellCountsData.ogu_percent_coverage_dict)
        if lengths_df is None:
            lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)

        min_coverage = 10
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_biom(
                params_df, TestCalcCellCountsData.linregresses_dict,
                counts_biom, coverages_df, lengths_df,
                min_coverage, min_rsquared, output_metric)

    def test_calc_ogu_cell_counts_biom_w_sample_info_missing_sample_id_err(self):
        # removes one of the sample ids from the params info; this will cause
        # an error (whereas the reverse--sample id in params info but not in
        # reads data--will NOT)
        params_dict = {
            k: TestCalcCellCountsData.sample_and_prep_input_dict[k] for
            k in [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
                  GDNA_CONCENTRATION_NG_UL_KEY, ELUTE_VOL_UL_KEY,
                  SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY]}

        # drop one sample from the params_df (don't care which one)
        params_df = pd.DataFrame(params_dict)
        params_df.drop(index=0, axis=0, inplace=True)

        err_msg = (r"Found sample ids in OGU counts data that were not in "
                   r"sample info: \{'example1'\}")
        self._help_test_calc_ogu_cell_counts_biom_w_id_err(
            err_msg, params_dict=params_dict,
            params_df=params_df)

    def test_calc_ogu_cell_counts_biom_w_coverages_missing_sample_id_err(self):
        # removes one of the sample ids from the coverages per sample df
        coverages_per_sample_df = \
            TestCalcCellCountsData.ogu_percent_coverage_per_sample_df.copy()
        coverages_per_sample_df.drop(
            coverages_per_sample_df.columns[-1], axis=1, inplace=True)

        err_msg = ("Found sample ids in OGU counts data that were not in"
                   " OGU coverage data: {'example2'}")
        self._help_test_calc_ogu_cell_counts_biom_w_id_err(
            err_msg, coverages_df=coverages_per_sample_df)

    def test_calc_ogu_cell_counts_biom_w_coverages_missing_ogu_id_err(self):
        # removes one of the ogu ids from the coverages per sample df
        coverages_per_sample_df = \
            TestCalcCellCountsData.ogu_percent_coverage_per_sample_df.copy()
        coverages_per_sample_df.drop(index=0, axis=0, inplace=True)

        err_msg = ("Found OGU ids in OGU counts data that were not in "
                   "OGU coverage data: {'Lactobacillus gasseri'}")
        self._help_test_calc_ogu_cell_counts_biom_w_id_err(
            err_msg, coverages_df=coverages_per_sample_df)

    def test_calc_ogu_cell_counts_biom_w_lengths_missing_ogu_id_err(self):
        # removes one of the ogu ids from the ogu lengths df
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)
        lengths_df.drop(index=0, axis=0, inplace=True)

        err_msg = ("Found OGU ids in OGU counts data that were not in "
                   "OGU lengths info: {'Lactobacillus gasseri'}")
        self._help_test_calc_ogu_cell_counts_biom_w_id_err(
            err_msg, lengths_df=lengths_df)

    def test_calc_ogu_cell_counts_biom_w_cast(self):
        # these values are the same as those in self.sample_and_prep_input_dict
        # except that some of them are represented as strings instead of #s
        params_dict = {
            SAMPLE_ID_KEY: ["example1", "example2"],
            GDNA_CONCENTRATION_NG_UL_KEY: ["2", 1.4],
            SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.027829017, "0.029491697"],
            ELUTE_VOL_UL_KEY: ["100", "70"],
            SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY: [5, "4.76"],
        }

        counts_vals = TestCalcCellCountsData.make_combined_counts_np_array()

        params_df = pd.DataFrame(params_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            params_dict[SAMPLE_ID_KEY])
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(TestCalcCellCountsData.reordered_results_dict[OGU_CELLS_PER_G_OF_GDNA_KEY]),
            TestCalcCellCountsData.reordered_results_dict[OGU_ID_KEY],
            TestCalcCellCountsData.reordered_results_dict[SAMPLE_ID_KEY])

        min_coverage = 10
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        # Note: 1) this is outputting the ogu_cell_counts_per_g_gdna, not the
        # ogu_cell_counts_per_g_sample (which is what is output by the qiita
        # version of this function) because I want to check that I really can
        # choose to get something else, and 2) this is using the full version
        # of Avogadro's #, not the truncated version that was used in the
        # notebook, so the results are slightly different (but more realistic)
        output_biom, output_msgs = calc_ogu_cell_counts_biom(
            params_df, TestCalcCellCountsData.linregresses_dict, counts_biom,
            coverages_df, lengths_df, min_coverage, min_rsquared,
            output_metric)

        # NB: only checking results to 1 decimal because Ubuntu and Mac
        # differ past that point. Not that it matters much since the decimal
        # portion of values this huge is not very important.
        a_tester = Testers()
        a_tester.assert_biom_tables_equal(expected_out_biom, output_biom,
                                          decimal_precision=1)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of "
             "10.0: ['Neisseria subflava', 'Haemophilus influenzae']"],
            output_msgs)

    def test__calc_long_format_ogu_cell_counts_df(self):
        counts_dict = {
            OGU_ID_KEY:
                TestCalcCellCountsData.example1_ogu_full_inputs_dict[
                    OGU_ID_KEY],
            "example1": SparseArray(
                TestCalcCellCountsData.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
            "example2": SparseArray(
                TestCalcCellCountsData.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        }

        ogu_masses = \
            TestCalcCellCountsData.example1_ogu_filtered_outputs_full_avogadro_dict[
                OGU_GDNA_MASS_NG_KEY]
        # NB: The example 2 gdna mass values come from the "PredictedOguMass"
        # column of the "Mass results for sample B regression for
        # full data on log10_read_count" table on the
        # "linear regressions counts" sheet of
        # "absolute_quant_example.xlsx", but with the 8th and 12th
        # (1-based) values removed since this is for *filtered*
        # example 2 data.
        ogu_masses.extend([0.04598325,
                           0.055539299,
                           0.050497812,
                           0.022733948,
                           0.014217626,
                           0.010754522,
                           0.000455761,
                           # NUM!
                           0.009360969,
                           0.142285222,
                           0.005462112,
                           # 1.10529E-05,
                           0.005523798])

        # NB: this test is NOT using the truncated version of Avogadro's # that
        # was used in the notebook, so the results are slightly different
        # (but more realistic)
        ogu_ids = TestCalcCellCountsData.combine_filtered_out(OGU_ID_KEY)
        expected_dict = {
            OGU_ID_KEY: ogu_ids,
            SAMPLE_ID_KEY: TestCalcCellCountsData.generate_sample_names_list(),
            OGU_READ_COUNT_KEY: SparseArray(
                TestCalcCellCountsData.combine_filtered_out(OGU_READ_COUNT_KEY)),
            OGU_AGNOSTIC_COVERAGE_KEY:
                TestCalcCellCountsData.combine_filtered_out(OGU_AGNOSTIC_COVERAGE_KEY),
            OGU_LEN_IN_BP_KEY:
                TestCalcCellCountsData.combine_filtered_out(
                    OGU_LEN_IN_BP_KEY),
            # TOTAL_OGU_READS_KEY: SparseArray(
            #     TestCalcCellCountsData.combine_filtered_out(TOTAL_OGU_READS_KEY)),
            OGU_GDNA_MASS_NG_KEY: SparseArray(ogu_masses),
            OGU_GENOMES_PER_G_OF_GDNA_KEY: SparseArray(
                TestCalcCellCountsData.combine_filtered_out(OGU_GENOMES_PER_G_OF_GDNA_KEY)),
            OGU_CELLS_PER_G_OF_GDNA_KEY: SparseArray(
                TestCalcCellCountsData.combine_filtered_out(OGU_CELLS_PER_G_OF_GDNA_KEY)),
            OGU_CELLS_PER_G_OF_SAMPLE_KEY: SparseArray(
                TestCalcCellCountsData.combine_filtered_out(OGU_CELLS_PER_G_OF_SAMPLE_KEY)),
            OGU_CELLS_PER_UL_OF_SAMPLE_KEY: SparseArray(
                TestCalcCellCountsData.combine_filtered_out(OGU_CELLS_PER_UL_OF_SAMPLE_KEY)),
            OGU_CELLS_PER_CM2_OF_SAMPLE_KEY: SparseArray(
                TestCalcCellCountsData.combine_filtered_out(OGU_CELLS_PER_CM2_OF_SAMPLE_KEY))
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)

        per_sample_calc_info_df = pd.DataFrame(
            TestCalcCellCountsData.mass_and_totals_dict)
        coverages_per_sample_df = \
            TestCalcCellCountsData.ogu_percent_coverage_per_sample_df.copy()
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)
        expected_df = pd.DataFrame(expected_dict)

        min_coverage = 10
        min_rsquared = 0.8

        output_df, output_msgs = _calc_long_format_ogu_cell_counts_df(
            TestCalcCellCountsData.linregresses_dict, counts_df,
            coverages_per_sample_df, lengths_df, per_sample_calc_info_df,
            min_coverage, min_rsquared)

        pd.testing.assert_frame_equal(expected_df, output_df)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of "
             "10.0: ['Neisseria subflava', 'Haemophilus influenzae']"],
            output_msgs)

    def test__calc_long_format_ogu_cell_counts_df_error(self):
        counts_dict = {
            OGU_ID_KEY:
                TestCalcCellCountsData.example1_ogu_full_inputs_dict[
                    OGU_ID_KEY],
            "example1": SparseArray(
                TestCalcCellCountsData.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
            "example2": SparseArray(
                TestCalcCellCountsData.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        }

        mass_ratio_dict = {k: TestCalcCellCountsData.mass_and_totals_dict[k] for k in
                           (SAMPLE_ID_KEY, GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY)}

        linregresses_dict = {
            'example1': None,
            'example2': None
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        mass_ratio_df = pd.DataFrame(mass_ratio_dict)
        coverages_df = \
            TestCalcCellCountsData.ogu_percent_coverage_per_sample_df.copy()
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)

        min_coverage = 10
        min_rsquared = 0.8

        err_msg = "No cell counts calculated for any sample"
        with self.assertRaisesRegex(ValueError, err_msg):
            _calc_long_format_ogu_cell_counts_df(
                linregresses_dict, counts_df, coverages_df, lengths_df,
                mass_ratio_df, min_coverage, min_rsquared)

    def test__prepare_cell_counts_calc_df_w_log_msgs_low_coverage(self):
        counts_dict = {
            OGU_ID_KEY:
                TestCalcCellCountsData.example1_ogu_full_inputs_dict[
                    OGU_ID_KEY],
            "example1": SparseArray(
                TestCalcCellCountsData.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
            "example2": SparseArray(
                TestCalcCellCountsData.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        }

        expected_out_dict = {
            OGU_ID_KEY: TestCalcCellCountsData.combine_filtered_out(OGU_ID_KEY),
            SAMPLE_ID_KEY: TestCalcCellCountsData.generate_sample_names_list(),
            OGU_READ_COUNT_KEY: SparseArray(
                TestCalcCellCountsData.combine_filtered_out(OGU_READ_COUNT_KEY)),
            OGU_AGNOSTIC_COVERAGE_KEY:
                TestCalcCellCountsData.combine_filtered_out(OGU_AGNOSTIC_COVERAGE_KEY),
            OGU_LEN_IN_BP_KEY:
                TestCalcCellCountsData.combine_filtered_out(
                    OGU_LEN_IN_BP_KEY)
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        coverages_df = \
            TestCalcCellCountsData.ogu_percent_coverage_per_sample_df.copy()
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)
        expected_out_df = pd.DataFrame(expected_out_dict)

        min_coverage = 10

        output_df, output_msgs = _prepare_cell_counts_calc_df(
            counts_df, coverages_df, lengths_df, min_coverage)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual(["The following items have coverage lower "
                              "than the minimum of 10.0: "
                              "['Neisseria subflava', "
                              "'Haemophilus influenzae']"
                              ], output_msgs)

    def test__prepare_cell_counts_calc_df_v_sparse(self):
        # the input and output values in this test are not based on the
        # worked examples; they are just made up to test that the code
        # correctly consumes and outputs sparse arrays

        min_coverage = 20

        # zero-based positions of removed OGUs
        removed = [4, 5, 7, 8, 9, 10, 11, 12]
        filtered_ogu_ids = _remove_filtered_entries(
            TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY], removed)
        filtered_ogu_lens = _remove_filtered_entries(
            TestCalcCellCountsData.ogu_lengths_dict[OGU_LEN_IN_BP_KEY],
            removed)
        filtered_ogu_coverage = _remove_filtered_entries(
            TestCalcCellCountsData.ogu_percent_coverage_dict[
                OGU_PERCENT_COVERAGE_KEY], removed)

        a_counts = [150, 0, 0, 0, 0, 0, 1975, 26130, 22303, 19783, 14478,
                    12145, 14609]
        b_counts = [0, 0, 0, 0, 0, 0, 1975, 0, 22303, 197830, 14478, 12, 14609]
        counts_dict = {
            OGU_ID_KEY: TestCalcCellCountsData.ogu_lengths_dict[OGU_ID_KEY],
            "A": SparseArray(a_counts),
            "B": SparseArray(b_counts),
        }

        expected_out_dict = {
            OGU_ID_KEY: filtered_ogu_ids + filtered_ogu_ids,
            SAMPLE_ID_KEY: ["A", "A", "A", "A", "A",
                            "B", "B", "B", "B", "B"],
            OGU_READ_COUNT_KEY: SparseArray(
                _remove_filtered_entries(a_counts, removed)
                + _remove_filtered_entries(b_counts, removed)),
            OGU_AGNOSTIC_COVERAGE_KEY:
                filtered_ogu_coverage + filtered_ogu_coverage,
            OGU_LEN_IN_BP_KEY: filtered_ogu_lens + filtered_ogu_lens
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        coverages_df = pd.DataFrame(
            TestCalcCellCountsData.ogu_percent_coverage_dict)
        coverages_df['A'] = coverages_df[OGU_PERCENT_COVERAGE_KEY]
        coverages_df['B'] = coverages_df[OGU_PERCENT_COVERAGE_KEY]
        coverages_df.drop(columns=[OGU_PERCENT_COVERAGE_KEY], inplace=True)
        lengths_df = pd.DataFrame(TestCalcCellCountsData.ogu_lengths_dict)
        expected_out_df = pd.DataFrame(expected_out_dict)

        output_df, output_msgs = _prepare_cell_counts_calc_df(
            counts_df, coverages_df, lengths_df, min_coverage)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum "
             "of 20.0: "
             "['Prevotella sp. oral taxon 299', 'Streptococcus mitis', "
             "'Neisseria subflava', 'Neisseria flavescens', "
             "'Fusobacterium periodonticum', 'Streptococcus pneumoniae', "
             "'Haemophilus influenzae', 'Veillonella dispar']"],
            output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample(self):
        input_dict = TestCalcCellCountsData.combine_inputs()
        input_df = pd.DataFrame(input_dict)
        per_sample_info_df = pd.DataFrame(TestCalcCellCountsData.mass_and_totals_dict)

        expected_additions_dict = {
            k: TestCalcCellCountsData
                .example1_ogu_full_outputs_full_avogadro_dict[k]
            for k in (OGU_ID_KEY, OGU_GDNA_MASS_NG_KEY,
                      OGU_GENOMES_PER_G_OF_GDNA_KEY,
                      OGU_CELLS_PER_G_OF_GDNA_KEY,
                      OGU_CELLS_PER_G_OF_SAMPLE_KEY,
                      OGU_CELLS_PER_UL_OF_SAMPLE_KEY,
                      OGU_CELLS_PER_CM2_OF_SAMPLE_KEY)}

        sample_a_df = input_df[input_df[SAMPLE_ID_KEY] == "example1"]
        expected_add_df = pd.DataFrame(expected_additions_dict)
        expected_out_df = sample_a_df.merge(expected_add_df, on=OGU_ID_KEY)

        sample_id = "example1"
        min_rsquared = 0.8

        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, TestCalcCellCountsData.linregresses_dict,
            per_sample_info_df, input_df, min_rsquared, is_test=False)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual([], output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample_w_nan_ratios(self):
        input_dict = TestCalcCellCountsData.combine_inputs()
        input_df = pd.DataFrame(input_dict)
        per_sample_info_df = pd.DataFrame(TestCalcCellCountsData.mass_and_totals_dict)
        # set two of the ratios to nans--meaning we didn't get the inputs
        # necessary to calculate them.  Outputs for these will exist but be nan
        per_sample_info_df[GDNA_MASS_TO_SAMPLE_VOL_RATIO_KEY] = np.nan
        per_sample_info_df[GDNA_MASS_TO_SAMPLE_SURFACE_AREA_RATIO_KEY] = np.nan

        expected_additions_dict = {
            k: TestCalcCellCountsData
                .example1_ogu_full_outputs_full_avogadro_dict[k]
            for k in (OGU_ID_KEY, OGU_GDNA_MASS_NG_KEY,
                      OGU_GENOMES_PER_G_OF_GDNA_KEY,
                      OGU_CELLS_PER_G_OF_GDNA_KEY,
                      OGU_CELLS_PER_G_OF_SAMPLE_KEY)}
        expected_additions_dict[OGU_CELLS_PER_UL_OF_SAMPLE_KEY] = np.nan
        expected_additions_dict[OGU_CELLS_PER_CM2_OF_SAMPLE_KEY] = np.nan

        sample_a_df = input_df[input_df[SAMPLE_ID_KEY] == "example1"]
        expected_add_df = pd.DataFrame(expected_additions_dict)
        expected_out_df = sample_a_df.merge(expected_add_df, on=OGU_ID_KEY)

        sample_id = "example1"
        min_rsquared = 0.8

        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, TestCalcCellCountsData.linregresses_dict,
            per_sample_info_df, input_df, min_rsquared, is_test=False)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual([], output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample_w_log_msgs_no_model(self):
        input_dict = TestCalcCellCountsData.combine_inputs()
        input_df = pd.DataFrame(input_dict)
        mass_ratio_df = pd.DataFrame(TestCalcCellCountsData.mass_and_totals_dict)

        # No entry for example 1, which should trigger a log message.
        linregresses_dict = copy.deepcopy(TestCalcCellCountsData.linregresses_dict)
        del linregresses_dict["example1"]

        sample_id = "example1"
        min_rsquared = 0.8

        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, linregresses_dict, mass_ratio_df, input_df,
            min_rsquared, is_test=True)

        self.assertIsNone(output_df)
        self.assertListEqual(
            ["No linear regression fitted for sample example1"], output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample_w_log_msgs_low_rsquared(self):
        input_dict = TestCalcCellCountsData.combine_inputs()
        input_df = pd.DataFrame(input_dict)
        mass_ratio_df = pd.DataFrame(TestCalcCellCountsData.mass_and_totals_dict)

        sample_id = "example1"
        high_min_rsquared = 0.99

        # here the minimum r_squared is set to 0.99 (which is probably
        # ridiculously high) so the linear model for sample A will be judged
        # not good enough to use
        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, TestCalcCellCountsData.linregresses_dict, mass_ratio_df,
            input_df, high_min_rsquared, is_test=True)

        self.assertIsNone(output_df)
        self.assertListEqual(['R^2 of linear regression for sample example1 '
                              'is 0.9731883614079859, which is less than the '
                              'minimum allowed value of 0.99.'],
                             output_msgs)

    def test__calc_ogu_genomes_per_g_of_gdna_series_for_sample(self):
        # this is the default value for our experimental system
        total_sample_gdna_mass_ng = 5

        # NOTE: the input mass values here are the originals from the Zaramela
        # R notebook, which had an issue in the calculation. HOWEVER,
        # that doesn't matter here because they are inputs, not outputs: IF
        # you input these values, you get these outputs (regardless of whether
        # these input values are meaningful).
        input_dict = {
            k: TestCalcCellCountsData
                .example1_ogu_full_outputs_short_avogadro_dict[k]
            for k in (OGU_ID_KEY, OGU_LEN_IN_BP_KEY,
                      OGU_GDNA_MASS_NG_KEY)}
        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(
            TestCalcCellCountsData.example1_ogu_full_outputs_short_avogadro_dict[
                OGU_CELLS_PER_G_OF_GDNA_KEY],
            index=TestCalcCellCountsData.example1_ogu_full_outputs_short_avogadro_dict[
                OGU_ID_KEY])
        expected_series.index.name = OGU_ID_KEY

        # NOTE: the is_test flag is set to True to use the *truncated* value of
        # Avogadro's number that is used in the notebook that is the source of
        # the expected results.  It should never be set to True in production.
        output_series = _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
            input_df, total_sample_gdna_mass_ng, is_test=True)

        assert_series_equal(expected_series, output_series)

    def test__calc_ogu_genomes_series_for_sample(self):
        # Expected results are taken from cell directly under the
        # header "Applying the linear models to sequencing data" of the
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1;
        # the "Species" column is OGU_ID_KEY, and the
        # CellNumber column is "ogu_cells_in_sample".
        expected_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            "ogu_cells_in_sample": [263469.8016, 138550.8742,
                                    109485.9657, 64330.93757,
                                    63369.31482, 57911.52782,
                                    56114.06446, 54395.84228,
                                    46448.32735, 35499.48595,
                                    29049.10845, 28593.0947,
                                    28574.60346]
        }

        # NOTE: the input mass values here are the originals from the Zaramela
        # R notebook, which had an issue in the calculation. HOWEVER,
        # that doesn't matter here because they are inputs, not outputs: IF
        # you input these values, you get these outputs (regardless of whether
        # these input values are meaningful).
        input_dict = {
            k: TestCalcCellCountsData
                .example1_ogu_full_outputs_short_avogadro_dict[k]
            for k in (OGU_ID_KEY, OGU_LEN_IN_BP_KEY,
                      OGU_GDNA_MASS_NG_KEY)}
        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(expected_dict["ogu_cells_in_sample"],
                                    index=expected_dict[OGU_ID_KEY])
        expected_series.index.name = OGU_ID_KEY

        # NOTE: the is_test flag is set to True to use the *truncated* value of
        # Avogadro's number that is used in the notebook that is the source of
        # the expected results.  It should never be set to True in production.
        output_series = _calc_ogu_genomes_series_for_sample(
            input_df, is_test=True)

        assert_series_equal(expected_series, output_series)

    def test__calc_ogu_gdna_mass_ng_series_for_sample(self):
        input_dict = \
            {k: TestCalcCellCountsData.example1_ogu_full_inputs_dict[k]
             for k in (OGU_ID_KEY, OGU_READ_COUNT_KEY)}

        # The slope and intercept values are taken from
        # those calculated in Excel (see
        # "sample A regression of log10_syndna_ng from log10_raw_counts"
        # for full data on "zaramela linear reg CPM&counts" sheet of
        # "absolute_quant_example.xlsx").
        # Note that the intercept does not and *should* NOT be expected to
        # match the intercept in Zaramela's linear models because we are
        # fitting log10(raw counts) not log10(CPM), but this comes
        # out in the wash when one actually *uses* the models to predict mass
        # (HOWEVER, this is not true for the masses taken directly from the
        # Zaramela R notebook because it unintentionally used a different total
        # read value when calculating the OGU CPM than when calculating
        # the syndna CPM to get the fit.)
        slope = 1.24487652379132
        intercept = -7.40709604550579

        # This list contains the gdna mass from the
        # "Local mass using RawCounts (no TotalReads)" column in the
        # "linear regressions counts" sheet of "absolute_quant_example.xlsx"
        expected_ogu_masses = \
            [0.04969441309413190000,
             0.06000552485579740000,
             0.05456646428684080000,
             0.02459526358752070000,
             0.01539258341742420000,
             0.01164819449827620000,
             0.00049599588089929300,
             0.01235085741392710000,
             0.01014096594158730000,
             0.00873488502026095000,
             0.00592205420878227000,
             0.00475854893636541000,
             0.00598883339989535000]

        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(
            expected_ogu_masses,
            index=TestCalcCellCountsData.example1_ogu_full_inputs_dict[OGU_ID_KEY],
            name=OGU_GDNA_MASS_NG_KEY)
        expected_series.index.name = OGU_ID_KEY

        output_series = _calc_ogu_gdna_mass_ng_series_for_sample(
            input_df, slope, intercept)

        assert_series_equal(expected_series, output_series)
