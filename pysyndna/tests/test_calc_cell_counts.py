import biom.table
import copy
import numpy as np
import numpy.testing as npt
import pandas as pd
from pandas.arrays import SparseArray
from pandas.testing import assert_series_equal
import os
from unittest import TestCase
from pysyndna import calc_ogu_cell_counts_biom, \
    calc_ogu_cell_counts_per_g_of_sample_for_qiita
from pysyndna.src.fit_syndna_models import SAMPLE_ID_KEY
from pysyndna.src.calc_cell_counts import OGU_ID_KEY, OGU_READ_COUNT_KEY, \
    OGU_LEN_IN_BP_KEY, OGU_GDNA_MASS_NG_KEY, \
    SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY, OGU_GENOMES_PER_G_OF_GDNA_KEY, \
    OGU_CELLS_PER_G_OF_GDNA_KEY, SYNDNA_POOL_MASS_NG_KEY, \
    GDNA_CONCENTRATION_NG_UL_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY, \
    ELUTE_VOL_UL_KEY, GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY, \
    OGU_CELLS_PER_G_OF_SAMPLE_KEY, TOTAL_OGU_READS_KEY, OGU_COVERAGE_KEY, \
    CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY, \
    _calc_long_format_ogu_cell_counts_df, \
    _prepare_cell_counts_calc_df, \
    _calc_ogu_cell_counts_df_for_sample, \
    _calc_gdna_mass_to_sample_mass_by_sample_df, \
    _calc_ogu_gdna_mass_ng_series_for_sample, \
    _calc_ogu_genomes_per_g_of_gdna_series_for_sample, \
    _calc_ogu_genomes_series_for_sample


class TestCalcCellCounts(TestCase):
    # Throughout below, example1 is the "Sample A" data from
    # test_fit_syndna_models.py, paired with "standard" elute volume and
    # gdna density values. example2 is the "Sample B" data from
    # test_fit_syndna_models.py, paired with a smaller elute volume and
    # and a lower gdna density value (which leadds to a smaller amount of
    # syndna pool and thus a bunch of interesting calculation knock-ons).

    # The example1 linear model is what we get from running Zaramela's
    # "A1_pool1_Fwd" sample data through the linear modelling code (see
    # modelling_input.tsv and test_fit_syndna_models.py.) The slope and
    # intercept (all that is provided by Zaramela) match those from that
    # work (see modelling_output.tsv for details).
    # The example2 linear model is what we get from running the linear
    # modelling code on the "Sample B" data from test_fit_syndna_models.py;
    # it does not match any Zaramela results because Sample B is made up.
    # See FitSyndnaModelsTest.lingress_results comments for details.
    linregresses_dict = {
        'example1': {
            "slope": 1.24487652379132, "intercept": -6.77539505390338,
            "rvalue": 0.9865030975156575, "pvalue": 1.428443560659758e-07,
            "stderr": 0.07305408550335003,
            "intercept_stderr": 0.2361976278251443},
        'example2': {
            "slope": 1.24675913604407, "intercept": -7.155318973708384,
            "rvalue": 0.9863241797356326, "pvalue": 1.505381146809759e-07,
            "stderr": 0.07365795255302438,
            "intercept_stderr": 0.2563956755844754}
    }

    # Values from "absolute_quant_example.xlsx"
    sample_and_prep_input_dict = {
        SAMPLE_ID_KEY: ["example1", "example2"],
        SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.027829017, 0.029491697],
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY: [5, 4.76],
        GDNA_CONCENTRATION_NG_UL_KEY: [2, 1.4],
        ELUTE_VOL_UL_KEY: [100, 100],
        SYNDNA_POOL_MASS_NG_KEY: [0.25, 0.238]
    }

    # Values from "absolute_quant_example.xlsx"
    mass_ratio_dict = {
        SAMPLE_ID_KEY: ["example1", "example2"],
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY: [5, 4.76],
        GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY: [7.1867431342E-06,
                                             4.7470988923E-06]
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
        # at this position), and by 12133 at position 12 (for H. influenzae,
        # bc example 2 has 12 counts here instead of 12145).
        # All other positions are identical.
        OGU_READ_COUNT_KEY: [79951, 93024, 86188, 45441, 31185, 24929,
                             1975, 0, 22303, 197830, 14478, 12, 14609]
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
        # detail for example1 in the "absolute_quant_example.xlsx" spreadsheet
        # (in the section using the truncated Avogadro's #, as Zaramela did).
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

    # This dict contains the results of calculations done on example1
    # using the full Avogadro's number (6.022e23) instead of the truncated one.
    example1_ogu_full_outputs_full_avogadro_dict = (
        example1_ogu_full_inputs_dict.copy())
    example1_ogu_full_outputs_full_avogadro_dict.update({
        # Note that this is assuming that examples 1 and 2 were run
        TOTAL_OGU_READS_KEY: [x + y for x, y in zip(
            example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY],
            example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY])],
        # These values are taken from cell directly under the
        # header "Applying the linear models to sequencing data" of the
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1.
        # The Cov column is OGU_COVERAGE_KEY. (Note that the coverage calc
        # is upstream of the part of the calculation using Avogadro's number
        # so it is not affected by the truncated vs. full Avogadro's number.)
        OGU_COVERAGE_KEY: [6.29597514334489, 3.19032039014754,
                           2.56862497316419, 1.76537729965377,
                           1.90692890582578, 1.84090986293668,
                           3.31880713389496, 1.70934318831428,
                           1.51731341482939, 1.19420333792332,
                           1.05484891342717, 1.08394039151921,
                           1.03533221485547],
        # for the arrays below, the 1st, 2nd, and 7th values (those for
        # L. gasseri, R. albus, and L. valderiana) match those worked out in
        # detail for example1 in the "absolute_quant_example.xlsx" spreadsheet
        # (in the section using the FULL Avogadro's #, not matching Zaramela).
        # The remainder were not worked out individually and come from the code
        # calculations.
        OGU_GENOMES_PER_G_OF_GDNA_KEY: [
            52695192015949.67, 27710822536547.69, 21897704979729.094,
            12866488251594.062, 12674159207435.06, 11582576292095.531,
            11223075218306.252, 10879422748260.775, 9289882608698.639,
            7100063146106.998, 5809957491032.718, 5718752608946.0205,
            5715054273735.247],
        OGU_CELLS_PER_G_OF_GDNA_KEY: [
            52695192015949.67, 27710822536547.69, 21897704979729.094,
            12866488251594.062, 12674159207435.06, 11582576292095.531,
            11223075218306.252, 10879422748260.775, 9289882608698.639,
            7100063146106.998, 5809957491032.718, 5718752608946.0205,
            5715054273735.247],
        OGU_CELLS_PER_G_OF_SAMPLE_KEY: [
            378706809.42597693, 199150563.60756874, 157373180.91780522,
            92468146.10340859, 91085926.66579163, 83241000.64356525,
            80657358.76977262, 78187616.74012242, 66764000.05558892,
            51026330.06767092, 41754672.108673245, 41099206.04853115,
            41072627.06334715]
    })

    # This dict contains the results of calculations done on *filtered*
    # example2 data using the full Avogadro's number (6.022e23) instead of
    # the truncated one.
    # The ogu id, ogu length, and ogu read counts are exactly the same as in
    # the example 2 full data *except* that Neisseria subflava and
    # Haemophilus influenzae have been removed (for falling below the
    # min_coverage = 1 threshold, which is the default min_coverage value in
    # our analysis system).
    example2_ogu_filtered_inputs_outputs_full_avogadro_dict = {
        OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                     "Escherichia coli", "Tyzzerella nexilis",
                     "Prevotella sp. oral taxon 299",
                     "Streptococcus mitis", "Leptolyngbya valderiana",
                      #"Neisseria subflava",
                     "Neisseria flavescens",
                     "Fusobacterium periodonticum",
                     "Streptococcus pneumoniae",
                     # "Haemophilus influenzae",
                     "Veillonella dispar"],
        OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                            2453028, 2031251, 89264,
                            # 2292986,
                            2204851, 2484878.333, 2058778.25,
                            # 1680673.6,
                            2116567],
        OGU_READ_COUNT_KEY: [79951, 93024, 86188, 45441, 31185, 24929,
                             1975,
                             # 0,
                             22303, 197830, 14478,
                             #12,
                             14609],
        # These count values are the same as those in
        # self.example1_ogu_full_outputs_full_avogadro_dict
        # except that the counts for Neisseria subflava and
        # Haemophilus influenzae have been removed
        TOTAL_OGU_READS_KEY: [159901, 186048, 172376, 90882,
                              62370, 49858, 3950,
                              44606, 217613, 28956,
                              29218],
        # These values are the same as those in
        # self.example1_ogu_full_outputs_full_avogadro_dict EXCEPT for the
        # for L. gasseri (first in this list) and F. periodonticum (9th in this
        # list) values, which are different because the example2 counts for
        # these positions are different from the example1 counts (and still
        # large enough to pass the min_coverage = 1 threshold).  These two
        # values have been checked by hand.
        OGU_COVERAGE_KEY: [6.296053893, 3.19032039014754,
                           2.56862497316419, 1.76537729965377,
                           1.90692890582578, 1.84090986293668,
                           3.31880713389496,
                           1.51731341482939, 11.94203338,
                           1.05484891342717,
                           1.03533221485547],
        OGU_GENOMES_PER_G_OF_GDNA_KEY: [
            17086455403978.045, 8987677125515.266, 7101240813289.261,
            4167468287030.2075, 4102264162505.8833, 3747369928484.789,
            3613767901730.258, 3004973286163.8184, 40527863244164.32,
            1877803149989.8623, 1847161346896.6194],
        OGU_CELLS_PER_G_OF_GDNA_KEY: [
            17086455403978.045, 8987677125515.266, 7101240813289.261,
            4167468287030.2075, 4102264162505.8833, 3747369928484.789,
            3613767901730.258, 3004973286163.8184, 40527863244164.32,
            1877803149989.8623, 1847161346896.6194],
        OGU_CELLS_PER_G_OF_SAMPLE_KEY: [
            81111093.52155752, 42665392.12688357, 33710292.398721,
            19783384.089056477, 19473853.661753666, 17789135.636548474,
            17154913.603333004, 14264905.358139353, 192389774.71365833,
            8914117.253274327, 8768657.583752317]
    }

    # This dict contains the results of calculations done on example1 and
    # filtered example2 data, in the order that they appear in biom outputs,
    # which are alphabetized by ogu
    reordered_results_dict = {
        SAMPLE_ID_KEY: ["example1", "example2"],
        # in a biom table, results are ordered alphabetically
        OGU_ID_KEY: [
            'Escherichia coli', 'Fusobacterium periodonticum',
            'Haemophilus influenzae', 'Lactobacillus gasseri',
            'Leptolyngbya valderiana', 'Neisseria flavescens',
            'Neisseria subflava', 'Prevotella sp. oral taxon 299',
            'Ruminococcus albus', 'Streptococcus mitis',
            'Streptococcus pneumoniae', 'Tyzzerella nexilis',
            'Veillonella dispar'],
        # The values in the first position of each sub-array below is that from
        # that ogu in self.example1_ogu_full_outputs_full_avogadro_dict,
        # while the value in the second position of each sub-array below is
        # that from that ogu in
        # self.example2_ogu_filtered_inputs_outputs_full_avogadro_dict.  Note
        # that with reordering, the 4th sub-array is the one for L. gasseri,
        # the 5th is for L. valderiana, and the 9th is for R. albus.
        OGU_CELLS_PER_G_OF_GDNA_KEY: [
            [21897704979729.094, 7101240813289.261],
            [7100063146106.998, 40527863244164.32],
            [5718752608946.0205, np.nan],
            [52695192015949.67, 17086455403978.045],
            [11223075218306.252, 3613767901730.258],
            [9289882608698.639, 3004973286163.8184],
            [10879422748260.775, np.nan],
            [12674159207435.06, 4102264162505.8833],
            [27710822536547.69, 8987677125515.266],
            [11582576292095.531, 3747369928484.789],
            [5809957491032.718, 1877803149989.8623],
            [12866488251594.062, 4167468287030.2075],
            [5715054273735.247, 1847161346896.6194]],
    }

    def _combine_inputs(self):
        sample_names = self._generate_sample_names_list(use_filtered_ex2=False)
        col_list = [OGU_ID_KEY, OGU_READ_COUNT_KEY, OGU_LEN_IN_BP_KEY]
        parallel_dicts_list = [self.example1_ogu_full_inputs_dict,
                               self.example2_ogu_full_inputs_dict]
        input_dict = {SAMPLE_ID_KEY: sample_names}
        for col_name in col_list:
            curr_col_list = []
            for curr_dict in parallel_dicts_list:
                curr_col_list.extend(curr_dict[col_name].copy())
            input_dict[col_name] = curr_col_list

        return input_dict

    def _combine_filtered_out(self, col_name):
        example1_copy = (
            self.example1_ogu_full_outputs_full_avogadro_dict[col_name].copy())
        example2_copy = (
            self.example2_ogu_filtered_inputs_outputs_full_avogadro_dict[
                col_name].copy())
        example1_copy.extend(example2_copy)
        return example1_copy

    def _make_combined_counts_np_array(self):
        # combine each item in the OGU_READ_COUNT_KEY array for
        # self.example1_ogu_full_inputs_dict with the analogous item in
        # self.example2_ogu_full_inputs_dict to make an array of two-item
        # arrays, and turn this into an np.array
        counts_array = np.array(
            [list(x) for x in zip(
                self.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY],
                self.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY])])
        return counts_array

    def _generate_sample_names_list(self, use_filtered_ex2=True):
        sample_names = ["example1", "example2"]
        parallel_dicts_list = [self.example1_ogu_full_inputs_dict]
        if use_filtered_ex2:
            parallel_dicts_list.append(
                self.example2_ogu_filtered_inputs_outputs_full_avogadro_dict)
        else:
            parallel_dicts_list.append(
                self.example2_ogu_full_inputs_dict)

        output = []
        for curr_index in range(len(sample_names)):
            curr_sample_name = sample_names[curr_index]
            curr_parallel_dict = parallel_dicts_list[curr_index]
            curr_names_list = [curr_sample_name for _ in
                               range(len(curr_parallel_dict[OGU_ID_KEY]))]
            output.extend(curr_names_list)
        return output

    def setUp(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # The built-in self.assertEqual works fine to compare biom tables that
    # don't have NaNs, but it doesn't work for tables that do have NaNs
    # because NaN != NaN so two tables that contain NaNs are by definition
    # unequal even if the NaNs occur at the same table locations.
    # This function is a workaround for that.
    def assert_biom_tables_equal(self, expected_out_biom, output_biom,
                                 decimal_precision=7):
        # default decimal precision is the set to the default for
        # npt.assert_almost_equal

        # check the ids are equal, then check the observations are equal
        self.assertEqual(set(expected_out_biom.ids()), set(output_biom.ids()))
        self.assertEqual(set(expected_out_biom.ids(axis='observation')),
                         set(output_biom.ids(axis='observation')))

        # check that the two tables have the same NaN positions
        npt.assert_equal(np.isnan(expected_out_biom.matrix_data.data),
                         np.isnan(output_biom.matrix_data.data))

        # check that the two tables have the same non-NaN values at the same
        # positions
        obs_an = ~(np.isnan(output_biom.matrix_data.data))
        exp_an = ~(np.isnan(expected_out_biom.matrix_data.data))
        npt.assert_equal(exp_an, obs_an)
        npt.assert_almost_equal(expected_out_biom.matrix_data.data[exp_an],
                                output_biom.matrix_data.data[obs_an],
                                decimal=decimal_precision)

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita(self):
        # example4 is the same as example2 except that the elute volume is 70;
        # see "absolute_quant_example.xlsx" for details.
        example4_elute_vol = 70
        sample_ids = ["example1", "example4"]
        sample_info_dict = {k: self.sample_and_prep_input_dict[k].copy() for k in
                            [SAMPLE_IN_ALIQUOT_MASS_G_KEY]}
        sample_info_dict[SAMPLE_ID_KEY] = sample_ids

        prep_info_dict = {k: self.sample_and_prep_input_dict[k].copy() for k in
                          [GDNA_CONCENTRATION_NG_UL_KEY,
                            ELUTE_VOL_UL_KEY, SYNDNA_POOL_MASS_NG_KEY]}
        prep_info_dict[SAMPLE_ID_KEY] = sample_ids
        prep_info_dict[ELUTE_VOL_UL_KEY][1] = example4_elute_vol

        # example4 has the same counts as example2
        counts_vals = self._make_combined_counts_np_array()

        # NB: The test values for example1 here are *slightly* different than
        # those in self.example1_ogu_full_outputs_full_avogadro_dict because
        # the gdna-to-sample mass ratio calculated internally during this
        # soup-to-nuts function has more digits past the decimal than does the
        # example1 entry in the manually-populated self.mass_ratio_dict.
        # Since we are multiplying/dividing by large numbers like e.g., 10^9
        # (to change ng to g), this ends up making a slight difference in the
        # end product: for example, for L.gasseri,
        # 3787068*15* cells instead of 3787068*09* cells,
        # 80657360 instead of 80657358 for L. valderiana,
        # and 199150566 instead of 199150563 for R. albus.
        # The values for example 4 are about an order of magnitude smaller
        # than those for example 1 and thus match those from
        # "absolute_quant_example.xlsx" more closely:
        # both 56777765 for L. gasseri (with rounding),
        # 12008439 instead of 12008440 for L. valderiana,
        # and both 29865774 for R. albus.
        # Remember, with reordering, the 4th sub-array is for L. gasseri,
        # the 5th is for L. valderiana, and the 9th is for R. albus.
        ogu_cell_counts_per_g_sample = np.array([
            [157373183.3914873, 23597204.3149076],
            [51026330.8697321, 134672840.2210325],
            [41099206.6945521, np.nan],
            [378706815.3787082, 56777764.5887874],
            [80657360.0375914, 12008439.3369959],
            [66764001.1050239, 9985433.5965833],
            [78187617.9691203, np.nan],
            [91085928.0975326, 13631697.3528372],
            [199150566.7379318, 29865774.0278729],
            [83241001.9519951, 12452394.7533948],
            [41754672.7649972, 6239881.9809863],
            [92468147.5568761, 13848368.6486051],
            [41072627.7089503, 6138060.2138924]])

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            self.ogu_lengths_dict[OGU_ID_KEY],
            sample_ids)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            ogu_cell_counts_per_g_sample,
            self.reordered_results_dict[OGU_ID_KEY],
            sample_ids)

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita(
            sample_info_df, prep_info_df, models_fp, counts_biom,
            lengths_fp, read_len, min_coverage, min_rsquared)

        self.assertSetEqual(
            set(output_dict.keys()),
            {CELL_COUNT_RESULT_KEY, CELL_COUNT_LOG_KEY})
        self.assert_biom_tables_equal(
            expected_out_biom, output_dict[CELL_COUNT_RESULT_KEY])
        self.assertEqual(
            "The following items have coverage lower than the minimum of 1: "
            "['example4;Neisseria subflava', "
            "'example4;Haemophilus influenzae']",
            output_dict[CELL_COUNT_LOG_KEY])

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_w_sample_err(self):
        # missing a required column column
        sample_info_dict = {k: self.sample_and_prep_input_dict for k in
                            [SAMPLE_ID_KEY]}

        prep_info_dict = {k: self.sample_and_prep_input_dict[k] for k in
                          [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
                           ELUTE_VOL_UL_KEY, SYNDNA_POOL_MASS_NG_KEY]}

        counts_vals = self._make_combined_counts_np_array()
        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            self.ogu_lengths_dict[OGU_ID_KEY],
            prep_info_dict[SAMPLE_ID_KEY])
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8

        err_msg = r"sample info is missing required column\(s\): " \
                  r"\['calc_mass_sample_aliquot_input_g'\]"
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_per_g_of_sample_for_qiita(
                sample_info_df, prep_info_df, models_fp, counts_biom,
                lengths_fp, read_len, min_coverage, min_rsquared)

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita_w_prep_err(self):
        sample_info_dict = {k: self.sample_and_prep_input_dict[k] for k in
                            [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]}

        # missing required columns
        prep_info_dict = {k: self.sample_and_prep_input_dict[k] for k in
                          [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY]}

        counts_vals = self._make_combined_counts_np_array()

        sample_info_df = pd.DataFrame(sample_info_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            self.ogu_lengths_dict[OGU_ID_KEY],
            prep_info_dict[SAMPLE_ID_KEY])
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8

        err_msg = r"prep info is missing required column\(s\): " \
                  r"\[\'mass_syndna_input_ng'\, 'vol_extracted_elution_ul'\]"
        with self.assertRaisesRegex(ValueError, err_msg):
            calc_ogu_cell_counts_per_g_of_sample_for_qiita(
                sample_info_df, prep_info_df, models_fp, counts_biom,
                lengths_fp, read_len, min_coverage, min_rsquared)

    def test_calc_ogu_cell_counts_biom(self):
        params_dict = {k: self.sample_and_prep_input_dict[k] for k in
                       [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
                        GDNA_CONCENTRATION_NG_UL_KEY, ELUTE_VOL_UL_KEY,
                        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY]}

        counts_vals = self._make_combined_counts_np_array()

        params_df = pd.DataFrame(params_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            self.ogu_lengths_dict[OGU_ID_KEY],
            params_dict[SAMPLE_ID_KEY])
        lengths_df = pd.DataFrame(self.ogu_lengths_dict)
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(self.reordered_results_dict[OGU_CELLS_PER_G_OF_GDNA_KEY]),
            self.reordered_results_dict[OGU_ID_KEY],
            self.reordered_results_dict[SAMPLE_ID_KEY])

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        # Note: 1) this is outputting the ogu_cell_counts_per_g_gdna, not the
        # ogu_cell_counts_per_g_sample (which is what is output by the qiita
        # version of this function) because I want to check that I really can
        # choose to get something else, and 2) this is using the full version
        # of Avogadro's #, not the truncated version that was used in the
        # notebook, so the results are slightly different (but more realistic)
        output_biom, output_msgs = calc_ogu_cell_counts_biom(
            params_df, self.linregresses_dict, counts_biom, lengths_df,
            read_len, min_coverage, min_rsquared, output_metric)

        # NB: only checking results to 2 decimals because Ubuntu and Mac
        # differ past that point. Not that it matters much since the decimal
        # portion of values this huge is not very important.
        self.assert_biom_tables_equal(expected_out_biom, output_biom,
                                      decimal_precision=2)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of 1: "
             "['example2;Neisseria subflava', "
             "'example2;Haemophilus influenzae']"],
            output_msgs)

    def test_calc_ogu_cell_counts_biom_w_cast(self):
        # these values are the same as those in self.sample_and_prep_input_dict
        # except that some of them are represented as strings instead of #s
        params_dict = {
            SAMPLE_ID_KEY: ["example1", "example2"],
            GDNA_CONCENTRATION_NG_UL_KEY: ["2", 1.4],
            SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.027829017, "0.029491697"],
            ELUTE_VOL_UL_KEY: ["100", "70"],
            SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY: [5, "4.76"]
        }

        counts_vals = self._make_combined_counts_np_array()

        params_df = pd.DataFrame(params_dict)
        counts_biom = biom.table.Table(
            counts_vals,
            self.ogu_lengths_dict[OGU_ID_KEY],
            params_dict[SAMPLE_ID_KEY])
        lengths_df = pd.DataFrame(self.ogu_lengths_dict)
        # Note that, in the output, the ogu_ids are apparently sorted
        # alphabetically--different than the input order
        expected_out_biom = biom.table.Table(
            np.array(self.reordered_results_dict[OGU_CELLS_PER_G_OF_GDNA_KEY]),
            self.reordered_results_dict[OGU_ID_KEY],
            self.reordered_results_dict[SAMPLE_ID_KEY])

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        # Note: 1) this is outputting the ogu_cell_counts_per_g_gdna, not the
        # ogu_cell_counts_per_g_sample (which is what is output by the qiita
        # version of this function) because I want to check that I really can
        # choose to get something else, and 2) this is using the full version
        # of Avogadro's #, not the truncated version that was used in the
        # notebook, so the results are slightly different (but more realistic)
        output_biom, output_msgs = calc_ogu_cell_counts_biom(
            params_df, self.linregresses_dict, counts_biom, lengths_df,
            read_len, min_coverage, min_rsquared, output_metric)

        self.assert_biom_tables_equal(expected_out_biom, output_biom)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of 1: "
             "['example2;Neisseria subflava', "
             "'example2;Haemophilus influenzae']"],
            output_msgs)

    def test__calc_long_format_ogu_cell_counts_df(self):
        counts_dict = {
            OGU_ID_KEY: self.example1_ogu_full_inputs_dict[OGU_ID_KEY],
            "example1": SparseArray(
                self.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
            "example2": SparseArray(
                self.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        }

        ogu_masses = \
            self.example1_ogu_full_outputs_short_avogadro_dict[
                OGU_GDNA_MASS_NG_KEY].copy()
        # NB: The example2 test values were NOT verified manually.  They
        # are what the code that produces correct results for all the
        # values downstream of these values produces for these values.
        ogu_masses.extend([
            0.16721225613234225, 0.20196161687112832, 0.1836288899469535,
            0.08266911949481899, 0.051700594315481685, 0.03910745610301345,
            0.0016573186101851826, 0.03403997782058165, 0.517402166874799,
            0.01986227733996378, 0.02008659206780049])

        # NB: this test is NOT using the truncated version of Avogadro's # that
        # was used in the notebook, so the results are slightly different
        # (but more realistic)
        expected_dict = {
            OGU_ID_KEY: self._combine_filtered_out(OGU_ID_KEY),
            SAMPLE_ID_KEY: self._generate_sample_names_list(),
            OGU_READ_COUNT_KEY: SparseArray(
                self._combine_filtered_out(OGU_READ_COUNT_KEY)),
            OGU_LEN_IN_BP_KEY: self._combine_filtered_out(OGU_LEN_IN_BP_KEY),
            TOTAL_OGU_READS_KEY: SparseArray(
                self._combine_filtered_out(TOTAL_OGU_READS_KEY)),
            OGU_COVERAGE_KEY: SparseArray(
                self._combine_filtered_out(OGU_COVERAGE_KEY)),
            OGU_GDNA_MASS_NG_KEY: SparseArray(ogu_masses),
            OGU_GENOMES_PER_G_OF_GDNA_KEY: SparseArray(
                self._combine_filtered_out(OGU_GENOMES_PER_G_OF_GDNA_KEY)),
            OGU_CELLS_PER_G_OF_GDNA_KEY: SparseArray(
                self._combine_filtered_out(OGU_CELLS_PER_G_OF_GDNA_KEY)),
            OGU_CELLS_PER_G_OF_SAMPLE_KEY: SparseArray(
                self._combine_filtered_out(OGU_CELLS_PER_G_OF_SAMPLE_KEY))
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        mass_ratio_df = pd.DataFrame(self.mass_ratio_dict)
        lengths_df = pd.DataFrame(self.ogu_lengths_dict)
        expected_df = pd.DataFrame(expected_dict)

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8

        output_df, output_msgs = _calc_long_format_ogu_cell_counts_df(
            self.linregresses_dict, counts_df, lengths_df, mass_ratio_df,
            read_len, min_coverage, min_rsquared)

        pd.testing.assert_frame_equal(expected_df, output_df)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of 1: "
             "['example2;Neisseria subflava',"
             " 'example2;Haemophilus influenzae']"],
            output_msgs)

    def test__calc_long_format_ogu_cell_counts_df_error(self):
        counts_dict = {
            OGU_ID_KEY: self.example1_ogu_full_inputs_dict[OGU_ID_KEY],
            "example1": SparseArray(
                self.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
            "example2": SparseArray(
                self.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        }

        mass_ratio_dict = {k: self.mass_ratio_dict[k] for k in
                           (SAMPLE_ID_KEY, GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY)}

        linregresses_dict = {
            'example1': None,
            'example2': None
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        mass_ratio_df = pd.DataFrame(mass_ratio_dict)
        lengths_df = pd.DataFrame(self.ogu_lengths_dict)

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8

        err_msg = "No cell counts calculated for any sample"
        with self.assertRaisesRegex(ValueError, err_msg):
            _calc_long_format_ogu_cell_counts_df(
                linregresses_dict, counts_df, lengths_df, mass_ratio_df,
                read_len, min_coverage, min_rsquared)

    def test__prepare_cell_counts_calc_df_w_log_msgs_low_coverage(self):
        counts_dict = {
            OGU_ID_KEY: self.example1_ogu_full_inputs_dict[OGU_ID_KEY],
            "example1": SparseArray(
                self.example1_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
            "example2": SparseArray(
                self.example2_ogu_full_inputs_dict[OGU_READ_COUNT_KEY]),
        }

        expected_out_dict = {
            OGU_ID_KEY: self._combine_filtered_out(OGU_ID_KEY),
            SAMPLE_ID_KEY: self._generate_sample_names_list(),
            OGU_READ_COUNT_KEY: SparseArray(
                self._combine_filtered_out(OGU_READ_COUNT_KEY)),
            OGU_LEN_IN_BP_KEY: self._combine_filtered_out(OGU_LEN_IN_BP_KEY),
            TOTAL_OGU_READS_KEY: SparseArray(
                self._combine_filtered_out(TOTAL_OGU_READS_KEY)),
            OGU_COVERAGE_KEY: SparseArray(
                self._combine_filtered_out(OGU_COVERAGE_KEY))
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        lengths_df = pd.DataFrame(self.ogu_lengths_dict)
        expected_out_df = pd.DataFrame(expected_out_dict)

        read_len = 150
        min_coverage = 1

        output_df, output_msgs = _prepare_cell_counts_calc_df(
            counts_df, lengths_df, read_len, min_coverage)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual(["The following items have coverage lower "
                              "than the minimum of 1: "
                              "['example2;Neisseria subflava',"
                              " 'example2;Haemophilus influenzae']"],
                             output_msgs)

    def test__prepare_cell_counts_calc_df_v_sparse(self):
        # the input and output values in this test are not based on the
        # worked examples; they are just made up to test that the code
        # correctly consumes and outputs sparse arrays
        counts_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            "A": SparseArray([0, 0, 0, 0, 0, 0,
                              1975, 26130, 22303, 19783, 14478, 12145, 14609]),
            "B": SparseArray([0, 0, 0, 0, 0, 0,
                              1975, 0, 22303, 197830, 14478, 12, 14609]),
        }

        expected_out_dict = {
            OGU_ID_KEY: ["Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar",
                         "Leptolyngbya valderiana",
                         "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae",
                         "Veillonella dispar"],
            SAMPLE_ID_KEY: ["A", "A", "A", "A", "A", "A", "A",
                            "B", "B", "B", "B", "B"],
            OGU_READ_COUNT_KEY: SparseArray([1975, 26130, 22303,
                                             19783, 14478, 12145, 14609,
                                             1975, 22303, 197830, 14478,
                                             14609]),
            OGU_LEN_IN_BP_KEY: [89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567,
                                89264, 2204851,
                                2484878.333, 2058778.25, 2116567],
            TOTAL_OGU_READS_KEY: SparseArray([3950, 26130, 44606, 217613,
                                              28956, 12157, 29218,
                                              3950, 44606, 217613, 28956,
                                              29218]),
            OGU_COVERAGE_KEY: SparseArray([
                3.318807134, 1.709343188, 1.517313415, 1.194203338,
                1.054848913, 1.083940392, 1.035332215,
                3.318807134, 1.517313415, 11.94203338, 1.054848913,
                1.035332215])
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        lengths_df = pd.DataFrame(self.ogu_lengths_dict)
        expected_out_df = pd.DataFrame(expected_out_dict)

        read_len = 150
        min_coverage = 1

        output_df, output_msgs = _prepare_cell_counts_calc_df(
            counts_df, lengths_df, read_len, min_coverage)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual(
            ["The following items have coverage lower than the minimum of 1: "
             "['A;Lactobacillus gasseri', 'A;Ruminococcus albus', "
             "'A;Escherichia coli', 'A;Tyzzerella nexilis', "
             "'A;Prevotella sp. oral taxon 299', 'A;Streptococcus mitis', "
             "'B;Lactobacillus gasseri', 'B;Ruminococcus albus', "
             "'B;Escherichia coli', 'B;Tyzzerella nexilis', "
             "'B;Prevotella sp. oral taxon 299', 'B;Streptococcus mitis', "
             "'B;Neisseria subflava', 'B;Haemophilus influenzae']"],
            output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample(self):
        input_dict = self._combine_inputs()
        input_df = pd.DataFrame(input_dict)
        mass_ratio_df = pd.DataFrame(self.mass_ratio_dict)

        expected_additions_dict = {
            k: self.example1_ogu_full_outputs_short_avogadro_dict[k] for k in
            (OGU_ID_KEY, OGU_GDNA_MASS_NG_KEY,
             OGU_GENOMES_PER_G_OF_GDNA_KEY,
             OGU_CELLS_PER_G_OF_GDNA_KEY,
             OGU_CELLS_PER_G_OF_SAMPLE_KEY)}

        sample_a_df = input_df[input_df[SAMPLE_ID_KEY] == "example1"]
        expected_add_df = pd.DataFrame(expected_additions_dict)
        expected_out_df = sample_a_df.merge(expected_add_df, on=OGU_ID_KEY)

        sample_id = "example1"
        min_rsquared = 0.8

        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, self.linregresses_dict, mass_ratio_df, input_df,
            min_rsquared, is_test=True)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual([], output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample_w_log_msgs_no_model(self):
        input_dict = self._combine_inputs()
        input_df = pd.DataFrame(input_dict)
        mass_ratio_df = pd.DataFrame(self.sample_and_prep_input_dict)

        # No entry for example 1, which should trigger a log message.
        linregresses_dict = copy.deepcopy(self.linregresses_dict)
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
        input_dict = self._combine_inputs()
        input_df = pd.DataFrame(input_dict)
        mass_ratio_df = pd.DataFrame(self.sample_and_prep_input_dict)

        sample_id = "example1"
        high_min_rsquared = 0.99

        # here the minimum r_squared is set to 0.99 (which is probably
        # ridiculously high) so the linear model for sample A will be judged
        # not good enough to use
        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, self.linregresses_dict, mass_ratio_df, input_df,
            high_min_rsquared, is_test=True)

        self.assertIsNone(output_df)
        self.assertListEqual(['R^2 of linear regression for sample example1 '
                              'is 0.9731883614079868, which is less than the '
                              'minimum allowed value of 0.99.'],
                             output_msgs)

    def test__calc_gdna_mass_to_sample_mass_by_sample_df(self):
        inputs_dict = {k: self.sample_and_prep_input_dict[k] for k in
                       (SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
                        SAMPLE_IN_ALIQUOT_MASS_G_KEY, ELUTE_VOL_UL_KEY)}

        expected_dict = {k: self.mass_ratio_dict[k] for k in
                         (SAMPLE_ID_KEY, GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY)}

        inputs_df = pd.DataFrame(inputs_dict)

        expected_series = pd.Series(
            expected_dict[GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY],
            index=expected_dict[SAMPLE_ID_KEY],
            name=GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY)
        expected_series.index.name = SAMPLE_ID_KEY

        output_series = _calc_gdna_mass_to_sample_mass_by_sample_df(inputs_df)
        pd.testing.assert_series_equal(expected_series, output_series)

    def test__calc_ogu_gdna_mass_ng_series_for_sample(self):
        input_dict = {k: self.example1_ogu_full_inputs_dict[k] for k in
                      (OGU_ID_KEY, OGU_READ_COUNT_KEY)}

        # Inputs are taken from the values for A1_pool1_Fwd in the
        # linear models file at https://github.com/lzaramela/SynDNA/blob/main/data/saliva_linear_models.tsv ,
        # EXCEPT the values of the a_intercept and b_intercept columns are
        # negated (because the Zaramela code generates regression models that
        # predict the *negative* log10 of the read weight while the code under
        # test predicts just log10 of the read weight).
        slope = 1.24487652379132
        intercept = -6.77539505390338

        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(
            self.example1_ogu_full_outputs_short_avogadro_dict[
                OGU_GDNA_MASS_NG_KEY],
            index=self.example1_ogu_full_inputs_dict[OGU_ID_KEY],
            name=OGU_GDNA_MASS_NG_KEY)
        expected_series.index.name = OGU_ID_KEY

        output_series = _calc_ogu_gdna_mass_ng_series_for_sample(
            input_df, slope, intercept)

        assert_series_equal(expected_series, output_series)

    def test__calc_ogu_genomes_per_g_of_gdna_series_for_sample(self):
        # this is the default value for our experimental system
        total_sample_gdna_mass_ng = 5

        input_dict = {k: self.example1_ogu_full_outputs_short_avogadro_dict[k]
                      for k in
                      (OGU_ID_KEY, OGU_LEN_IN_BP_KEY, OGU_GDNA_MASS_NG_KEY)}
        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(
            self.example1_ogu_full_outputs_short_avogadro_dict[
                OGU_CELLS_PER_G_OF_GDNA_KEY],
            index=self.example1_ogu_full_outputs_short_avogadro_dict[
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

        input_dict = {k: self.example1_ogu_full_outputs_short_avogadro_dict[k]
                      for k in
                      (OGU_ID_KEY, OGU_LEN_IN_BP_KEY, OGU_GDNA_MASS_NG_KEY)}
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
