import biom.table
import numpy as np
import pandas as pd
from pandas.arrays import SparseArray
from pandas.testing import assert_series_equal
import os
from unittest import TestCase
from src.calc_cell_counts import OGU_ID_KEY, OGU_READ_COUNT_KEY, \
    OGU_LEN_IN_BP_KEY, OGU_GDNA_MASS_NG_KEY, OGU_GENOMES_PER_G_OF_GDNA_KEY, \
    OGU_CELLS_PER_G_OF_GDNA_KEY, SAMPLE_ID_KEY, \
    GDNA_CONCENTRATION_NG_UL_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY, \
    ELUTE_VOL_UL_KEY, GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY, \
    OGU_CELLS_PER_G_OF_SAMPLE_KEY, TOTAL_OGU_READS_KEY, OGU_COVERAGE_KEY, \
    _calc_long_format_ogu_cell_counts_df, \
    _prepare_cell_counts_calc_df, \
    _calc_ogu_cell_counts_df_for_sample, \
    _calc_gdna_mass_to_sample_mass_by_sample_df, \
    _calc_ogu_gdna_mass_ng_series_for_sample, \
    _calc_ogu_genomes_per_g_of_gdna_series_for_sample, \
    calc_ogu_cell_counts_biom, calc_ogu_cell_counts_per_g_of_sample_for_qiita


class TestCalcCellCounts(TestCase):
    def setUp(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test_calc_ogu_cell_counts_per_g_of_sample_for_qiita(self):
        sample_id_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.027829017, 0.0402847],
        }

        prep_info_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            GDNA_CONCENTRATION_NG_UL_KEY: [5.7, 12.6],
            ELUTE_VOL_UL_KEY: [70, 100],
        }

        ogu_ids = ["Lactobacillus gasseri", "Ruminococcus albus",
                   "Escherichia coli", "Tyzzerella nexilis",
                   "Prevotella sp. oral taxon 299",
                   "Streptococcus mitis", "Leptolyngbya valderiana",
                   "Neisseria subflava", "Neisseria flavescens",
                   "Fusobacterium periodonticum",
                   "Streptococcus pneumoniae", "Haemophilus influenzae",
                   "Veillonella dispar"]
        sample_ids = ["A", "B"]
        counts_vals = np.array([[79950, 79951],
                                [93024, 93024],
                                [86188, 86188],
                                [45441, 45441],
                                [31185, 31185],
                                [24929, 24929],
                                [1975, 1975],
                                [26130, 0],
                                [22303, 22303],
                                [19783, 197830],
                                [14478, 14478],
                                [12145, 12],
                                [14609, 14609]])

        sample_info_df = pd.DataFrame(sample_id_dict)
        prep_info_df = pd.DataFrame(prep_info_dict)
        counts_biom = biom.table.Table(counts_vals, ogu_ids, sample_ids)
        models_fp = os.path.join(self.test_data_dir, "models.yml")
        lengths_fp = os.path.join(self.test_data_dir, "ogu_lengths.tsv")

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8

        output_dict = calc_ogu_cell_counts_per_g_of_sample_for_qiita(
            sample_info_df, prep_info_df, models_fp, counts_biom,
            lengths_fp, read_len, min_coverage, min_rsquared)

        # TODO: finish test
        print(output_dict)

    def test_calc_ogu_cell_counts_biom(self):
        params_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            GDNA_CONCENTRATION_NG_UL_KEY: [5.7, 12.6],
            SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.027829017, 0.0402847],
            ELUTE_VOL_UL_KEY: [70, 100],
        }

        ogu_ids = ["Lactobacillus gasseri", "Ruminococcus albus",
                   "Escherichia coli", "Tyzzerella nexilis",
                   "Prevotella sp. oral taxon 299",
                   "Streptococcus mitis", "Leptolyngbya valderiana",
                   "Neisseria subflava", "Neisseria flavescens",
                   "Fusobacterium periodonticum",
                   "Streptococcus pneumoniae", "Haemophilus influenzae",
                   "Veillonella dispar"]
        sample_ids = ["A", "B"]
        counts_vals = np.array([[79950, 79951],
                                [93024, 93024],
                                [86188, 86188],
                                [45441, 45441],
                                [31185, 31185],
                                [24929, 24929],
                                [1975, 1975],
                                [26130, 0],
                                [22303, 22303],
                                [19783, 197830],
                                [14478, 14478],
                                [12145, 12],
                                [14609, 14609]])

        lengths_dict = {
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
                                2484878.333, 2058778.25, 1680673.6, 2116567],
        }

        linregresses_dict = {
            'A': {
                "slope": 1.24487652379132, "intercept": -6.77539505390338,
                "rvalue": 0.9865030975156575, "pvalue": 1.428443560659758e-07,
                "stderr": 0.07305408550335003,
                "intercept_stderr": 0.2361976278251443},
            'B': {
                "slope": 1.24675913604407, "intercept": -7.155318973708384,
                "rvalue": 0.9863241797356326, "pvalue": 1.505381146809759e-07,
                "stderr": 0.07365795255302438,
                "intercept_stderr": 0.2563956755844754}
        }

        params_df = pd.DataFrame(params_dict)
        counts_biom = biom.table.Table(counts_vals, ogu_ids, sample_ids)
        lengths_df = pd.DataFrame(lengths_dict)

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8
        output_metric = OGU_CELLS_PER_G_OF_GDNA_KEY

        output_biom, output_msgs = calc_ogu_cell_counts_biom(
            params_df, linregresses_dict, counts_biom, lengths_df,
            read_len, min_coverage, min_rsquared, output_metric)

        # TODO: finish test
        print(output_msgs)

    def test__calc_long_format_ogu_cell_counts_df(self):
        counts_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            "A": SparseArray([79950, 93024, 86188, 45441, 31185, 24929,
                              1975, 26130, 22303, 19783, 14478, 12145, 14609]),
            "B": SparseArray([79951, 93024, 86188, 45441, 31185, 24929,
                              1975, 0, 22303, 197830, 14478, 12, 14609]),
        }

        lengths_dict = {
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
                                2484878.333, 2058778.25, 1680673.6, 2116567],
        }

        linregresses_dict = {
            'A': {
                "slope": 1.24487652379132, "intercept": -6.77539505390338,
                "rvalue": 0.9865030975156575, "pvalue": 1.428443560659758e-07,
                "stderr": 0.07305408550335003,
                "intercept_stderr": 0.2361976278251443},
            'B': {
                "slope": 1.24675913604407, "intercept": -7.155318973708384,
                "rvalue": 0.9863241797356326, "pvalue": 1.505381146809759e-07,
                "stderr": 0.07365795255302438,
                "intercept_stderr": 0.2563956755844754}
        }

        mass_ratio_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY: [0.014337553, 0.031277383]
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        mass_ratio_df = pd.DataFrame(mass_ratio_dict)
        lengths_df = pd.DataFrame(lengths_dict)

        read_len = 150
        min_coverage = 1
        min_rsquared = 0.8

        output_df, output_msgs = _calc_long_format_ogu_cell_counts_df(
            linregresses_dict, counts_df, lengths_df, mass_ratio_df,
            read_len, min_coverage, min_rsquared)

        # TODO: finish test
        print(output_df)

    def test__prepare_cell_counts_calc_df_w_log_msgs_low_coverage(self):
        # Inputs and expected results for sample A are taken from cell directly
        # under the header "Applying the linear models to sequencing data" of
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # RawCounts column is OGU_READ_COUNT_KEY.
        # Sample B inputs/results are just slightly modified versions of A
        counts_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            "A": SparseArray([79950, 93024, 86188, 45441, 31185, 24929,
                              1975, 26130, 22303, 19783, 14478, 12145, 14609]),
            "B": SparseArray([79951, 93024, 86188, 45441, 31185, 24929,
                              1975, 0, 22303, 197830, 14478, 12, 14609]),
        }

        lengths_dict = {
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
                                2484878.333, 2058778.25, 1680673.6, 2116567],
        }

        expected_out_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar",
                         "Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae",
                         "Veillonella dispar"],
            SAMPLE_ID_KEY: ["A", "A", "A", "A", "A", "A",
                            "A", "A", "A", "A", "A", "A", "A",
                            "B", "B", "B", "B", "B", "B",
                            "B", "B", "B", "B", "B"],
            OGU_READ_COUNT_KEY: SparseArray([79950, 93024, 86188, 45441,
                                             31185, 24929, 1975, 26130, 22303,
                                             19783, 14478, 12145, 14609, 79951,
                                             93024, 86188, 45441, 31185, 24929,
                                             1975, 22303, 197830, 14478,
                                             14609]),
            OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567,
                                1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2204851,
                                2484878.333, 2058778.25, 2116567],
            TOTAL_OGU_READS_KEY: [159901, 186048, 172376, 90882, 62370, 49858,
                                  3950, 26130, 44606, 217613, 28956, 12157,
                                  29218,
                                  159901, 186048, 172376, 90882, 62370, 49858,
                                  3950, 44606, 217613, 28956, 29218],
            OGU_COVERAGE_KEY: [6.295975144, 3.19032039, 2.568624973, 1.7653773,
                               1.906928906, 1.840909863, 3.318807134,
                               1.709343188, 1.517313415, 1.194203338,
                               1.054848913, 1.083940392, 1.035332215,
                               6.296053893, 3.19032039, 2.568624973,
                               1.7653773, 1.906928906, 1.840909863,
                               3.318807134, 1.517313415, 11.94203338,
                               1.054848913, 1.035332215]
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        lengths_df = pd.DataFrame(lengths_dict)
        expected_out_df = pd.DataFrame(expected_out_dict)

        read_len = 150
        min_coverage = 1

        output_df, output_msgs = _prepare_cell_counts_calc_df(
            counts_df, lengths_df, read_len, min_coverage)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual(["The following items have coverage lower "
                              "than the minimum of 1: ['B;Neisseria subflava',"
                              " 'B;Haemophilus influenzae']"], output_msgs)

    def test__prepare_cell_counts_calc_df_v_sparse(self):
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

        lengths_dict = {
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
                                2484878.333, 2058778.25, 1680673.6, 2116567],
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
            OGU_COVERAGE_KEY: SparseArray([3.318807134,
                               1.709343188, 1.517313415, 1.194203338,
                               1.054848913, 1.083940392, 1.035332215,
                               3.318807134, 1.517313415, 11.94203338,
                               1.054848913, 1.035332215])
        }

        counts_df = pd.DataFrame(counts_dict)
        counts_df.set_index(OGU_ID_KEY, inplace=True)
        lengths_df = pd.DataFrame(lengths_dict)
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
        # Inputs and expected results for sample A are taken from cell directly
        # under the header "Applying the linear models to sequencing data" of
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # RawCounts column is OGU_READ_COUNT_KEY.
        # Sample B inputs/results are just slightly modified versions of A
        input_dict = {
            SAMPLE_ID_KEY: ["A", "A", "A", "A", "A", "A",
                            "A", "A", "A", "A","A", "A", "A",
                            "B", "B", "B", "B", "B", "B",
                            "B", "B", "B", "B", "B", "B", "B"],
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar",
                         "Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            OGU_READ_COUNT_KEY: [79950, 93024, 86188, 45441, 31185, 24929,
                                 1975, 26130, 22303, 19783, 14478, 12145,
                                 14609,
                                 79951, 93024, 86188, 45441, 31185, 24929,
                                 1975, 0, 22303, 197830, 14478, 12, 14609],
            OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567,
                                1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567],

        }
        input_df = pd.DataFrame(input_dict)

        linregresses_dict = {
            'A': {
                "slope": 1.24487652379132, "intercept": -6.77539505390338,
                "rvalue": 0.9865030975156575, "pvalue": 1.428443560659758e-07,
                "stderr": 0.07305408550335003,
                "intercept_stderr": 0.2361976278251443},
            'B': {
                "slope": 1.24675913604407, "intercept": -7.155318973708384,
                "rvalue": 0.9863241797356326, "pvalue": 1.505381146809759e-07,
                "stderr": 0.07365795255302438,
                "intercept_stderr": 0.2563956755844754}
        }

        mass_ratio_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY: [0.014337553, 0.031277383]
        }
        mass_ratio_df = pd.DataFrame(mass_ratio_dict)

        expected_additions_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
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
            OGU_GENOMES_PER_G_OF_GDNA_KEY: [263469.8016, 138550.8742,
                                            109485.9657, 64330.93757,
                                            63369.31482, 57911.52782,
                                            56114.06446, 54395.84228,
                                            46448.32735, 35499.48595,
                                            29049.10845, 28593.0947,
                                            28574.60346],
            OGU_CELLS_PER_G_OF_GDNA_KEY: [263469.8016, 138550.8742,
                                          109485.9657, 64330.93757,
                                          63369.31482, 57911.52782,
                                          56114.06446, 54395.84228,
                                          46448.32735, 35499.48595,
                                          29049.10845, 28593.0947,
                                          28574.60346],
            OGU_CELLS_PER_G_OF_SAMPLE_KEY: [3777.512244, 1986.480502,
                                            1569.760836,
                                            922.3482269, 908.5609098,
                                            830.3095994, 804.5383732,
                                            779.9032717, 665.9553551,
                                            508.9757613, 416.493132,
                                            409.9550107, 409.6898916]
        }

        sample_a_df = input_df[input_df[SAMPLE_ID_KEY] == "A"]
        expected_add_df = pd.DataFrame(expected_additions_dict)
        expected_out_df = sample_a_df.merge(expected_add_df, on=OGU_ID_KEY)

        sample_id = "A"
        min_rsquared = 0.8

        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, linregresses_dict, mass_ratio_df, input_df,
            min_rsquared, is_test=True)

        pd.testing.assert_frame_equal(expected_out_df, output_df)
        self.assertListEqual([], output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample_w_log_msgs_no_model(self):
        # Inputs and expected results for sample A are taken from cell directly
        # under the header "Applying the linear models to sequencing data" of
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # RawCounts column is OGU_READ_COUNT_KEY.
        # Sample B inputs/results are just slightly modified versions of A
        input_dict = {
            SAMPLE_ID_KEY: ["A", "A", "A", "A", "A", "A",
                            "A", "A", "A", "A","A", "A", "A",
                            "B", "B", "B", "B", "B", "B",
                            "B", "B", "B", "B", "B", "B", "B"],
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar",
                         "Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            OGU_READ_COUNT_KEY: [79950, 93024, 86188, 45441, 31185, 24929,
                                 1975, 26130, 22303, 19783, 14478, 12145,
                                 14609,
                                 79951, 93024, 86188, 45441, 31185, 24929,
                                 1975, 0, 22303, 197830, 14478, 12, 14609],
            OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567,
                                194788.333, 437330, 50331200.886, 381016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1673.6, 211667],

        }
        input_df = pd.DataFrame(input_dict)

        mass_ratio_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY: [0.014337553, 0.031277383]
        }
        mass_ratio_df = pd.DataFrame(mass_ratio_dict)

        # No entry for sample A, which should trigger a log message
        linregresses_dict = {
            'B': {
                "slope": 1.24675913604407, "intercept": -7.155318973708384,
                "rvalue": 0.9863241797356326, "pvalue": 1.505381146809759e-07,
                "stderr": 0.07365795255302438,
                "intercept_stderr": 0.2563956755844754}
        }

        sample_id = "A"
        min_rsquared = 0.8

        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, linregresses_dict, mass_ratio_df, input_df,
            min_rsquared, is_test=True)

        self.assertIsNone(output_df)
        self.assertListEqual(["No linear regression fitted for sample A"],
                             output_msgs)

    def test__calc_ogu_cell_counts_df_for_sample_w_log_msgs_low_rsquared(self):
        # Inputs and expected results for sample A are taken from cell directly
        # under the header "Applying the linear models to sequencing data" of
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # RawCounts column is OGU_READ_COUNT_KEY.
        # Sample B inputs/results are just slightly modified versions of A
        input_dict = {
            SAMPLE_ID_KEY: ["A", "A", "A", "A", "A", "A",
                            "A", "A", "A", "A","A", "A", "A",
                            "B", "B", "B", "B", "B", "B",
                            "B", "B", "B", "B", "B", "B", "B"],
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar",
                         "Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            OGU_READ_COUNT_KEY: [79950, 93024, 86188, 45441, 31185, 24929,
                                 1975, 26130, 22303, 19783, 14478, 12145,
                                 14609,
                                 79951, 93024, 86188, 45441, 31185, 24929,
                                 1975, 0, 22303, 197830, 14478, 12, 14609],
            OGU_LEN_IN_BP_KEY: [1904788.333, 4373730, 5033120.886, 3861016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1680673.6, 2116567,
                                194788.333, 437330, 50331200.886, 381016,
                                2453028, 2031251, 89264, 2292986, 2204851,
                                2484878.333, 2058778.25, 1673.6, 211667],

        }
        input_df = pd.DataFrame(input_dict)

        mass_ratio_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY: [0.014337553, 0.031277383]
        }
        mass_ratio_df = pd.DataFrame(mass_ratio_dict)

        linregresses_dict = {
            'A': {
                "slope": 1.24487652379132, "intercept": -6.77539505390338,
                "rvalue": 0.9865030975156575, "pvalue": 1.428443560659758e-07,
                "stderr": 0.07305408550335003,
                "intercept_stderr": 0.2361976278251443},
            'B': {
                "slope": 1.24675913604407, "intercept": -7.155318973708384,
                "rvalue": 0.9863241797356326, "pvalue": 1.505381146809759e-07,
                "stderr": 0.07365795255302438,
                "intercept_stderr": 0.2563956755844754}
        }

        sample_id = "A"
        high_min_rsquared = 0.99

        # here the minimum r_squared is set to 0.99 (which is probably
        # ridiculously high) so the linear model for sample A will be judged
        # not good enough to use
        output_df, output_msgs = _calc_ogu_cell_counts_df_for_sample(
            sample_id, linregresses_dict, mass_ratio_df, input_df,
            high_min_rsquared, is_test=True)

        self.assertIsNone(output_df)
        self.assertListEqual(['R^2 of linear regression for sample A is '
                              '0.9731883614079868, which is less than the '
                              'minimum allowed value of 0.99.'],
                             output_msgs)

    def test__calc_gdna_mass_to_sample_mass_by_sample_df(self):
        inputs_dict = {
            SAMPLE_ID_KEY: ["X", "Y"],
            GDNA_CONCENTRATION_NG_UL_KEY: [5.7, 12.6],
            SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.027829017, 0.0402847],
            ELUTE_VOL_UL_KEY: [70, 100],
        }

        expected_dict = {
            SAMPLE_ID_KEY: ["X", "Y"],
            GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY: [0.014337553, 0.031277383]
        }

        inputs_df = pd.DataFrame(inputs_dict)

        expected_series = pd.Series(
            expected_dict[GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY],
            index=expected_dict[SAMPLE_ID_KEY],
            name=GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY)
        expected_series.index.name = SAMPLE_ID_KEY

        output_series = _calc_gdna_mass_to_sample_mass_by_sample_df(inputs_df)
        pd.testing.assert_series_equal(expected_series, output_series)

    def test__calc_ogu_gdna_mass_ng_series_for_sample(self):
        # Inputs and expected results are taken from cell directly under the
        # header "Applying the linear models to sequencing data" of the
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # RawCounts column is OGU_READ_COUNT_KEY.
        input_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            OGU_READ_COUNT_KEY: [79950, 93024, 86188, 45441, 31185, 24929,
                                 1975, 26130, 22303, 19783, 14478, 12145,
                                 14609],
        }

        expected_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
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
                                   0.06528070535624650000]
        }

        slope = 1.24487652379132
        intercept = -6.77539505390338

        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(expected_dict[OGU_GDNA_MASS_NG_KEY],
                                    index=expected_dict[OGU_ID_KEY],
                                    name=OGU_GDNA_MASS_NG_KEY)
        expected_series.index.name = OGU_ID_KEY

        output_series = _calc_ogu_gdna_mass_ng_series_for_sample(
            input_df, slope, intercept)

        assert_series_equal(expected_series, output_series)

    def test__calc_ogu_genomes_per_g_of_gdna_series_for_sample(self):
        # Inputs and expected results are taken from cell directly under the
        # header "Applying the linear models to sequencing data" of the
        # https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        # notebook by capturing the value of the dataframe when i=1, with an
        # added column holding the "reads2weigth" [sic] values (which are
        # OGU_GDNA_MASS_NG_KEY values). The "GenomeLenght" [sic] column is
        # the OGU_LEN_IN_BP_KEY, the "Species" column is OGU_ID_KEY, and the
        # CellNumber column is OGU_CELLS_PER_G_OF_GDNA_KEY.
        input_dict = {
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
                                2484878.333, 2058778.25, 1680673.6, 2116567],
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
                                   0.06528070535624650000]
        }

        expected_dict = {
            OGU_ID_KEY: ["Lactobacillus gasseri", "Ruminococcus albus",
                         "Escherichia coli", "Tyzzerella nexilis",
                         "Prevotella sp. oral taxon 299",
                         "Streptococcus mitis", "Leptolyngbya valderiana",
                         "Neisseria subflava", "Neisseria flavescens",
                         "Fusobacterium periodonticum",
                         "Streptococcus pneumoniae", "Haemophilus influenzae",
                         "Veillonella dispar"],
            OGU_CELLS_PER_G_OF_GDNA_KEY: [263469.8016, 138550.8742,
                                          109485.9657, 64330.93757,
                                          63369.31482, 57911.52782,
                                          56114.06446, 54395.84228,
                                          46448.32735, 35499.48595,
                                          29049.10845, 28593.0947,
                                          28574.60346]
        }

        input_df = pd.DataFrame(input_dict)
        expected_series = pd.Series(expected_dict[OGU_CELLS_PER_G_OF_GDNA_KEY],
                                    index=expected_dict[OGU_ID_KEY])
        expected_series.index.name = OGU_ID_KEY

        # NOTE: the is_test flag is set to True to use the *truncated* value of
        # Avogadro's number that is used in the notebook that is the source of
        # the expected results.  It should never be set to True in production.
        output_series = _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
            input_df, is_test=True)

        assert_series_equal(expected_series, output_series)
