import biom.table
import numpy as np
import os
import pandas
from pandas.testing import assert_frame_equal
from unittest import TestCase
from pysyndna import calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs, \
    calc_copies_of_ogu_orf_ssrna_per_g_sample, \
    calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita
from pysyndna.src.quant_orfs import _read_ogu_orf_coords_to_df, \
    _calc_ogu_orf_copies_per_g_from_coords, \
    _calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs, \
    OGU_ORF_ID_KEY, OGU_ORF_START_KEY, OGU_ORF_END_KEY, OGU_ORF_LEN_KEY, \
    COPIES_PER_G_OGU_ORF_SSRNA_KEY, SAMPLE_ID_KEY, \
    SAMPLE_IN_ALIQUOT_MASS_G_KEY, SSRNA_CONCENTRATION_NG_UL_KEY, \
    ELUTE_VOL_UL_KEY, TOTAL_BIOLOGICAL_READS_KEY


class TestQuantOrfs(TestCase):
    COORDS_DICT = {
        OGU_ORF_ID_KEY: ["G000005825_1", "G000005825_2", "G000005825_3",
                         "G000005825_4", "G000005825_5", "G900163845_3247",
                         "G900163845_3248", "G900163845_3249",
                         "G900163845_3250", "G900163845_3251"],
        OGU_ORF_START_KEY: [816, 2348, 3744, 3971, 5098, 3392209, 3393051,
                            3393938, 3394702, 3395077],
        OGU_ORF_END_KEY: [2168, 3490, 3959, 5086, 5373, 3390413, 3392206,
                          3393048, 3393935, 3395721]
    }

    LEN_AND_COPIES_DICT = {
            OGU_ORF_ID_KEY: ["G000005825_1", "G000005825_2", "G000005825_3",
                             "G000005825_4", "G000005825_5", "G900163845_3247",
                             "G900163845_3248", "G900163845_3249",
                             "G900163845_3250", "G900163845_3251"],
            OGU_ORF_LEN_KEY: [1353, 1143, 216, 1116, 276, 1797, 846, 891,
                              768, 645],
            COPIES_PER_G_OGU_ORF_SSRNA_KEY: [1.3091041E+18, 1.5496219E+18,
                                             8.2000827E+18, 1.5871128E+18,
                                             6.4174561E+18, 9.8565268E+17,
                                             2.0936381E+18, 1.9878988E+18,
                                             2.3062733E+18, 2.7460742E+18]
        }

    SAMPLE_IDS = ["IBSRS3526007", "IQSRS3526010"]
    COUNT_VALS = np.array([
        [0, 0],
        [2, 0],
        [0, 1],
        [35, 0],
        [0, 694],
        [10292, 382],
        [0, 0],
        [190, 10],
        [0, 630],
        [34, 1003]])

    PARAMS_DICT = {
        SAMPLE_ID_KEY: SAMPLE_IDS,
        SAMPLE_IN_ALIQUOT_MASS_G_KEY: [0.003, 0.00082],
        SSRNA_CONCENTRATION_NG_UL_KEY: [0.132714286, 0.0042],
        ELUTE_VOL_UL_KEY: [70, 70],
        TOTAL_BIOLOGICAL_READS_KEY: [213988, 3028580]
    }

    COPIES_PER_G_SAMPLE_VALS = np.array([
        [0, 0],
        [4.4849829E+07, 0],
        [0, 9.7076176E+05],
        [8.0386085E+08, 0],
        [0, 5.2725026E+08],
        [1.4680090E+11, 4.4574009E+07],
        [0, 0],
        [5.4657898E+09, 2.3533619E+06],
        [0, 1.7200685E+08],
        [1.3511272E+09, 3.2606759E+08]])

    def setUp(self):
        self.maxDiff = None
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def test__read_ogu_orf_coords_to_df(self):
        expected_df = pandas.DataFrame(self.COORDS_DICT)

        ogu_orf_coords_fp = os.path.join(self.data_dir, "coords.txt")
        output_df = _read_ogu_orf_coords_to_df(ogu_orf_coords_fp)
        assert_frame_equal(output_df, expected_df)

    def test__calc_ogu_orf_copies_per_g_from_coords(self):
        expected_dict = self.COORDS_DICT.copy()
        expected_dict.update(self.LEN_AND_COPIES_DICT)
        expected_df = pandas.DataFrame(
            expected_dict, index=expected_dict[OGU_ORF_ID_KEY])
        expected_df.index.name = OGU_ORF_ID_KEY

        input_df = pandas.DataFrame(self.COORDS_DICT)
        output_df = _calc_ogu_orf_copies_per_g_from_coords(input_df)

        assert_frame_equal(expected_df, output_df)

    def test__calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs(self):
        input_quant_params_per_sample_df = pandas.DataFrame(self.PARAMS_DICT)
        input_ogu_orf_copies_per_g_ssrna_df = pandas.DataFrame(
            self.LEN_AND_COPIES_DICT,
            index=self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY])

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_biom = biom.table.Table(
            self.COPIES_PER_G_SAMPLE_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        output_biom = _calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs(
            input_quant_params_per_sample_df,
            input_reads_per_ogu_orf_per_sample_biom,
            input_ogu_orf_copies_per_g_ssrna_df)

        # NB: Comparing the bioms as dataframes because the biom equality
        # compare does not allow "almost equal" checking for float values,
        # whereas rtol and atol are built in to assert_frame_equal
        output_df = output_biom.to_dataframe()
        expected_df = expected_biom.to_dataframe()
        pandas.testing.assert_frame_equal(output_df, expected_df)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs(self):
        # This is the same as the previous test, but using the public function
        input_quant_params_per_sample_df = pandas.DataFrame(self.PARAMS_DICT)
        input_ogu_orf_copies_per_g_ssrna_df = pandas.DataFrame(
            self.LEN_AND_COPIES_DICT,
            index=self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY])

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_biom = biom.table.Table(
            self.COPIES_PER_G_SAMPLE_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        output_biom = calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs(
            input_quant_params_per_sample_df,
            input_reads_per_ogu_orf_per_sample_biom,
            input_ogu_orf_copies_per_g_ssrna_df)

        # NB: Comparing the bioms as dataframes because the biom equality
        # compare does not allow "almost equal" checking for float values,
        # whereas rtol and atol are built in to assert_frame_equal
        output_df = output_biom.to_dataframe()
        expected_df = expected_biom.to_dataframe()
        pandas.testing.assert_frame_equal(output_df, expected_df)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs_ids_err(self):
        # drop the first sample from the params dataframe; now the reads
        # will contain a sample that the params dataframe does not
        input_quant_params_per_sample_df = pandas.DataFrame(self.PARAMS_DICT)
        input_quant_params_per_sample_df.drop(index=0, axis=0, inplace=True)

        input_ogu_orf_copies_per_g_ssrna_df = pandas.DataFrame(
            self.LEN_AND_COPIES_DICT,
            index=self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY])

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_msg = r"Found sample ids in reads data that were not in" \
                       r" sample info: \{'IBSRS3526007'\}"
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs(
                input_quant_params_per_sample_df,
                input_reads_per_ogu_orf_per_sample_biom,
                input_ogu_orf_copies_per_g_ssrna_df)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs_col_err(self):
        params_dict = self.PARAMS_DICT.copy()

        # drop a necessary column from the params dict
        del params_dict[TOTAL_BIOLOGICAL_READS_KEY]
        input_quant_params_per_sample_df = pandas.DataFrame(params_dict)
        input_quant_params_per_sample_df.drop(index=0, axis=0, inplace=True)

        input_ogu_orf_copies_per_g_ssrna_df = pandas.DataFrame(
            self.LEN_AND_COPIES_DICT,
            index=self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY])

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_msg = r"parameters dataframe is missing required " \
                       r"column\(s\): \['total_biological_reads_r1r2'\]"
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = calc_copies_of_ogu_orf_ssrna_per_g_sample_from_dfs(
                input_quant_params_per_sample_df,
                input_reads_per_ogu_orf_per_sample_biom,
                input_ogu_orf_copies_per_g_ssrna_df)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample(self):
        input_quant_params_per_sample_df = pandas.DataFrame(self.PARAMS_DICT)
        ogu_orf_coords_fp = os.path.join(self.data_dir, "coords.txt")

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_biom = biom.table.Table(
            self.COPIES_PER_G_SAMPLE_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        output_biom = calc_copies_of_ogu_orf_ssrna_per_g_sample(
            input_quant_params_per_sample_df,
            input_reads_per_ogu_orf_per_sample_biom,
            ogu_orf_coords_fp)

        # NB: Comparing the bioms as dataframes because the biom equality
        # compare does not allow "almost equal" checking for float values,
        # whereas rtol and atol are built in to assert_frame_equal
        output_df = output_biom.to_dataframe()
        expected_df = expected_biom.to_dataframe()
        pandas.testing.assert_frame_equal(output_df, expected_df)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita(self):
        sample_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                            [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]}

        prep_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                          [SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY,
                           SSRNA_CONCENTRATION_NG_UL_KEY,
                           TOTAL_BIOLOGICAL_READS_KEY]}

        sample_info_df = pandas.DataFrame(sample_info_dict)
        prep_info_df = pandas.DataFrame(prep_info_dict)
        ogu_orf_coords_fp = os.path.join(self.data_dir, "coords.txt")

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_biom = biom.table.Table(
            self.COPIES_PER_G_SAMPLE_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        output_biom = calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita(
            sample_info_df, prep_info_df,
            input_reads_per_ogu_orf_per_sample_biom,
            ogu_orf_coords_fp)

        # NB: Comparing the bioms as dataframes because the biom equality
        # compare does not allow "almost equal" checking for float values,
        # whereas rtol and atol are built in to assert_frame_equal
        output_df = output_biom.to_dataframe()
        expected_df = expected_biom.to_dataframe()
        pandas.testing.assert_frame_equal(output_df, expected_df)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita_w_casts(self):
        sample_info_dict = {k: [str(x) for x in self.PARAMS_DICT[k]] for k in
                            [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]}

        prep_info_dict = {k: [str(x) for x in self.PARAMS_DICT[k]] for k in
                          [SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY,
                           SSRNA_CONCENTRATION_NG_UL_KEY,
                           TOTAL_BIOLOGICAL_READS_KEY]}

        sample_info_df = pandas.DataFrame(sample_info_dict)
        prep_info_df = pandas.DataFrame(prep_info_dict)
        ogu_orf_coords_fp = os.path.join(self.data_dir, "coords.txt")

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_biom = biom.table.Table(
            self.COPIES_PER_G_SAMPLE_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        output_biom = calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita(
            sample_info_df, prep_info_df,
            input_reads_per_ogu_orf_per_sample_biom,
            ogu_orf_coords_fp)

        # NB: Comparing the bioms as dataframes because the biom equality
        # compare does not allow "almost equal" checking for float values,
        # whereas rtol and atol are built in to assert_frame_equal
        output_df = output_biom.to_dataframe()
        expected_df = expected_biom.to_dataframe()
        pandas.testing.assert_frame_equal(output_df, expected_df)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita_col_err(self):
        sample_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                            [SAMPLE_ID_KEY]}

        prep_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                          [SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY,
                           SSRNA_CONCENTRATION_NG_UL_KEY,
                           TOTAL_BIOLOGICAL_READS_KEY]}

        sample_info_df = pandas.DataFrame(sample_info_dict)
        prep_info_df = pandas.DataFrame(prep_info_dict)
        ogu_orf_coords_fp = os.path.join(self.data_dir, "coords.txt")

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_msg = r"sample info is missing required " \
                       r"column\(s\): \['calc_mass_sample_aliquot_input_g'\]"
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita(
                sample_info_df, prep_info_df,
                input_reads_per_ogu_orf_per_sample_biom,
                ogu_orf_coords_fp)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita_col_err2(self):
        sample_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                            [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]}

        prep_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                          [SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY,
                           SSRNA_CONCENTRATION_NG_UL_KEY]}

        sample_info_df = pandas.DataFrame(sample_info_dict)
        prep_info_df = pandas.DataFrame(prep_info_dict)
        ogu_orf_coords_fp = os.path.join(self.data_dir, "coords.txt")

        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_msg = r"prep info is missing required " \
                       r"column\(s\): \['total_biological_reads_r1r2'\]"
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita(
                sample_info_df, prep_info_df,
                input_reads_per_ogu_orf_per_sample_biom,
                ogu_orf_coords_fp)

    def test_calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita_id_err(self):
        sample_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                            [SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY]}

        prep_info_dict = {k: self.PARAMS_DICT[k].copy() for k in
                          [SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY,
                           SSRNA_CONCENTRATION_NG_UL_KEY,
                           TOTAL_BIOLOGICAL_READS_KEY]}

        sample_info_df = pandas.DataFrame(sample_info_dict)

        # drop the first sample from the prep dataframe; now the sample info
        # will contain a sample that the prep dataframe does not.
        prep_info_df = pandas.DataFrame(prep_info_dict)
        prep_info_df.drop(index=0, axis=0, inplace=True)

        ogu_orf_coords_fp = os.path.join(self.data_dir, "coords.txt")
        input_reads_per_ogu_orf_per_sample_biom = biom.table.Table(
            self.COUNT_VALS,
            self.LEN_AND_COPIES_DICT[OGU_ORF_ID_KEY],
            self.SAMPLE_IDS)

        expected_msg = (r"Found sample ids in reads data that were not in "
                        r"sample info: \{'IBSRS3526007'\}")
        with self.assertRaisesRegex(ValueError, expected_msg):
            _ = calc_copies_of_ogu_orf_ssrna_per_g_sample_for_qiita(
                sample_info_df, prep_info_df,
                input_reads_per_ogu_orf_per_sample_biom,
                ogu_orf_coords_fp)
