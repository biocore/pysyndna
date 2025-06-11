import biom
import numpy as np
import numpy.testing as npt
import pandas
from pandas.testing import assert_series_equal, assert_frame_equal
from unittest import TestCase
from pysyndna.src.util import (calc_copies_genomic_element_per_g_series, \
    calc_gs_genomic_element_in_aliquot, get_ids_from_df_or_biom, \
    filter_data_by_sample_info, cast_cols, \
    validate_id_consistency_between_datasets, \
    validate_required_columns_exist, SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY, \
    OGU_ID_KEY)


class Testers(TestCase):
    def assert_dicts_almost_equal(self, d1, d2):
        """Assert that two dicts are almost equal.

        Parameters
        ----------
        d1 : dict
            The first dict to compare
        d2 : dict
            The second dict to compare

        Raises
        ------
        AssertionError
            If the dicts are not almost equal
        """
        self.assertIsInstance(d1, dict)
        self.assertIsInstance(d2, dict)
        self.assertEqual(d1.keys(), d2.keys())
        for k in d1.keys():
            for m in d1[k].keys():
                m1 = d1[k][m]
                m2 = d2[k][m]
                self.assertAlmostEqual(m1, m2)

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


class TestUtils(TestCase):
    def test_get_ids_from_df_or_biom_biom_sample_ids(self):
        input_biom = biom.table.Table(
            np.array([[1, 2], [3, 4]]),
            ['ogu01', 'ogu02'],
            ['sample1', 'sample2'])

        obs_ids = get_ids_from_df_or_biom(input_biom, True)
        expected_ids = ['sample1', 'sample2']
        self.assertListEqual(obs_ids, expected_ids)

    def test_get_ids_from_df_or_biom_biom_ogu_ids(self):
        input_biom = biom.table.Table(
            np.array([[1, 2], [3, 4]]),
            ['ogu01', 'ogu02'],
            ['sample1', 'sample2'])

        obs_ids = get_ids_from_df_or_biom(input_biom, False)
        expected_ids = ['ogu01', 'ogu02']
        self.assertListEqual(obs_ids, expected_ids)

    def test_get_ids_from_df_or_biom_df_sample_ids_explicit(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1', 'sample2'],
            'prep_id': ['prep1', 'prep2'],
        }
        input_df = pandas.DataFrame(input_dict)

        obs_ids = get_ids_from_df_or_biom(input_df, True)
        expected_ids = ['sample1', 'sample2']
        self.assertListEqual(obs_ids, expected_ids)

    def test_get_ids_from_df_or_biom_df_sample_ids_implicit(self):
        input_dict = {
            'sample1': [1, 2],
            'sample2': [3, 4],
        }
        input_df = pandas.DataFrame(input_dict)

        obs_ids = get_ids_from_df_or_biom(input_df, True)
        expected_ids = ['sample1', 'sample2']
        self.assertListEqual(obs_ids, expected_ids)

    def test_get_ids_from_df_or_biom_df_sample_ids_implicit_w_ogu_id_col(self):
        input_dict = {
            OGU_ID_KEY: ['ogu01', 'ogu02'],
            'sample1': [1, 2],
            'sample2': [3, 4],
        }
        input_df = pandas.DataFrame(input_dict)

        obs_ids = get_ids_from_df_or_biom(input_df, True)
        expected_ids = ['sample1', 'sample2']
        self.assertListEqual(obs_ids, expected_ids)

    def test_get_ids_from_df_or_biom_df_ogu_ids_explicit(self):
        input_dict = {
            'ogu_id': ['ogu01', 'ogu02'],
            'prep_id': ['prep1', 'prep2'],
        }
        input_df = pandas.DataFrame(input_dict)

        obs_ids = get_ids_from_df_or_biom(input_df, False)
        expected_ids = ['ogu01', 'ogu02']
        self.assertListEqual(obs_ids, expected_ids)

    def test_get_ids_from_df_or_biom_df_ogu_ids_err(self):
        input_dict = {
            'sample_id': ['sample1', 'sample2'],
            'prep_id': ['prep1', 'prep2'],
        }
        input_df = pandas.DataFrame(input_dict)

        expected_err = "DataFrame does not have a column named 'ogu_id'"
        with self.assertRaisesRegex(ValueError, expected_err):
            _ = get_ids_from_df_or_biom(input_df, False)

    def test_validate_required_columns_exist_true(self):
        input_dict = {
            'sample_id': ['sample1'],
            'prep_id': ['prep1'],
        }
        input_df = pandas.DataFrame(input_dict)
        required_columns = ['sample_id', 'prep_id']

        validate_required_columns_exist(
            input_df, required_columns, "missing")

        # Pass test if we made it this far
        self.assertTrue(True)

    def test_validate_required_columns_exist_err(self):
        input_dict = {
            'sample_id': ['sample1'],
        }
        input_df = pandas.DataFrame(input_dict)
        required_columns = ['sample_id', 'prep_id']

        expected_err = r"missing: \['prep_id'\]"
        with self.assertRaisesRegex(ValueError, expected_err):
            validate_required_columns_exist(
                input_df, required_columns, "missing")

    def test_validate_metadata_vs_prep_id_consistency_true(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1'],
            'color': ['blue'],
        }
        input_df = pandas.DataFrame(input_dict)

        prep_dict = {
            SAMPLE_ID_KEY: ['sample1'],
            'prep_id': ['prep1'],
        }
        prep_df = pandas.DataFrame(prep_dict)

        _ = validate_id_consistency_between_datasets(
                input_df, prep_df, "sample info", "prep info", True)

        # Pass test if we made it this far
        self.assertTrue(True)

    def test_validate_metadata_vs_prep_id_consistency_true_w_msg(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1', 'sample2'],
            'color': ['blue', 'aqua'],
        }
        input_df = pandas.DataFrame(input_dict)

        prep_dict = {
            SAMPLE_ID_KEY: ['sample1'],
            'prep_id': ['prep1'],
        }
        prep_df = pandas.DataFrame(prep_dict)

        not_in_prep_ids = validate_id_consistency_between_datasets(
            input_df, prep_df, "sample info", "prep info", True)

        expected_not_in_prep_ids = ['sample2']
        self.assertEqual(not_in_prep_ids, expected_not_in_prep_ids)

    def test_validate_metadata_vs_prep_id_consistency_err(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1'],
            'color': ['blue'],
        }
        input_df = pandas.DataFrame(input_dict)

        prep_dict = {
            SAMPLE_ID_KEY: ['sample1', 'sample2'],
            'prep_id': ['prep1', 'prep2'],
        }
        prep_df = pandas.DataFrame(prep_dict)

        expected_err = (r"Found sample ids in prep info that were not in "
                        r"sample info: \{'sample2'\}")
        with self.assertRaisesRegex(ValueError, expected_err):
            _ = validate_id_consistency_between_datasets(
                input_df, prep_df, "sample info", "prep info", True)

    def test_validate_coverage_vs_reads_id_consistency_df_true(self):
        coverages_dict = {
            'ogu_id': ['ogu01', 'ogu02'],
            'sample1': [0.1, 2.0],
            'sample2': [3.3, 0.04],
        }
        coverages_df = pandas.DataFrame(coverages_dict)

        reads_dict = {
            'ogu_id': ['ogu01', 'ogu02'],
            'sample1': [1, 2],
            'sample2': [3, 4],
        }
        reads_df = pandas.DataFrame(reads_dict)

        _ = validate_id_consistency_between_datasets(coverages_df, reads_df, "coverage data", "reads data", True)

        # Pass test if we made it this far
        self.assertTrue(True)

    def test_validate_coverage_vs_reads_id_consistency_df_true_w_msg(self):
        coverages_dict = {
            'ogu_id': ['ogu01', 'ogu02'],
            'sample1': [0.1, 2.0],
            'sample2': [3.3, 0.04],
            'sample3': [0.5, 0.6],
        }
        coverages_df = pandas.DataFrame(coverages_dict)

        reads_dict = {
            'ogu_id': ['ogu01', 'ogu02'],
            'sample1': [1, 2],
            'sample2': [3, 4],
        }
        reads_df = pandas.DataFrame(reads_dict)

        not_in_read_ids = validate_id_consistency_between_datasets(
            coverages_df, reads_df, "coverage data", "reads data", True)

        expected_not_in_read_ids = ['sample3']
        self.assertEqual(not_in_read_ids, expected_not_in_read_ids)

    # def test_validate_coverage_vs_reads_id_consistency_df_err(self):
    #     coverages_dict = {
    #         OGU_ID_KEY: ['ogu01'],
    #         'sample1': [0.1],
    #         'sample2': [3.3],
    #     }
    #
    #     coverages_df = pandas.DataFrame(coverages_dict)
    #
    #     reads_dict = {
    #         OGU_ID_KEY: ['ogu01', 'ogu02'],
    #         'sample1': [1, 2],
    #         'sample2': [3, 4],
    #     }
    #
    #     reads_df = pandas.DataFrame(reads_dict)
    #     expected_err = (r"Found sample ids in reads data that were not in "
    #                     r"coverage data: \{'sample2'\}")
    #     with self.assertRaisesRegex(ValueError, expected_err):
    #         _ = validate_id_consistency_between_datasets(
    #             coverages_df, reads_df, "coverage data", "reads data", True)

    def test_validate_metadata_vs_reads_id_consistency_df_true(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1', 'sample2'],
            'color': ['blue', 'aqua'],
        }
        input_df = pandas.DataFrame(input_dict)

        reads_dict = {
            'sample1': [1, 2],
            'sample2': [3, 4],
        }
        reads_df = pandas.DataFrame(reads_dict)

        _ = validate_id_consistency_between_datasets(
                input_df, reads_df, "sample info", "reads data", True)

        # Pass test if we made it this far
        self.assertTrue(True)

    def test_validate_metadata_vs_reads_id_consistency_df_true_w_msg(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1', 'sample2', 'sample3'],
            'color': ['blue', 'aqua', 'cerulean'],
        }
        input_df = pandas.DataFrame(input_dict)

        reads_dict = {
            'sample1': [1, 2],
            'sample2': [3, 4],
        }
        reads_df = pandas.DataFrame(reads_dict)

        not_in_prep_ids = validate_id_consistency_between_datasets(
            input_df, reads_df, "sample info", "reads data", True)

        expected_not_in_prep_ids = ['sample3']
        self.assertEqual(not_in_prep_ids, expected_not_in_prep_ids)

    def test_validate_metadata_vs_reads_id_consistency_df_err(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1'],
            'color': ['blue'],
        }
        input_df = pandas.DataFrame(input_dict)

        reads_dict = {
            'sample1': [1, 2],
            'sample2': [3, 4],
        }
        reads_df = pandas.DataFrame(reads_dict)

        expected_err = (r"Found sample ids in reads data that were not in "
                        r"sample info: \{'sample2'\}")
        with self.assertRaisesRegex(ValueError, expected_err):
            _ = validate_id_consistency_between_datasets(
                    input_df, reads_df, "sample info", "reads data", True)

    def test_validate_metadata_vs_reads_id_consistency_biom_true(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1', 'sample2'],
            'color': ['blue', 'aqua'],
        }
        input_df = pandas.DataFrame(input_dict)

        reads_biom = biom.table.Table(
            np.array([[1, 2], [3, 4]]),
            ['obs1', 'obs2'],
            ['sample1', 'sample2'])

        _ = validate_id_consistency_between_datasets(
                input_df, reads_biom, "sample info", "reads data", True)

        # Pass test if we made it this far
        self.assertTrue(True)

    def test_validate_metadata_vs_reads_id_consistency_biom_true_w_msg(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1', 'sample2', 'sample3'],
            'color': ['blue', 'aqua', 'cerulean'],
        }
        input_df = pandas.DataFrame(input_dict)

        reads_biom = biom.table.Table(
            np.array([[1, 2], [3, 4]]),
            ['obs1', 'obs2'],
            ['sample1', 'sample2'])

        not_in_prep_ids = validate_id_consistency_between_datasets(
            input_df, reads_biom, "sample info", "reads data", True)

        expected_not_in_prep_ids = ['sample3']
        self.assertEqual(not_in_prep_ids, expected_not_in_prep_ids)

    def test_validate_metadata_vs_reads_id_consistency_biom_err(self):
        input_dict = {
            SAMPLE_ID_KEY: ['sample1'],
            'color': ['blue'],
        }
        input_df = pandas.DataFrame(input_dict)

        reads_biom = biom.table.Table(
            np.array([[1, 2], [3, 4]]),
            ['obs1', 'obs2'],
            ['sample1', 'sample2'])

        expected_err = (r"Found sample ids in reads data that were not in "
                        r"sample info: \{'sample2'\}")
        with self.assertRaisesRegex(ValueError, expected_err):
            _ = validate_id_consistency_between_datasets(
                    input_df, reads_biom, "sample info", "reads data", True)

    def test_cast_cols(self):
        # all inputs are strings
        input_dict = {
            'sample_id': ['sample1', 'sample2'],
            'conc_ng_ul': ['1', '2'],
            'elute_vol_ul': ['3', 'not applicable'],
            'mass_key': ['5', '6'],
        }
        input_df = pandas.DataFrame(input_dict)

        # all columns containing just numbers become ints; the specified col
        # that contains a string becomes float bc int type can't hold NaN
        expected_dict = {
            'sample_id': ['sample1', 'sample2'],
            'conc_ng_ul': [1, 2],
            'elute_vol_ul': [3, np.nan],
            'mass_key': [5, 6],
        }
        expected_df = pandas.DataFrame(expected_dict)

        obs_df = cast_cols(
            input_df, ['conc_ng_ul', 'elute_vol_ul', 'mass_key'])
        assert_frame_equal(expected_df, obs_df)

    def test_cast_cols_force_float(self):
        # all inputs are strings
        input_dict = {
            'sample_id': ['sample1', 'sample2'],
            'conc_ng_ul': ['1', '2'],
            'elute_vol_ul': ['3.0', 'not applicable'],
            'mass_key': ['5.0', '6.0'],
        }
        input_df = pandas.DataFrame(input_dict)

        # specified columns become floats even if they could be ints;
        # any that can't become floats become NaN
        expected_dict = {
            'sample_id': ['sample1', 'sample2'],
            'conc_ng_ul': [1.0, 2.0],
            'elute_vol_ul': [3.0, np.nan],
            'mass_key': [5.0, 6.0],
        }
        expected_df = pandas.DataFrame(expected_dict)

        obs_df = cast_cols(
            input_df, ['conc_ng_ul', 'elute_vol_ul', 'mass_key'], True)
        assert_frame_equal(expected_df, obs_df)

    def test_cast_cols_absent_cols(self):
        # all inputs are strings
        input_dict = {
            'sample_id': ['sample1', 'sample2'],
            'mass_key': ['5.0', 'not applicable'],
        }
        input_df = pandas.DataFrame(input_dict)

        # mass_key becomes float, other cols in input list are absent
        expected_dict = {
            'sample_id': ['sample1', 'sample2'],
            'mass_key': [5.0, np.nan],
        }
        expected_df = pandas.DataFrame(expected_dict)

        obs_df = cast_cols(
            input_df, ['conc_ng_ul', 'elute_vol_ul', 'mass_key'])
        assert_frame_equal(expected_df, obs_df)

    def test_filter_data_by_sample_info_df(self):
        sample_ids = ['samp1', 'samp2', 'samp3']
        required_params = ['b', 'c', 'd']

        # make one of the needed params NaN for first sample.
        # make one of the needed params negative for third sample.
        # make one of the UNneeded params NaN for second sample.
        # the first and third should be filtered out of resulting biom table,
        # the second should remain.
        params_dict = {
            'a': sample_ids,
            'b': [np.nan, 0.00082, 0.45],
            'c': [0.132714286, 0.0042, -0.183],
            'd': [70, 70, 70],
            'e': [213988, np.nan, 3031038]
        }

        obs_ids = ["G000005825_1", "G000005825_2", "G000005825_3",
                   "G000005825_4", "G000005825_5", "G900163845_3247",
                   "G900163845_3248", "G900163845_3249",
                   "G900163845_3250", "G900163845_3251"]

        input_dict = {
            sample_ids[0]: [0, 2, 0, 35, 0, 10292, 0, 190, 0, 34],
            sample_ids[1]: [0, 0, 1, 0, 694, 382, 0, 10, 630, 1003],
            sample_ids[2]: [0, 1, 0, 4, 29, 435, 0, 18, 30, 452]}

        input_quant_params_per_sample_df = pandas.DataFrame(
            params_dict, index=sample_ids)
        input_df = pandas.DataFrame(input_dict, index=obs_ids)

        # only one sample left after filtering
        expected_df = pandas.DataFrame(
            {sample_ids[1]: input_dict[sample_ids[1]]},
            index=obs_ids)

        expected_msgs = ['Dropping samples with NaNs in necessary prep/sample '
                         'column(s): samp1',
                         'Dropping samples with negative values in necessary '
                         'prep/sample column(s): samp3']

        output_df, output_msgs = filter_data_by_sample_info(
            input_quant_params_per_sample_df, input_df, required_params)

        pandas.testing.assert_frame_equal(output_df, expected_df)
        self.assertListEqual(expected_msgs, output_msgs)

    def test_filter_data_by_sample_info_df_none_filtered(self):
        sample_ids = ['samp1', 'samp2']
        required_params = ['b', 'c', 'd']

        # make one of the UNneeded params NaN for second sample.
        # this should not cause any filtering in the df.
        params_dict = {
            'a': sample_ids,
            'b': [0.003, 0.00082],
            'c': [0.132714286, 0.0042],
            'd': [70, 70],
            'e': [213988, np.nan]
        }

        obs_ids = ["G000005825_1", "G000005825_2", "G000005825_3",
                   "G000005825_4", "G000005825_5", "G900163845_3247",
                   "G900163845_3248", "G900163845_3249",
                   "G900163845_3250", "G900163845_3251"]

        input_dict = {
            sample_ids[0]: [0, 2, 0, 35, 0, 10292, 0, 190, 0, 34],
            sample_ids[1]: [0, 0, 1, 0, 694, 382, 0, 10, 630, 1003]}

        input_quant_params_per_sample_df = pandas.DataFrame(
            params_dict, index=sample_ids)
        input_df = pandas.DataFrame(input_dict, index=obs_ids)

        output_df, output_msgs = filter_data_by_sample_info(
            input_quant_params_per_sample_df, input_df, required_params)

        pandas.testing.assert_frame_equal(output_df, input_df)
        self.assertListEqual([], output_msgs)

    def test_filter_data_by_sample_info_df_err(self):
        sample_ids = ['samp1', 'samp2', 'samp3']
        required_params = ['b', 'c', 'd']

        # make one of the needed params NaN for first sample.
        # make one of the needed params negative for third sample.
        # make one of the UNneeded params NaN for second sample.
        # the first and third should be filtered out of resulting biom table,
        # the second should remain.
        params_dict = {
            'a': sample_ids,
            'b': [np.nan, 0.00082, 0.45],
            'c': [0.132714286, 0.0042, -0.183],
            'd': [70, 70, 70],
            'e': [213988, np.nan, 3031038]
        }

        obs_ids = ["G000005825_1", "G000005825_2", "G000005825_3",
                   "G000005825_4", "G000005825_5", "G900163845_3247",
                   "G900163845_3248", "G900163845_3249",
                   "G900163845_3250", "G900163845_3251"]

        # if there are nan values in the input table that AREN'T caused by
        # NaNs in the quant params dataframe, an error should be raised
        input_dict = {
            sample_ids[0]: [0, 2, 0, 35, 0, 10292, 0, 190, 0, 34],
            sample_ids[1]: [0, 0, 1, 0, 694, 382, np.nan, 10, 630, 1003],
            sample_ids[2]: [0, 1, 0, 4, 29, 435, 0, 18, 30, 452]}

        input_quant_params_per_sample_df = pandas.DataFrame(
            params_dict, index=sample_ids)
        input_df = pandas.DataFrame(input_dict, index=obs_ids)

        expected_err = "There are NaNs remaining in the filtered table."

        with self.assertRaisesRegex(ValueError, expected_err):
            _ = filter_data_by_sample_info(
                input_quant_params_per_sample_df, input_df, required_params)

    def test_filter_data_by_sample_info_biom(self):
        sample_ids = ['samp1', 'samp2', 'samp3']
        required_params = ['b', 'c', 'd']

        # make one of the needed params NaN for first sample.
        # make one of the needed params negative for third sample.
        # make one of the UNneeded params NaN for second sample.
        # the first and third should be filtered out of resulting biom table,
        # the second should remain.
        params_dict = {
            'a': sample_ids,
            'b': [np.nan, 0.00082, 0.45],
            'c': [0.132714286, 0.0042, -0.183],
            'd': [70, 70, 70],
            'e': [213988, np.nan, 3031038]
        }

        obs_ids = ["G000005825_1", "G000005825_2", "G000005825_3",
                   "G000005825_4", "G000005825_5", "G900163845_3247",
                   "G900163845_3248", "G900163845_3249",
                   "G900163845_3250", "G900163845_3251"]

        input_vals = np.array([
            [0, 0, 0],
            [2, 0, 1],
            [0, 1, 0],
            [35, 0, 4],
            [0, 694, 29],
            [10292, 382, 435],
            [0, 0, 0],
            [190, 10, 18],
            [0, 630, 30],
            [34, 1003, 452]])

        remaining_count_vals = np.array([
            [0],
            [0],
            [1],
            [0],
            [694],
            [382],
            [0],
            [10],
            [630],
            [1003]])

        input_quant_params_per_sample_df = pandas.DataFrame(
            params_dict, index=sample_ids)
        input_biom = biom.table.Table(input_vals, obs_ids, sample_ids)

        # only one sample left after filtering
        expected_biom = biom.table.Table(remaining_count_vals, obs_ids,
                                         [sample_ids[1]])

        expected_msgs = ['Dropping samples with NaNs in necessary prep/sample '
                         'column(s): samp1',
                         'Dropping samples with negative values in necessary '
                         'prep/sample column(s): samp3']

        output_biom, output_msgs = filter_data_by_sample_info(
            input_quant_params_per_sample_df, input_biom, required_params)

        # NB: Comparing the bioms as dataframes because the biom equality
        # compare does not allow "almost equal" checking for float values,
        # whereas rtol and atol are built in to assert_frame_equal
        output_df = output_biom.to_dataframe()
        expected_df = expected_biom.to_dataframe()
        pandas.testing.assert_frame_equal(output_df, expected_df)

        self.assertListEqual(expected_msgs, output_msgs)

    def test_filter_data_by_sample_info_biom_err(self):
        sample_ids = ['samp1', 'samp2']
        required_params = ['b', 'c', 'd']

        # make one of the needed params NaN for first sample.
        # make one of the UNneeded params NaN for second sample.
        # the first one should be filtered out of the resulting biom table,
        # the second should remain.
        params_dict = {
            'a': sample_ids,
            'b': [np.nan, 0.00082],
            'c': [0.132714286, 0.0042],
            'd': [70, 70],
            'e': [213988, np.nan]
        }

        obs_ids = ["G000005825_1", "G000005825_2", "G000005825_3",
                   "G000005825_4", "G000005825_5", "G900163845_3247",
                   "G900163845_3248", "G900163845_3249",
                   "G900163845_3250", "G900163845_3251"]

        # if there are nan values in the input table that AREN'T caused by
        # NaNs in the quant params dataframe, an error should be raised
        input_vals = np.array([
            [0, 0],
            [2, 0],
            [0, 1],
            [35, 0],
            [0, 694],
            [10292, 382],
            [0, np.nan],
            [190, 10],
            [0, 630],
            [34, 1003]])

        expected_err = "There are NaNs remaining in the filtered table."

        input_quant_params_per_sample_df = pandas.DataFrame(
            params_dict, index=sample_ids)
        input_biom = biom.table.Table(input_vals, obs_ids, sample_ids)

        with self.assertRaisesRegex(ValueError, expected_err):
            _ = filter_data_by_sample_info(
                input_quant_params_per_sample_df, input_biom, required_params)

    def test_filter_data_by_sample_info_biom_none_filtered(self):
        sample_ids = ['samp1', 'samp2']
        required_params = ['b', 'c', 'd']

        # make one of the UNneeded params NaN for second sample.
        # this should not cause any filtering in the biom.
        params_dict = {
            'a': sample_ids,
            'b': [0.003, 0.00082],
            'c': [0.132714286, 0.0042],
            'd': [70, 70],
            'e': [213988, np.nan]
        }

        obs_ids = ["G000005825_1", "G000005825_2", "G000005825_3",
                   "G000005825_4", "G000005825_5", "G900163845_3247",
                   "G900163845_3248", "G900163845_3249",
                   "G900163845_3250", "G900163845_3251"]

        input_vals = np.array([
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

        input_quant_params_per_sample_df = pandas.DataFrame(
            params_dict, index=sample_ids)
        input_biom = biom.table.Table(input_vals, obs_ids, sample_ids)

        output_biom, output_msgs = filter_data_by_sample_info(
            input_quant_params_per_sample_df, input_biom, required_params)

        # NB: Comparing the bioms as dataframes because the biom equality
        # compare does not allow "almost equal" checking for float values,
        # whereas rtol and atol are built in to assert_frame_equal
        output_df = output_biom.to_dataframe()
        expected_df = input_biom.to_dataframe()
        pandas.testing.assert_frame_equal(output_df, expected_df)

        self.assertListEqual([], output_msgs)

    def test_calc_copies_genomic_element_per_g_series(self):
        # example from "rna_copy_quant_example.xlsx" "full_calc" tab,
        # ogu_orf_calculations table
        elements_lens = [1353, 1143, 216, 1116, 276, 1797, 846, 891, 768, 645]
        copies_per_g = [1.309104e+18, 1.549622e+18, 8.200083e+18, 1.587113e+18,
                        6.417456e+18, 9.856527e+17, 2.093638e+18, 1.987899e+18,
                        2.306273e+18, 2.746074e+18]
        expected_series = pandas.Series(copies_per_g)
        obs_series = calc_copies_genomic_element_per_g_series(
            pandas.Series(elements_lens), 340)
        assert_series_equal(expected_series, obs_series)

    def test_calc_gs_genomic_element_in_aliquot(self):
        # example from "rna_copy_quant_example.xlsx" "full_calc" tab,
        # quant_params_per_sample table
        input_dict = {
            SAMPLE_ID_KEY: ["IBSRS3526007", "IQSRS3526010"],
            'conc_ng_ul': [0.132714, 0.004200],
            ELUTE_VOL_UL_KEY: [70, 70]
        }

        added_dict = {'mass_key': [9.290000e-09, 2.940000e-10]}
        expected_dict = input_dict.copy()
        expected_dict.update(added_dict)

        input_df = pandas.DataFrame(input_dict)
        expected_df = pandas.DataFrame(expected_dict)

        obs_df = calc_gs_genomic_element_in_aliquot(
            input_df, 'conc_ng_ul', 'mass_key')
        assert_frame_equal(expected_df, obs_df)
