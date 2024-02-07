import biom
import numpy as np
import pandas
from pandas.testing import assert_series_equal, assert_frame_equal
from unittest import TestCase
from pysyndna.src.util import calc_copies_genomic_element_per_g_series, \
    calc_gs_genomic_element_in_aliquot, \
    validate_metadata_vs_prep_id_consistency, \
    validate_metadata_vs_reads_id_consistency, \
    validate_required_columns_exist, SAMPLE_ID_KEY, ELUTE_VOL_UL_KEY


class TestCalcCellCounts(TestCase):
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

        _ = validate_metadata_vs_prep_id_consistency(input_df, prep_df)

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

        not_in_prep_ids = validate_metadata_vs_prep_id_consistency(
            input_df, prep_df)

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
            _ = validate_metadata_vs_prep_id_consistency(
                input_df, prep_df)

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

        _ = validate_metadata_vs_reads_id_consistency(input_df, reads_df)

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

        not_in_prep_ids = validate_metadata_vs_reads_id_consistency(
            input_df, reads_df)

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
            _ = validate_metadata_vs_reads_id_consistency(
                input_df, reads_df)

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

        _ = validate_metadata_vs_reads_id_consistency(input_df, reads_biom)

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

        not_in_prep_ids = validate_metadata_vs_reads_id_consistency(
            input_df, reads_biom)

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
            _ = validate_metadata_vs_reads_id_consistency(
                input_df, reads_biom)

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
