import biom.table
import copy
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import os
from collections import namedtuple
from unittest import TestCase
from pysyndna import fit_linear_regression_models, \
    fit_linear_regression_models_for_qiita
from pysyndna.src.fit_syndna_models import SAMPLE_ID_KEY, SYNDNA_ID_KEY, \
    INPUT_SYNDNA_POOL_MASS_NG_KEY, \
    SYNDNA_POOL_MASS_NG_KEY, SYNDNA_INDIV_NG_UL_KEY, \
    SYNDNA_FRACTION_OF_POOL_KEY, SYNDNA_INDIV_NG_KEY, \
    SAMPLE_TOTAL_READS_KEY, SYNDNA_POOL_NUM_KEY, \
    _validate_syndna_id_consistency, _validate_sample_id_consistency, \
    _calc_indiv_syndna_weights, _fit_linear_regression_models, \
    _convert_linregressresults_to_dict
from pysyndna.tests.test_util import Testers


class FitSyndnaModelsTestData():
    syndna_concs_dict = {
        SYNDNA_ID_KEY: ["p126", "p136", "p146", "p156", "p166", "p226",
                        "p236", "p246", "p256", "p266"],
        SYNDNA_INDIV_NG_UL_KEY: [1, 0.1, 0.01, 0.001, 0.0001, 0.0001,
                                 0.001, 0.01, 0.1, 1],
    }

    # made-up placeholders :)
    sample_ids = ["A", "B"]

    # Total reads come from the "TotalReads" column of
    # https://github.com/lzaramela/SynDNA/blob/main/data/synDNA_metadata_updated.tsv
    # for the record with "ID" = "A1_pool1_Fwd".
    # Syndna pool mass is the default value expected in our experimental
    # system.
    a_sample_syndna_weights_and_total_reads_dict = {
        SAMPLE_ID_KEY: [sample_ids[0]],
        INPUT_SYNDNA_POOL_MASS_NG_KEY: [0.25],
    }

    # Total reads come from the "TotalReads" column of
    # https://github.com/lzaramela/SynDNA/blob/main/data/synDNA_metadata_updated.tsv
    # for the record with "ID" of "A1_pool1_Fwd" and "C1_pool1_Fwd".
    # Syndna pool masses are plausible values for our experimental system.
    a_b_sample_syndna_weights_and_total_reads_dict = {
        SAMPLE_ID_KEY: sample_ids,
        INPUT_SYNDNA_POOL_MASS_NG_KEY: [0.25, 0.2],
    }

    # same as a_b_sample_syndna_weights_and_total_reads_dict, but
    # the INPUT_SYNDNA_POOL_MASS_NG_KEY values are scaled up by 2.5 so that,
    # when multiplied by a syndna_contributing_fraction of 0.4, they
    # give the same masses as in a_b_sample_syndna_weights_and_total_reads_dict
    scaled_a_b_sample_syndna_weights_and_total_reads_dict = {
        SAMPLE_ID_KEY: sample_ids,
        INPUT_SYNDNA_POOL_MASS_NG_KEY: [0.625, 0.5],
    }

    # Total reads come from the "TotalReads" column of
    # https://github.com/lzaramela/SynDNA/blob/main/data/synDNA_metadata_updated.tsv
    # for the record with "ID" of "A1_pool1_Fwd", "C1_pool1_Fwd", and
    # "D1_pool1_Fwd".
    # Syndna pool masses are plausible values for our experimental system.
    a_b_c_sample_syndna_weights_and_total_reads_dict = {
        SAMPLE_ID_KEY: [sample_ids[0], sample_ids[1], "C"],
        # NOTE: total reads are no longer used in the code since we switched
        # from fitting CPMs to fitting raw counts, but this is left here to
        # demonstrate that the code is able to ignore extra info in the input
        # dataframes.
        SAMPLE_TOTAL_READS_KEY: [3216923, 1723417, 2606004],
        INPUT_SYNDNA_POOL_MASS_NG_KEY: [0.25, 0.2, 0.3],
    }

    # The below sample values come from the
    # "A1_pool1_S21_L001_R1_001.fastq_output_forward_paired
    # .fq.sam.bam.f13_r1.fq_synDNA" and
    # "A1_pool2_S22_L001_R1_001.fastq_output_forward_paired
    # .fq.sam.bam.f13_r1.fq_synDNA" columns of
    # https://github.com/lzaramela/SynDNA/blob/main/data/synDNA_Fwd_Rev_sam.biom.tsv ,
    # while the syndna ids are inferred from the contents of the "OTUID"
    # column and a knowledge of the Zaramela naming scheme.
    reads_per_syndna_per_sample_dict = {
        SYNDNA_ID_KEY: ["p126", "p136", "p146", "p156", "p166", "p226",
                        "p236", "p246", "p256", "p266"],
        sample_ids[0]: [93135, 15190, 2447, 308, 77, 149, 1075, 3189, 25347, 237329],
        sample_ids[1]: [90897, 15002, 2421, 296, 77, 148, 1059, 3129, 24856, 230898],
    }

    # The slope, intercept, rvalue, stderr (of slope),
    # intercept_stderr, and p-value (of slope) values for these results
    # match those calculated in Excel (see results for full data
    # on "linear regressions counts" sheet of "absolute_quant_example.xlsx").
    # Note that these do not and *should* NOT be expected to match any results
    # in Zaramela's linear models because:
    # (i) we are fitting log10(raw counts) not log10(CPM).  Note this comes out
    # in the wash when one actually *uses* the models to predict mass:
    # you get the same masses using the model fit to log10(raw counts)
    # as you do using the model fit to log10(CPM) (which makes sense since all
    # the CPM calculation does is divide the raw counts by a constant
    # [total reads] and then multiply by another constant [a million]) but
    # the models themselves are slightly different--same slope but different
    # intercepts.
    # (ii) sample A has the same counts as Zaramela's sample
    # "A1_pool1_Fwd" *but* we use a different pool mass than Zaramela,
    # so the same syndna counts are based on different masses.
    # (iii) sample B is a chimera of realistic data from multiple Zaramela
    # samples but isn't directly comparable to a single one of them
    lingress_results = {
        'A': {
            "slope": 1.24487652379132,
            "intercept": -7.35593916054843,
            "rvalue": 0.986503097515657,
            "pvalue": 1.42844356065977E-07,
            "stderr": 0.0730540855033502,
            "intercept_stderr": 0.271274537363401},
        'B': {
            "slope": 1.24675913604407,
            "intercept": -7.45004083037736,
            "rvalue": 0.986324179735633,
            "pvalue": 1.5053811468097E-07,
            "stderr": 0.073657952553024,
            "intercept_stderr": 0.2729411999326}
    }

    prep_info_dict = copy.deepcopy(
        a_b_sample_syndna_weights_and_total_reads_dict)
    prep_info_dict["sequencing_type"] = ["shotgun", "shotgun"]
    prep_info_dict["syndna_pool_number"] = [0, 0]

    scaled_prep_info_dict = copy.deepcopy(
        scaled_a_b_sample_syndna_weights_and_total_reads_dict)
    scaled_prep_info_dict["sequencing_type"] = ["shotgun", "shotgun"]
    # NB: in the alt config, syndna pool number 1 has
    # syndna_contributing_fraction of 0.4
    scaled_prep_info_dict["syndna_pool_number"] = [1, 1]

    # combine each item in self.reads_per_syndna_per_sample_dict["A"] with
    # the analogous item in self.reads_per_syndna_per_sample_dict["B"]
    # to make an array of two-item arrays, and turn this into an np.array
    reads_per_syndna_per_sample_array = np.array(
        [list(x) for x in zip(
            reads_per_syndna_per_sample_dict["A"],
            reads_per_syndna_per_sample_dict["B"])])


class FitSyndnaModelsTest(TestCase):
    def setUp(self):
        self.maxDiff = None
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.alt_config_fp = os.path.join(self.data_dir, 'alt_config.yml')

    def test_fit_linear_regression_models_for_qiita(self):
        prep_info_df = pd.DataFrame(FitSyndnaModelsTestData.prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = 50

        # These are text versions of the linear regression results
        # for the full data (see self.lingress_results and the
        # "linear regressions counts" sheet of "absolute_quant_example.xlsx").
        expected_out = {
            'lin_regress_by_sample_id':
                'A:\n'
                '  intercept: -7.355939160548\n'
                '  intercept_stderr: 0.271274537363\n'
                '  pvalue: 1.42844e-07\n'
                '  rvalue: 0.986503097515\n'
                '  slope: 1.244876523791\n'
                '  stderr: 0.073054085503\n'
                'B:\n'
                '  intercept: -7.450040830377\n'
                '  intercept_stderr: 0.272941199932\n'
                '  pvalue: 1.50538e-07\n'
                '  rvalue: 0.986324179735\n'
                '  slope: 1.246759136044\n'
                '  stderr: 0.073657952553\n',
            'fit_syndna_models_log': ''
        }

        output_dict = fit_linear_regression_models_for_qiita(
            prep_info_df, input_biom, min_counts)

        self.assertDictEqual(expected_out, output_dict)

    def test_fit_linear_regression_models_for_qiita_w_scaling(self):
        scaled_prep_info_df = pd.DataFrame(
            FitSyndnaModelsTestData.scaled_prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = 50

        # These are text versions of the linear regression results
        # for the full data (see self.lingress_results and the
        # "linear regressions counts" sheet of "absolute_quant_example.xlsx").
        expected_out = {
            'lin_regress_by_sample_id':
                'A:\n'
                '  intercept: -7.355939160548\n'
                '  intercept_stderr: 0.271274537363\n'
                '  pvalue: 1.42844e-07\n'
                '  rvalue: 0.986503097515\n'
                '  slope: 1.244876523791\n'
                '  stderr: 0.073054085503\n'
                'B:\n'
                '  intercept: -7.450040830377\n'
                '  intercept_stderr: 0.272941199932\n'
                '  pvalue: 1.50538e-07\n'
                '  rvalue: 0.986324179735\n'
                '  slope: 1.246759136044\n'
                '  stderr: 0.073657952553\n',
            'fit_syndna_models_log': ''
        }

        # NB: these test data require a different syndna_contributing_fraction
        # than that in the real config, so must pass in the alt config file
        output_dict = fit_linear_regression_models_for_qiita(
            scaled_prep_info_df, input_biom, min_counts,
            syndna_pool_config_fp=self.alt_config_fp)

        self.assertDictEqual(expected_out, output_dict)

    def test_fit_linear_regression_models_for_qiita_w_casts(self):
        # same as test_fit_linear_regression_models_for_qiita, but with
        # all param values passed in as strings
        prep_info_dict = {
            k: [str(x) for x in
                FitSyndnaModelsTestData.scaled_prep_info_dict[k]]
            for k in FitSyndnaModelsTestData.prep_info_dict}
        prep_info_df = pd.DataFrame(prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = "50"

        # These are text versions of the linear regression results
        # for the full data (see self.lingress_results and the
        # "linear regressions counts" sheet of "absolute_quant_example.xlsx").
        expected_out = {
            'lin_regress_by_sample_id':
                'A:\n'
                '  intercept: -7.355939160548\n'
                '  intercept_stderr: 0.271274537363\n'
                '  pvalue: 1.42844e-07\n'
                '  rvalue: 0.986503097515\n'
                '  slope: 1.244876523791\n'
                '  stderr: 0.073054085503\n'
                'B:\n'
                '  intercept: -7.450040830377\n'
                '  intercept_stderr: 0.272941199932\n'
                '  pvalue: 1.50538e-07\n'
                '  rvalue: 0.986324179735\n'
                '  slope: 1.246759136044\n'
                '  stderr: 0.073657952553\n',
            'fit_syndna_models_log': ''
        }

        # NB: these test data require a different syndna_contributing_fraction
        # than that in the real config, so must pass in the alt config file
        output_dict = fit_linear_regression_models_for_qiita(
            prep_info_df, input_biom, min_counts,
            syndna_pool_config_fp=self.alt_config_fp)

        self.assertDictEqual(expected_out, output_dict)

    def test_fit_linear_regression_models_for_qiita_w_alt_config(self):
        prep_info_df = pd.DataFrame(FitSyndnaModelsTestData.prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = 50

        # these are the linear regression results for running the code
        # using completely different (and spurious) syndna concentrations
        # represented in an alternate config file. Don't use these results
        # for anything else!
        expected_out = {
            'lin_regress_by_sample_id':
                'A:\n'
                '  intercept: -9.005671912625\n'
                '  intercept_stderr: 0.624713705225\n'
                '  pvalue: 1.287067e-05\n'
                '  rvalue: 0.958056670088\n'
                '  slope: 1.590774502959\n'
                '  stderr: 0.168235061352\n'
                'B:\n'
                '  intercept: -9.100255596079\n'
                '  intercept_stderr: 0.624155431954\n'
                '  pvalue: 1.2820953e-05\n'
                '  rvalue: 0.958097757898\n'
                '  slope: 1.593537551784\n'
                '  stderr: 0.168439250666\n',
            'fit_syndna_models_log': ''
        }

        output_dict = fit_linear_regression_models_for_qiita(
            prep_info_df, input_biom, min_counts, self.alt_config_fp)

        self.assertDictEqual(expected_out, output_dict)

    def test_fit_linear_regression_models_for_qiita_w_negs(self):
        prep_info_dict = copy.deepcopy(FitSyndnaModelsTestData.prep_info_dict)
        prep_info_dict[INPUT_SYNDNA_POOL_MASS_NG_KEY][0] = \
            -1 * prep_info_dict[INPUT_SYNDNA_POOL_MASS_NG_KEY][0]
        prep_info_df = pd.DataFrame(prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = 50

        # These are text versions of the linear regression results
        # for the full data (see self.lingress_results and the full data fit on
        # "linear regressions counts" sheet of "absolute_quant_example.xlsx").
        expected_out = {
            'lin_regress_by_sample_id':
                'B:\n'
                '  intercept: -7.450040830377\n'
                '  intercept_stderr: 0.272941199932\n'
                '  pvalue: 1.50538e-07\n'
                '  rvalue: 0.986324179735\n'
                '  slope: 1.246759136044\n'
                '  stderr: 0.073657952553\n',
            'fit_syndna_models_log':
                'Dropping samples with negative values in necessary '
                'prep/sample column(s): A'
        }

        output_dict = fit_linear_regression_models_for_qiita(
            prep_info_df, input_biom, min_counts)

        self.assertDictEqual(expected_out, output_dict)

    def test_fit_linear_regression_models_for_qiita_w_log_msgs(self):
        prep_info_df = pd.DataFrame(FitSyndnaModelsTestData.prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = 200

        # These are text versions of the linear regression results
        # for the data with syndnas with <200 total counts removed (see
        # "linear regressions counts" sheet of "absolute_quant_example.xlsx").
        expected_out = {
            'lin_regress_by_sample_id':
                'A:\n'
                '  intercept: -7.404604502655\n'
                '  intercept_stderr: 0.34490591545\n'
                '  pvalue: 2.170514e-06\n'
                '  rvalue: 0.982777689569\n'
                '  slope: 1.256194910944\n'
                '  stderr: 0.089276147107\n'
                'B:\n'
                '  intercept: -7.49322862874\n'
                '  intercept_stderr: 0.347041014613\n'
                '  pvalue: 2.289073e-06\n'
                '  rvalue: 0.982512701026\n'
                '  slope: 1.25681918648\n'
                '  stderr: 0.090023307568\n',
            'fit_syndna_models_log':
                "The following syndnas were dropped because they had fewer "
                "than 200 total reads aligned:['p166']"
        }

        output_dict = fit_linear_regression_models_for_qiita(
            prep_info_df, input_biom, min_counts)

        self.assertDictEqual(expected_out, output_dict)

    def test_fit_linear_regression_models_for_qiita_w_pool_error(self):
        prep_info_dict = copy.deepcopy(FitSyndnaModelsTestData.prep_info_dict)
        prep_info_dict[SYNDNA_POOL_NUM_KEY] = [1, 2]
        prep_info_df = pd.DataFrame(prep_info_dict)

        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = 50

        # NB: the error message is a regex, so we need to escape the brackets
        expected_err_msg = \
            r"Multiple syndna_pool_numbers found in prep info: \[1 2\]"

        with self.assertRaisesRegex(ValueError, expected_err_msg):
            fit_linear_regression_models_for_qiita(
                prep_info_df, input_biom, min_counts)

    def test_fit_linear_regression_models_for_qiita_w_col_error(self):
        prep_info_dict = {
            SAMPLE_ID_KEY: ["A", "B"],
            "sequencing_type": ["shotgun", "shotgun"],
            SAMPLE_TOTAL_READS_KEY: [3216923, 1723417],
            INPUT_SYNDNA_POOL_MASS_NG_KEY: [0.25, 0.2],
            # missing the SYNDNA_POOL_NUM_KEY column
        }
        prep_info_df = pd.DataFrame(prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)
        min_counts = 50

        # NB: the error message is a regex, so we need to escape the brackets
        expected_err_msg = \
            r"prep info is missing required column\(s\): " \
            r"\['syndna_pool_number'\]"

        with self.assertRaisesRegex(ValueError, expected_err_msg):
            fit_linear_regression_models_for_qiita(
                prep_info_df, input_biom, min_counts)

    def test_fit_linear_regression_models_for_qiita_w_min_counts_err(self):
        prep_info_df = pd.DataFrame(FitSyndnaModelsTestData.prep_info_dict)
        input_biom = biom.table.Table(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_array,
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict[SYNDNA_ID_KEY],
            FitSyndnaModelsTestData.sample_ids)

        min_counts = 0
        err_msg = "min_sample_counts must be a positive integer but received 0"
        with self.assertRaisesRegex(ValueError, err_msg):
            fit_linear_regression_models_for_qiita(
                prep_info_df, input_biom, min_counts)

    def test_fit_linear_regression_models(self):
        min_count = 50
        syndna_contributing_fraction = 1

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_b_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        out_linregress_dict, out_msgs = fit_linear_regression_models(
            syndna_concs_df,
            sample_syndna_weights_and_total_reads_df,
            reads_per_syndna_per_sample_df, min_count,
            syndna_contributing_fraction)

        a_tester = Testers()
        a_tester.assert_dicts_almost_equal(
            FitSyndnaModelsTestData.lingress_results, out_linregress_dict)
        a_tester.assertEqual([], out_msgs)

    def test_fit_linear_regression_models_scaled(self):
        min_count = 50
        syndna_contributing_fraction = 0.4

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.scaled_a_b_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        out_linregress_dict, out_msgs = fit_linear_regression_models(
            syndna_concs_df,
            sample_syndna_weights_and_total_reads_df,
            reads_per_syndna_per_sample_df, min_count,
            syndna_contributing_fraction)

        a_tester = Testers()
        a_tester.assert_dicts_almost_equal(
            FitSyndnaModelsTestData.lingress_results, out_linregress_dict)
        a_tester.assertEqual([], out_msgs)

    def test_fit_linear_regression_models_w_casts(self):
        min_count = "50"
        syndna_contributing_fraction = "1"

        # same as test_fit_linear_regression_models, but with
        # all param values passed in as strings
        syndna_concs_dict = {
            k: [str(x) for x in
                FitSyndnaModelsTestData.syndna_concs_dict[k]]
            for k in FitSyndnaModelsTestData.syndna_concs_dict}
        syndna_concs_df = pd.DataFrame(syndna_concs_dict)
        a_b_reads_dict = \
            FitSyndnaModelsTestData \
            .a_b_sample_syndna_weights_and_total_reads_dict
        a_b_sample_syndna_weights_and_total_reads_dict = {
            k: [str(x) for x in a_b_reads_dict[k]]
            for k in a_b_reads_dict}
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            a_b_sample_syndna_weights_and_total_reads_dict)

        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        out_linregress_dict, out_msgs = fit_linear_regression_models(
            syndna_concs_df,
            sample_syndna_weights_and_total_reads_df,
            reads_per_syndna_per_sample_df, min_count,
            syndna_contributing_fraction)

        a_tester = Testers()
        a_tester.assert_dicts_almost_equal(
            FitSyndnaModelsTestData.lingress_results, out_linregress_dict)
        self.assertEqual([], out_msgs)

    def test_fit_linear_regression_models_w_log_msgs(self):
        min_count = 200
        syndna_contributing_fraction = 1

        # The slope, intercept, rvalue, stderr (of slope),
        # intercept_stderr, and p-value (of slope) values for these results
        # match those calculated in Excel (see results for data with
        # syndnas with <200 total counts removed on "linear regressions counts"
        # sheet of "absolute_quant_example.xlsx").
        expected_out_dict = {
            'A': {
                "slope": 1.25619491094468,
                "intercept": -7.40460450265582,
                "rvalue": 0.982777689569875,
                "pvalue": 2.17051437085363E-06,
                "stderr": 0.0892761471071481,
                "intercept_stderr": 0.344905915450109},
            'B': {
                "slope": 1.2568191864802,
                "intercept": -7.49322862874098,
                "rvalue": 0.982512701026673,
                "pvalue": 2.28907333341599E-06,
                "stderr": 0.0900233075686737,
                "intercept_stderr": 0.347041014613366}
        }
        expected_out_msgs = [
            "The following sample ids were in the experiment info but not in "
            "the data: ['C']",
            "The following syndnas were dropped because they had fewer than "
            "200 total reads aligned:['p166']"
        ]

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_b_c_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        out_linregress_dict, out_msgs = fit_linear_regression_models(
            syndna_concs_df,
            sample_syndna_weights_and_total_reads_df,
            reads_per_syndna_per_sample_df, min_count,
            syndna_contributing_fraction)

        a_tester = Testers()
        a_tester.assert_dicts_almost_equal(
            expected_out_dict, out_linregress_dict)
        self.assertEqual(expected_out_msgs, out_msgs)

    def test_fit_linear_regression_models_w_sample_error(self):
        min_count = 200
        syndna_contributing_fraction = 1

        # use self.a_sample_syndna_weights_and_total_reads_dict,
        # which includes only info for sample A, not for sample B,
        # which is in the data

        expected_err_msg = \
            (r"Found sample ids in reads data that were not in sample info: "
             r"\{'B'\}")

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        with self.assertRaisesRegex(ValueError, expected_err_msg):
            fit_linear_regression_models(
                syndna_concs_df,
                sample_syndna_weights_and_total_reads_df,
                reads_per_syndna_per_sample_df, min_count,
                syndna_contributing_fraction)

    def test_fit_linear_regression_models_w_syndna_config_error(self):
        # syndnas in the data that aren't in the config

        # These data are the same as those in
        # self.reads_per_syndna_per_sample_dict EXCEPT for the deletion of
        # elements for syndna id "p266".
        syndna_concs_dict = {
            SYNDNA_ID_KEY: ["p126", "p136", "p146", "p156", "p166", "p226",
                            "p236", "p246", "p256",],
            SYNDNA_INDIV_NG_UL_KEY: [1, 0.1, 0.01, 0.001, 0.0001, 0.0001,
                                     0.001, 0.01, 0.1],
        }

        min_count = 200
        syndna_contributing_fraction = 1

        expected_err_msg = \
            r"Detected 1 syndna feature\(s\) in the read data that " \
            r"were not in the config: {'p266'}"

        syndna_concs_df = pd.DataFrame(syndna_concs_dict)
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_b_c_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        with self.assertRaisesRegex(ValueError, expected_err_msg):
            fit_linear_regression_models(
                syndna_concs_df,
                sample_syndna_weights_and_total_reads_df,
                reads_per_syndna_per_sample_df, min_count,
                syndna_contributing_fraction)

    def test_fit_linear_regression_models_w_syndna_data_error(self):
        # syndnas in the config that aren't in the data

        # These data are the same as those in
        # self.reads_per_syndna_per_sample_dict EXCEPT for the deletion of
        # elements for syndna id "p266".
        reads_per_syndna_per_sample_dict = {
            SYNDNA_ID_KEY: ["p126", "p136", "p146", "p156", "p166", "p226",
                            "p236", "p246", "p256"],
            "A": [93135, 15190, 2447, 308, 77, 149, 1075, 3189, 25347],
            "B": [90897, 15002, 2421, 296, 77, 148, 1059, 3129, 24856],
        }

        min_count = 200
        syndna_contributing_fraction = 1

        expected_err_msg = \
            r"Missing the following 1 required syndna feature\(s\) in the " \
            r"read data: \{'p266'\}"

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_b_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        with self.assertRaisesRegex(ValueError, expected_err_msg):
            fit_linear_regression_models(
                syndna_concs_df,
                sample_syndna_weights_and_total_reads_df,
                reads_per_syndna_per_sample_df, min_count,
                syndna_contributing_fraction)

    def test_fit_linear_regression_models_w_syndna_contrib_frac_err(self):
        min_count = 50

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_b_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)
        reads_per_syndna_per_sample_df.set_index(SYNDNA_ID_KEY, inplace=True)

        syndna_contributing_fraction = 2
        err_msg = ("syndna_contributing_fraction must be a float >0 and <= 1 "
                   "but received 2.0")
        with self.assertRaisesRegex(ValueError, err_msg):
            _ = fit_linear_regression_models(
                syndna_concs_df,
                sample_syndna_weights_and_total_reads_df,
                reads_per_syndna_per_sample_df, min_count,
                syndna_contributing_fraction)

    def test__validate_syndna_id_consistency(self):
        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)

        try:
            _validate_syndna_id_consistency(
                syndna_concs_df, reads_per_syndna_per_sample_df)
        except ValueError:
            self.fail("Raised ValueError incorrectly")

    def test__validate_syndna_id_consistency_w_error_missing_data(self):
        # These data are the same as those in
        # self.reads_per_syndna_per_sample_dict EXCEPT for the deletion of
        # elements for syndna id "p266".
        reads_per_syndna_per_sample_dict = {
            SYNDNA_ID_KEY: ["p126", "p136", "p146", "p156", "p166", "p226",
                            "p236", "p246", "p256"],
            "A": [93135, 15190, 2447, 308, 77, 149, 1075, 3189, 25347],
            "B": [90897, 15002, 2421, 296, 77, 148, 1059, 3129, 24856],
        }

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            reads_per_syndna_per_sample_dict)

        # NB: the error message is a regex, so we need to escape the brackets
        err_msg = \
            r"Missing the following 1 required syndna feature\(s\) " \
            r"in the read data: \{'p266'\}"
        with self.assertRaisesRegex(ValueError, err_msg):
            _validate_syndna_id_consistency(
                syndna_concs_df,
                reads_per_syndna_per_sample_df)

    def test__validate_syndna_id_consistency_w_error_missing_info(self):
        # These data are the same as those in
        # self.syndna_concs_dict EXCEPT for the deletion of
        # elements for syndna id "p266".
        syndna_concs_dict = {
            SYNDNA_ID_KEY: ["p126", "p136", "p146", "p156", "p166", "p226",
                            "p236", "p246", "p256"],
            SYNDNA_INDIV_NG_UL_KEY: [1, 0.1, 0.01, 0.001, 0.0001, 0.0001,
                                     0.001, 0.01, 0.1],
        }

        syndna_concs_df = pd.DataFrame(syndna_concs_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)

        # NB: the error message is a regex, so we need to escape the brackets
        err_msg = r"Detected 1 syndna feature\(s\) in the read data " \
                  r"that were not in the config: \{'p266'\}"
        with self.assertRaisesRegex(ValueError, err_msg):
            _validate_syndna_id_consistency(
                syndna_concs_df,
                reads_per_syndna_per_sample_df)

    def test__validate_sample_id_consistency(self):
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_b_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)

        try:
            output = _validate_sample_id_consistency(
                sample_syndna_weights_and_total_reads_df,
                reads_per_syndna_per_sample_df)
        except ValueError:
            self.fail("Raised ValueError incorrectly")

        # all samples are in both, so no extras reported
        self.assertIsNone(output)

    def test__validate_sample_id_consistency_w_output(self):
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_b_c_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)

        try:
            output = _validate_sample_id_consistency(
                sample_syndna_weights_and_total_reads_df,
                reads_per_syndna_per_sample_df)
        except ValueError:
            self.fail("Raised ValueError incorrectly")

        # sample C is in sample info but not in sequencing results;
        # for example, maybe it failed to sequence
        self.assertEqual(["C"], output)

    def test__validate_sample_id_consistency_w_error(self):
        sample_syndna_weights_and_total_reads_df = pd.DataFrame(
            FitSyndnaModelsTestData.a_sample_syndna_weights_and_total_reads_dict)
        reads_per_syndna_per_sample_df = pd.DataFrame(
            FitSyndnaModelsTestData.reads_per_syndna_per_sample_dict)

        err_msg = (r"Found sample ids in reads data that were not in sample "
                   r"info: \{'B'\}")
        with self.assertRaisesRegex(ValueError, err_msg):
            _validate_sample_id_consistency(
                sample_syndna_weights_and_total_reads_df,
                reads_per_syndna_per_sample_df)

    def test__calc_indiv_syndna_weights(self):
        # The below values were made up using sample and syndna ids from the
        # Zaramela data and masses (only two, one for each sample--they are
        # just repeated for each syndna) that are realistic for our
        # experimental system.
        working_dict = {
            SAMPLE_ID_KEY: ["A1_pool1_Fwd", "A1_pool1_Rev", "A1_pool1_Fwd",
                            "A1_pool1_Rev", "A1_pool1_Fwd", "A1_pool1_Rev",
                            "A1_pool1_Fwd", "A1_pool1_Rev", "A1_pool1_Fwd",
                            "A1_pool1_Rev", "A1_pool1_Fwd", "A1_pool1_Rev",
                            "A1_pool1_Fwd", "A1_pool1_Rev", "A1_pool1_Fwd",
                            "A1_pool1_Rev", "A1_pool1_Fwd", "A1_pool1_Rev",
                            "A1_pool1_Fwd", "A1_pool1_Rev"],
            SYNDNA_ID_KEY: ["p126", "p126", "p136", "p136", "p146", "p146",
                            "p156", "p156", "p166", "p166", "p226", "p226",
                            "p236", "p236", "p246", "p246", "p256", "p256",
                            "p266", "p266"],
            SYNDNA_POOL_MASS_NG_KEY: [0.25, 0.2, 0.25, 0.2, 0.25, 0.2, 0.25,
                                      0.2, 0.25, 0.2, 0.25, 0.2, 0.25, 0.2,
                                      0.25, 0.2, 0.25, 0.2, 0.25, 0.2]
        }

        # These test values come from the "linear regressions" tab of the
        # absolute_quant_example.xlsx file.
        expected_addl_dict = {
            SYNDNA_INDIV_NG_UL_KEY: [1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001,
                                     0.0001, 0.0001, 0.0001, 0.0001, 0.001,
                                     0.001, 0.01, 0.01, 0.1, 0.1, 1, 1],
            # NB: the below values are perforce rounded since the division
            # produces a repeating decimal, e.g. 0.45000450004500045 ...)
            SYNDNA_FRACTION_OF_POOL_KEY: [0.4500045, 0.4500045, 0.04500045,
                                          0.04500045, 0.004500045, 0.004500045,
                                          0.000450005, 0.000450005,
                                          4.50005E-05, 4.50005E-05,
                                          4.50005E-05, 4.50005E-05,
                                          0.000450005, 0.000450005,
                                          0.004500045, 0.004500045,
                                          0.04500045, 0.04500045, 0.4500045,
                                          0.4500045],
            SYNDNA_INDIV_NG_KEY: [0.112501125, 0.0900009, 0.011250113,
                                  0.00900009, 0.001125011, 0.000900009,
                                  0.000112501, 9.00009E-05, 1.12501E-05,
                                  9.00009E-06, 1.12501E-05, 9.00009E-06,
                                  0.000112501, 9.00009E-05, 0.001125011,
                                  0.000900009, 0.011250113, 0.00900009,
                                  0.112501125, 0.0900009]
        }

        syndna_concs_df = pd.DataFrame(FitSyndnaModelsTestData.syndna_concs_dict)
        working_df = pd.DataFrame(working_dict)

        output_df = _calc_indiv_syndna_weights(syndna_concs_df, working_df)

        expected_dict = working_dict | expected_addl_dict
        expected_df = pd.DataFrame(expected_dict)
        assert_frame_equal(expected_df, output_df)

    def test__fit_linear_regression_models(self):
        # See input and output files for descriptions of the test data
        # provenance; broadly, they are taken from the example notebook
        # and results file at https://github.com/lzaramela/SynDNA/ .
        input_fp = os.path.join(self.data_dir, 'modelling_input.tsv')
        working_df = pd.read_csv(input_fp, sep="\t", comment="#")

        output, out_msgs_list = _fit_linear_regression_models(working_df)

        expected_fp = os.path.join(self.data_dir, 'modelling_output.tsv')
        expected_df = pd.read_csv(expected_fp, sep="\t", comment="#")

        for k, v in output.items():
            # the LingressResult object isn't really exposed via the scipy
            # api, so resorting to duck testing
            self.assertTrue(hasattr(v, "slope"))
            self.assertTrue(hasattr(v, "intercept"))
            self.assertTrue(hasattr(v, "rvalue"))
            self.assertTrue(hasattr(v, "pvalue"))
            self.assertTrue(hasattr(v, "stderr"))
            self.assertTrue(hasattr(v, "intercept_stderr"))

            item_mask = expected_df["ID"] == k
            expected_slope = expected_df.loc[item_mask, "b_slope"].iloc[0]
            expected_intercept = expected_df.loc[
                item_mask, "a_intercept"].iloc[0]
            self.assertAlmostEqual(expected_slope, v.slope)
            self.assertAlmostEqual(expected_intercept, v.intercept)
        # next model

        self.assertEqual([], out_msgs_list)

    def test__convert_linregressresults_to_dict(self):
        linregress_mock = namedtuple(
            "linregress_mock",
            ["slope", "intercept", "rvalue", "pvalue",
             "stderr", "intercept_stderr"])

        a_result = linregress_mock(
            slope=np.float64(1.6609640474436806), intercept=-8.316627866835997,
            rvalue=0.9999999999999998, pvalue=0.0, stderr=0.0,
            intercept_stderr=0.0)

        input_dict = {"sample_a": a_result}

        expected_dict = {"sample_a": {
            "slope": 1.660964047443,
            "intercept": -8.316627866835,
            "rvalue": 0.999999999999,
            "pvalue": 0.0,
            "stderr": 0.0,
            "intercept_stderr": 0.0}
        }

        output_dict = _convert_linregressresults_to_dict(input_dict)
        self.assertDictEqual(expected_dict, output_dict)

    def test__convert_linregressresults_to_dict_failed_regression(self):
        linregress_mock = namedtuple(
            "linregress_mock",
            ["slope", "intercept", "rvalue", "pvalue",
             "stderr", "intercept_stderr"])

        # if any values in the linregressresults are NaN,
        # the regression failed
        a_result = linregress_mock(
            slope=np.nan, intercept=-8.316627866835997,
            rvalue=0.9999999999999998, pvalue=0.0, stderr=0.0,
            intercept_stderr=0.0)

        input_dict = {"sample_a": a_result}

        output_dict = _convert_linregressresults_to_dict(input_dict)
        # expected to return an empty dict
        self.assertDictEqual({}, output_dict)
