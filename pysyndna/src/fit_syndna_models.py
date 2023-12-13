from __future__ import annotations

import biom
import numpy as np
import os
import pandas as pd
import scipy
import yaml

from typing import Optional

DEFAULT_MIN_SAMPLE_COUNTS = 1

# NB: sample_name instead of sample_id bc that's what qiita uses
SAMPLE_ID_KEY = 'sample_name'
SYNDNA_ID_KEY = 'syndna_id'
SYNDNA_POOL_NUM_KEY = 'syndna_pool_number'
SYNDNA_INDIV_NG_UL_KEY = 'syndna_indiv_ng_ul'
SYNDNA_FRACTION_OF_POOL_KEY = 'syndna_fraction_of_pool'
SYNDNA_POOL_MASS_NG_KEY = 'mass_syndna_input_ng'
SYNDNA_TOTAL_READS_KEY = 'raw_reads_r1r2'
SYNDNA_COUNTS_KEY = 'read_count'
COUNTS_PER_MIL_KEY = 'CPM'
LOG10_COUNTS_PER_MIL_KEY = 'log10_CPM'
SYNDNA_INDIV_NG_KEY = 'syndna_ng'
LOG10_SYNDNA_INDIV_NG_KEY = 'log10_syndna_ng'
LIN_REGRESS_RESULT_KEY = 'lin_regress_by_sample_id'
FIT_SYNDNA_MODELS_LOG_KEY = 'fit_syndna_models_log'


# TODO: if they sequenced over multiple lanes, would be different prep
#  info files--talk to lab about whether they will ever do this :(
#  this would require merge of multiple preparations
def fit_linear_regression_models_for_qiita(
        prep_info_df: pd.DataFrame,
        reads_per_syndna_per_sample_biom: biom.Table,
        min_sample_counts: int = DEFAULT_MIN_SAMPLE_COUNTS,
        syndna_pool_config_fp: Optional[str] = None) -> dict[str: str]:

    """Fits linear regressions predicting mass from counts using Qiita inputs.

    Parameters
    ----------
    prep_info_df: pd.DataFrame
        Dataframe containing prep info for all samples in the prep,
        including SAMPLE_ID, SYNDNA_POOL_NUM_KEY, SYNDNA_POOL_MASS_NG_KEY,
        and SYNDNA_TOTAL_READS_KEY
    reads_per_syndna_per_sample_biom: biom.Table
        Biom table holding read counts aligned to each synDNA in each sample.
        Note: should already have combined forward and reverse counts.
    min_sample_counts: int
        Minimum number of counts required for a sample to be included in
        the regression.  Samples with fewer counts will be excluded.
    syndna_pool_config_fp: str, optional
        Path to the yaml file holding the concentrations of each syndna
        in the syndna pool used in this experiment.  If not provided, will
        look for the config.yml file in the parent directory of this file.

    Returns
    -------
    out_txt_by_out_type : dict of str
        Dictionary of output strings (ready to be written to files) keyed
        by the type of output they contain.  Currently, the following keys
        are defined:
        LIN_REGRESS_RESULT_KEY: yaml of dict[str, dict[str, float]]
        FIT_SYNDNA_MODELS_LOG_KEY: txt log of messages from the fitting process
    """

    # check that the prep_info_df has the expected columns
    expected_prep_info_cols = [
        SAMPLE_ID_KEY, SYNDNA_POOL_NUM_KEY, SYNDNA_POOL_MASS_NG_KEY,
        SYNDNA_TOTAL_READS_KEY]
    _validate_required_columns_exist(
        prep_info_df, expected_prep_info_cols,
        "prep info is missing required column(s)")

    # pull the syndna pool number from the prep info, ensure it is the same for
    # all samples, and convert to the pool name
    syndna_pool_number = prep_info_df[SYNDNA_POOL_NUM_KEY].unique()
    if len(syndna_pool_number) > 1:
        raise ValueError(
            f"Multiple syndna_pool_numbers found in prep info: "
            f"{syndna_pool_number}")
    syndna_pool_name = f"pool{syndna_pool_number[0]}"

    # look in the SYNDNA_INDIV_NG_UL_KEY section of the config file to find the
    # individual syndna concentrations associated with the relevant syndna
    # pool name and turn the resulting dictionary into a dataframe
    config_dict = _extract_config_dict(syndna_pool_config_fp)
    conc_ng_ul_per_indiv_syndna = \
        config_dict[SYNDNA_INDIV_NG_UL_KEY][syndna_pool_name]
    syndna_concs_df = pd.DataFrame(
        conc_ng_ul_per_indiv_syndna.items(),
        columns=[SYNDNA_ID_KEY, SYNDNA_INDIV_NG_UL_KEY])

    # convert input biom table to a pd.SparseDataFrame, which is should act
    # basically like a pd.DataFrame but take up less memory
    reads_per_syndna_per_sample_df = \
        reads_per_syndna_per_sample_biom.to_dataframe(dense=False)

    # fit linear regression models for each sample
    linregress_by_sample_id, msg_list = fit_linear_regression_models(
        syndna_concs_df, prep_info_df, reads_per_syndna_per_sample_df,
        min_sample_counts)
    linregress_results_dict = _convert_linregressresults_to_dict(
        linregress_by_sample_id)

    out_txt_by_out_type = {
        LIN_REGRESS_RESULT_KEY: yaml.safe_dump(linregress_results_dict),
        FIT_SYNDNA_MODELS_LOG_KEY: '\n'.join(msg_list)}

    return out_txt_by_out_type


def _validate_required_columns_exist(
        input_df: pd.DataFrame,
        required_cols_list: list[str],
        error_msg: str):

    """Checks that the input dataframe has the required columns.

    Parameters
    ----------
    input_df: pd.DataFrame
        Dataframe to be checked.
    required_cols_list: list[str]
        List of column names that must be present in the dataframe.
    error_msg: str
        Error message to be raised if any of the required columns are missing.
    """

    missing_cols = set(required_cols_list) - set(input_df.columns)
    if len(missing_cols) > 0:
        raise ValueError(
            f"{error_msg}: {missing_cols}")


def _extract_config_dict(config_fp=None):
    """Extracts a dictionary of config setting from a config file.

    Parameters
    ----------
    config_fp: str, optional
        Path to a yaml config file.  If not provided, will look for a
        "config.yml" file in the parent directory of this file.

    Returns
    -------
    config_dict : dict
        Dictionary of config values keyed by config keys.
    """

    if config_fp is None:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.join(curr_dir, os.pardir)
        config_fp = os.path.join(parent_dir, "config.yml")

    with open(config_fp, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def fit_linear_regression_models(
        syndna_concs_df: pd.DataFrame,
        sample_syndna_weights_and_total_reads_df: pd.DataFrame,
        reads_per_syndna_per_sample_df: pd.DataFrame,
        min_sample_counts: int) -> \
        (dict[str, scipy.stats.LinregressResult], list[str]):

    """Fits per-sample linear regression models predicting mass from counts.

    This fits a linear regression model for each sample, predicting
    log10(mass of instances of a sequence) within a sample from
    log10(counts per million for that sequence) within the sample,
    using spike-in data from synDNAs.

    Parameters
    ----------
    syndna_concs_df: pd.DataFrame
        Dataframe containing SYNDNA_ID_KEY and SYNDNA_INDIV_NG_UL_KEY
        (e.g. 1, 0.1, 0.01, 0.001, 0.0001) for all syndnas in the syndna pool
        used in this experiment
    sample_syndna_weights_and_total_reads_df: pd.DataFrame
        Dataframe containing at least SAMPLE_ID_KEY, SYNDNA_POOL_MASS_NG_KEY
        (the total weight of all syndnas in the sample combined, in ng), and
        SYNDNA_TOTAL_READS_KEY (the number of total reads--not just aligned
        reads--for all syndnas in the sample, including both r1 and r2)
    reads_per_syndna_per_sample_df: pd.DataFrame
        Wide-format dataframe with syndna ids as index and one
        column for each sample id, which holds the read counts
        aligned to that syndna in that sample. Note: should already have
        combined forward and reverse counts.
    min_sample_counts : int
        Minimum number of counts required for a sample to be included in
        the regression.  Samples with fewer counts will be excluded.

    Returns
    -------
    linregress_by_sample_id : dict[str, scipy.stats.LinregressResult]
        returns a dictionary keyed by sample_id, for each sample_id in
        reads_per_syndna_per_sample_df.  Dictionary values are either None
        (if no model could be trained for that sample_id) or a
        scipy.stats.LinregressResult object defining the trained model.
        Suitable for pickling to a file.
    log_messages_list : list[str]
        List of log messages generated during the fitting process.
    """

    log_messages_list = []

    # id any samples that have an inadequate total number of reads aligned
    # to syndna (i.e. less than min_sample_counts). Don't drop yet.
    # Gathering this now bc it is easier while syndna id is still in the index,
    # but we want the full column set while doing the validation checks.
    # Note: synDNA author also made passing mention of dropping samples with
    # inadequate "quality" but didn't provide any guidance on that.
    too_low_counts_mask = \
        reads_per_syndna_per_sample_df.sum(axis=1) < min_sample_counts
    syndnas_to_drop = \
        reads_per_syndna_per_sample_df[too_low_counts_mask].index.tolist()

    # move the syndna ids from the index to a column, bc I hate implicit
    reads_per_syndna_per_sample_df = \
        reads_per_syndna_per_sample_df.reset_index(names=[SYNDNA_ID_KEY])

    # validate that the syndna ids in the config and the data are consistent
    _validate_syndna_id_consistency(syndna_concs_df,
                                    reads_per_syndna_per_sample_df)

    # validate that sample ids in the experiment info and data are consistent
    missing_sample_ids = _validate_sample_id_consistency(
        sample_syndna_weights_and_total_reads_df,
        reads_per_syndna_per_sample_df)
    if missing_sample_ids is not None:
        log_messages_list.append(f'The following sample ids were in the '
                                 f'experiment info but not in the data: '
                                 f'{missing_sample_ids}')

    # NOW remove any syndnas with too few counts from the dataframe,
    # and log if there were any
    filtered_reads_per_syndna_per_sample_df = \
        reads_per_syndna_per_sample_df[
            ~reads_per_syndna_per_sample_df[SYNDNA_ID_KEY].isin(
                syndnas_to_drop)]
    if len(syndnas_to_drop) > 0:
        log_messages_list.append(f'The following syndnas were dropped '
                                 f'because they had fewer than '
                                 f'{min_sample_counts} total reads aligned:'
                                 f'{syndnas_to_drop}')

    # reformat filtered_reads_per_syndna_per_sample_df into "long form":
    # columns for syndna id, sample id, and read count
    working_df = filtered_reads_per_syndna_per_sample_df.melt(
        id_vars=[SYNDNA_ID_KEY], var_name=SAMPLE_ID_KEY,
        value_name=SYNDNA_COUNTS_KEY)

    # merge w sample_total_reads_df to include total_reads column
    working_df = working_df.merge(sample_syndna_weights_and_total_reads_df,
                                  on=SAMPLE_ID_KEY, how='left')

    # calculate the weight in ng of *each* syndna in each sample
    working_df = _calc_indiv_syndna_weights(syndna_concs_df, working_df)

    # fit linear regression models for each sample
    linregress_by_sample_id = _fit_linear_regression_models(working_df)

    return linregress_by_sample_id, log_messages_list


def _validate_syndna_id_consistency(
        syndna_concs_df: pd.DataFrame,
        reads_per_syndna_per_sample_df: pd.DataFrame):
    """
    Checks that the syndna ids in the config and the data are consistent.

    Parameters
    ----------
    syndna_concs_df:
        Dataframe containing SYNDNA_ID_KEY and SYNDNA_INDIV_NG_UL_KEY
        (e.g. 1, 0.1, 0.01, 0.001, 0.0001) for all syndnas in the syndna pool
        used in this experiment
    reads_per_syndna_per_sample_df:
        Dataframe with a column for syndna id and then one additional column
        for each sample_id, which holds the read counts aligned to that syndna
        in that sample. Note: should already have combined forward and
        reverse counts.

    Raises
    ------
    ValueError
        If there are syndna ids in the data that are not in the config, or
        vice versa.
    """

    syndna_ids_in_config = set(syndna_concs_df[SYNDNA_ID_KEY])
    syndna_ids_in_data = set(reads_per_syndna_per_sample_df[SYNDNA_ID_KEY])

    # if there are syndna ids in the data that are not in the config, raise
    # an error, since we don't know how to process that
    data_only_syndnas = syndna_ids_in_data - syndna_ids_in_config
    if len(data_only_syndnas) > 0:
        raise ValueError(
            f"Detected {len(data_only_syndnas)} syndna feature(s) in the "
            f"read data that were not in the config: {data_only_syndnas}")

    # if there are syndna ids in the config that are not in the data,
    # raise an error.... that means at least one of the syndnas in the pool
    # didn't sequence at all, and that seems like a problem.
    config_only_syndnas = syndna_ids_in_config - syndna_ids_in_data
    if len(config_only_syndnas) > 0:
        raise ValueError(
            f"Missing the following {len(config_only_syndnas)} "
            f"required syndna feature(s) in the read data: "
            f"{config_only_syndnas}")


def _validate_sample_id_consistency(
        sample_syndna_weights_and_total_reads_df: pd.DataFrame,
        reads_per_syndna_per_sample_df: pd.DataFrame) -> list[str] | None:
    """
    Checks that the sample ids in the experiment info and data are consistent.

    Parameters
    ----------
    sample_syndna_weights_and_total_reads_df: pd.DataFrame
        Dataframe containing at least SAMPLE_ID_KEY, SYNDNA_POOL_MASS_NG_KEY
        (the total weight of all syndnas in the sample combined, in ng), and
        SYNDNA_TOTAL_READS_KEY (the number of total reads--not just aligned
        reads--for all syndnas in the sample, including both r1 and r2)
    reads_per_syndna_per_sample_df: pd.DataFrame
        Dataframe with a column for syndna_id and then one additional column
        for each sample_id, which holds the read counts aligned to that syndna
        in that sample. Note: should already have combined forward and
        reverse counts.

    Raises
    ------
    ValueError
        If there are sample ids in the data that aren't in the experiment info

    Returns
    -------
    missing_sample_ids : List[str] | None
        List of sample ids that are in the experiment info but not in the
        data.  None if all sample ids in the experiment info were in the data.
    """

    sample_ids_in_info = \
        set(sample_syndna_weights_and_total_reads_df[SAMPLE_ID_KEY])
    sample_ids_in_data = set(reads_per_syndna_per_sample_df.columns)
    sample_ids_in_data.remove(SYNDNA_ID_KEY)

    # if there are sample ids in the data that are not in the info, raise
    # an error, since we don't know how to process that
    data_only_samples = sample_ids_in_data - sample_ids_in_info
    if len(data_only_samples) > 0:
        raise ValueError(
            f"Found sample ids in reads_per_syndna_per_sample_df that were "
            f"not in sample_syndna_weights_and_total_reads_df: "
            f"{data_only_samples}")

    # check if there are sample ids in the info that are not in the data and
    # if so, capture a list of them. Sometimes a sample just fails sequencing
    # and that shouldn't preclude processing the others that did work, but we
    # want to know about it.
    missing_sample_ids_set = sample_ids_in_info - sample_ids_in_data

    if len(missing_sample_ids_set) > 0:
        missing_sample_ids = list(missing_sample_ids_set)
    else:
        missing_sample_ids = None

    return missing_sample_ids


def _calc_indiv_syndna_weights(
        syndna_concs_df: pd.DataFrame,
        working_df: pd.DataFrame) -> pd.DataFrame:

    """Calculates the weight in ng of each syndna in each sample.

    Parameters
    ----------
    syndna_concs_df: pd.DataFrame
        Dataframe containing SYNDNA_ID_KEY and SYNDNA_INDIV_NG_UL_KEY
        (e.g. 1, 0.1, 0.01, 0.001, 0.0001) for all syndnas in the syndna pool
        used in this experiment
    working_df: pd.DataFrame
        Long-form dataframe containing at least SAMPLE_ID_KEY, SYNDNA_ID_KEY,
        and SYNDNA_POOL_MASS_NG_KEY

    Returns
    -------
    working_df : pd.DataFrame
        Returns the input working_df with additional columns:
        SYNDNA_INDIV_NG_UL_KEY: the weight in ng of each syndna in each sample
        SYNDNA_FRACTION_OF_POOL_KEY: the fraction of the mass of the syndna
            pool represented by each individual syndna
        SYNDNA_INDIV_NG_KEY: the weight in ng of each syndna in each sample
    """

    # get the total concentration of syndna in the pool (a scalar)
    # by summing up the concentrations of each individual syndna
    total_syndna_ng_per_ul = syndna_concs_df[SYNDNA_INDIV_NG_UL_KEY].sum()

    # add a column for the unitless fraction of the syndna pool made up of
    # each individual syndna by dividing the syndna_ng_per_uL of each
    # syndna by the total_syndna_ng_per_ul for the pool
    syndna_concs_df[SYNDNA_FRACTION_OF_POOL_KEY] = (
            syndna_concs_df[SYNDNA_INDIV_NG_UL_KEY] / total_syndna_ng_per_ul)
    working_df = working_df.merge(
        syndna_concs_df, on=SYNDNA_ID_KEY, how='left')

    # calculate the weight in ng of *each* syndna in each sample by multiplying
    # the total weight of all syndna input added to the sample by the fraction
    # of the syndna pool represented by each syndna
    working_df[SYNDNA_INDIV_NG_KEY] = (
            working_df[SYNDNA_POOL_MASS_NG_KEY] *
            working_df[SYNDNA_FRACTION_OF_POOL_KEY])

    return working_df


def _fit_linear_regression_models(working_df: pd.DataFrame) -> \
        dict[str, scipy.stats.LinregressResult]:

    """Fits per-sample linear regression models predicting mass from counts.

    This function fits a linear regression model for each sample,
    predicting log10(mass of instances of a sequence) within a sample
    from log10(counts per million for that sequence) within the sample,
    using spike-in data from synDNAs.

    Parameters
    ----------
    working_df: pd.DataFrame
        Long-form dataframe containing at least SAMPLE_ID_KEY,
        SYNDNA_COUNTS_KEY, SYNDNA_TOTAL_READS_KEY, and
        SYNDNA_INDIV_NG_KEY columns.

    Returns
    -------
    linregress_by_sample_id : dict[str, scipy.stats.LinregressResult]
        returns a dictionary keyed by sample_id, for each sample_id in
        reads_per_syndna_per_sample_df.  Dictionary values are
        scipy.stats.LinregressResult objects defining the trained models.
    """

    # drop any rows where the count value is 0--can't take log of 0
    working_df = working_df[working_df[SYNDNA_COUNTS_KEY] > 0].copy()

    # add a column of counts per million (CPM) by dividing the count value
    # in each read_count by the total number of reads for its sample_id and
    # then multiplying by a million (1,000,000)
    working_df.loc[:, COUNTS_PER_MIL_KEY] = \
        (working_df[SYNDNA_COUNTS_KEY] /
         working_df[SYNDNA_TOTAL_READS_KEY]) * 1000000

    # add a column of log10(CMP) by taking the log base 10 of the CPM column
    working_df.loc[:, LOG10_COUNTS_PER_MIL_KEY] = \
        np.log10(working_df[COUNTS_PER_MIL_KEY])

    # add a column for the log10 of the syndna ng column
    working_df.loc[:, LOG10_SYNDNA_INDIV_NG_KEY] = \
        np.log10(working_df[SYNDNA_INDIV_NG_KEY])

    # loop over each sample id and fit a linear regression model predicting
    # log10(dna ng) from log10(counts per million)
    linregress_by_sample_id = {}
    for curr_sample_id in working_df[SAMPLE_ID_KEY].unique():
        curr_sample_df = \
            working_df[working_df[SAMPLE_ID_KEY] == curr_sample_id]

        # TODO: I need to know what kind of errors this can throw; some of them
        #  may just mean a linear regression can't be fit for this sample, but
        #  others may mean something is wrong with the data or the code.
        #  Once I know which is which, I can decide whether to try/catch
        #  anything silently.

        try:
            curr_linregress_result = scipy.stats.linregress(
                curr_sample_df[LOG10_COUNTS_PER_MIL_KEY],
                curr_sample_df[LOG10_SYNDNA_INDIV_NG_KEY])
        except Exception:
            curr_linregress_result = None

        # record the whole lingregress result object in the output dictionary
        linregress_by_sample_id[curr_sample_id] = curr_linregress_result
    # next sample_id

    return linregress_by_sample_id


def _convert_linregressresults_to_dict(
        linregress_by_sample_id: dict[str, scipy.stats.LinregressResult]) -> \
        dict[str, dict[str, float]]:

    """Converts a scipy.stats.LinregressResult object to a dictionary.

    Returns
    -------
    linregress_result_dict :  dict[str, dict[str, float]]
        Dictionary keyed by sample id, containing for each sample either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a dictionary
        representation of the sample's LinregressResult, with each property
        name as a key and that property's value as the value.  Values are
        rounded to no more than 15 decimal places.
    """

    linregress_result_dict = {}
    for curr_sample_id, curr_linregress_result in \
            linregress_by_sample_id.items():
        if curr_linregress_result is None:
            linregress_result_dict[curr_sample_id] = None
        else:
            new_dict = {}
            curr_dict = curr_linregress_result._asdict()
            for k, v in curr_dict.items():
                # if any of the values is NaN, this regression failed
                if np.isnan(v):
                    new_dict = None
                    break

                # convert to regular floats, bc yaml doesn't like np.float64
                if isinstance(v, np.float64):
                    new_float = float(v)
                    # truncate to 12 decimal places; the precision of float in
                    # python is dependent upon the underlying C implementation,
                    # and sometimes differs between mac/ubuntu past this point.
                    new_dict[k] = truncate(new_float, 12)

            linregress_result_dict[curr_sample_id] = new_dict

    return linregress_result_dict


def truncate(a_float, num_decimals):
    """Truncates a float to the specified number of decimal places.

    Parameters
    ----------
    a_float : float
        Float to be truncated.
    num_decimals : int
        Number of decimal places to which the float should be truncated.

    Returns
    -------
    truncated_float : float
        Float truncated to the specified number of decimal places.
    """

    # multiply a_float by 10^num_decimals, convert to an integer, then divide
    # by 10^num_decimals to get the truncated float
    truncated_float = int(a_float * 10 ** num_decimals) / 10 ** num_decimals
    return truncated_float
