import biom
import numpy as np
import pandas as pd
import yaml
from typing import Optional, Union, Dict, List
from pysyndna.src.util import calc_copies_genomic_element_per_g_series, \
    calc_gs_genomic_element_in_aliquot, \
    validate_required_columns_exist, \
    validate_metadata_vs_reads_id_consistency, filter_data_by_sample_info, \
    validate_metadata_vs_prep_id_consistency, cast_cols, \
    DNA_BASEPAIR_G_PER_MOLE, NANOGRAMS_PER_GRAM, \
    SAMPLE_ID_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY, ELUTE_VOL_UL_KEY, \
    REQUIRED_SAMPLE_INFO_KEYS

from pysyndna.src.fit_syndna_models import SYNDNA_POOL_MASS_NG_KEY, \
    SLOPE_KEY, INTERCEPT_KEY, SAMPLE_TOTAL_READS_KEY

DEFAULT_SYNDNA_MASS_FRACTION_OF_SAMPLE = 0.05
DEFAULT_READ_LENGTH = 150
DEFAULT_MIN_PERCENT_COVERAGE = 1
DEFAULT_MIN_RSQUARED = 0.8

CELL_COUNT_RESULT_KEY = 'cell_count_biom'
CELL_COUNT_LOG_KEY = 'calc_cell_counts_log'

GDNA_CONCENTRATION_NG_UL_KEY = 'extracted_gdna_concentration_ng_ul'
GDNA_FROM_ALIQUOT_MASS_G_KEY = 'extracted_gdna_concentration_g'
# NB: below is NOT the full mass of gDNA extracted from the sample, but
# ONLY the mass of gDNA that was put into sequencing. This mass should
# NOT include the additional mass of the syndna pool added to sequencing.
SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY = 'sequenced_sample_gdna_mass_ng'
OGU_ID_KEY = 'ogu_id'
OGU_READ_COUNT_KEY = 'ogu_read_count'
OGU_CPM_KEY = 'ogu_CPM'
LOG_10_OGU_CPM_KEY = 'log10_ogu_CPM'
OGU_PERCENT_COVERAGE_KEY = 'percent_coverage_of_ogu'
TOTAL_OGU_READS_KEY = 'total_reads_per_ogu'
LOG_10_OGU_GDNA_MASS_NG_KEY = 'log10_ogu_gdna_mass_ng'
OGU_LEN_IN_BP_KEY = 'ogu_len_in_bp'
OGU_GDNA_MASS_NG_KEY = 'ogu_gdna_mass_ng'
OGU_GENOMES_PER_G_OF_GDNA_KEY = 'ogu_genomes_per_g_of_gdna'
OGU_CELLS_PER_G_OF_GDNA_KEY = 'ogu_cells_per_g_of_gdna'
OGU_CELLS_PER_G_OF_SAMPLE_KEY = 'ogu_cells_per_g_of_sample'
# NB: below is based on the full mass of gDNA extracted from the sample
# (NOT limited to the amount of gDNA that was put into sequencing, unlike
# SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY)
GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY = 'gdna_mass_to_sample_mass_ratio'
REQUIRED_DNA_PREP_INFO_KEYS = [SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
                               ELUTE_VOL_UL_KEY, SAMPLE_TOTAL_READS_KEY]


def _calc_gdna_mass_to_sample_mass_by_sample_df(
        absolute_quant_params_per_sample_df: pd.DataFrame) -> pd.Series:

    """Calculates ratio of extracted gDNA mass to sample mass for each sample.

    Note that the sample mass is the mass of the sample material (only, not
    buffer, tube, etc) that went into the extraction, which may be different
    from the total mass of sample that was collected.

    Parameters
    ----------
    absolute_quant_params_per_sample_df:  pd.DataFrame
        A Dataframe of at least SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
        SAMPLE_IN_ALIQUOT_MASS_G_KEY, and ELUTE_VOL_UL_KEY for
        each sample.

    Returns
    -------
    gdna_mass_to_sample_mass_by_sample_series : pd.Series
        A Series with index of sample id and values of the ratio of gDNA mass
        units extracted from each mass unit of input sample (only) mass.
    """

    # get the total grams of gDNA that are in the elute after extraction;
    # this is sample-specific
    working_df = calc_gs_genomic_element_in_aliquot(
        absolute_quant_params_per_sample_df, GDNA_CONCENTRATION_NG_UL_KEY,
        GDNA_FROM_ALIQUOT_MASS_G_KEY)

    # determine how many mass units of gDNA are produced from the extraction of
    # each mass unit of sample material; this is sample-specific:
    # grams of gDNA after extraction divided grams of sample material.
    gdna_mass_to_sample_mass_ratio = \
        working_df[GDNA_FROM_ALIQUOT_MASS_G_KEY] / \
        working_df[SAMPLE_IN_ALIQUOT_MASS_G_KEY]

    gdna_mass_to_sample_mass_ratio.name = GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY
    gdna_mass_to_sample_mass_ratio.index = working_df[SAMPLE_ID_KEY]
    gdna_mass_to_sample_mass_ratio.index.name = SAMPLE_ID_KEY

    return gdna_mass_to_sample_mass_ratio


def _series_to_df(a_series, index_col_name, val_col_name):
    """Converts a pd.Series to two-column pd.DataFrame (from index and value)

    Parameters
    ----------
    a_series : pd.Series
        A Series to be converted to a dataframe.
    index_col_name : str
        Name of the index-derived in the resulting dataframe.
    val_col_name : str
        Name of the values-derived column in the resulting dataframe.

    Returns
    -------
    a_df : pd.DataFrame
        A Dataframe with two columns, one from the index and one containing the
        values from the input series.
    """

    a_df = a_series.to_frame().reset_index()
    a_df.columns = [index_col_name, val_col_name]

    return a_df


def _calc_long_format_ogu_cell_counts_df(
        linregress_by_sample_id: Dict[str, Dict[str, float]],
        ogu_counts_per_sample_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        per_sample_calc_info_df: pd.DataFrame,
        read_length: int,
        min_coverage: float,
        min_rsquared: float) -> (Union[pd.DataFrame, None], List[str]):

    """Predicts the # of cells of each OGU in each sample from the read counts.

    Parameters
    ----------
    linregress_by_sample_id : dict[str, dict[str, float]]
        Dictionary keyed by sample id, containing for each sample either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a dictionary
        representation of the sample's LinregressResult.
    ogu_counts_per_sample_df: pd.DataFrame
        A Dataframe with a column for OGU_ID_KEY and then one additional column
        for each sample id, which holds the read counts aligned to that OGU in
        that sample.
    ogu_lengths_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    per_sample_calc_info_df : pd.DataFrame
        A Dataframe of SAMPLE_ID_KEY, GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY,
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY, and SAMPLE_TOTAL_READS_KEY
        for each sample.
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable coverage of an OGU needed to include that OGU
        in the output.
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.

    Returns
    -------
    ogu_cell_counts_df : pd.DataFrame | None
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        OGU_CELLS_PER_G_OF_GDNA_KEY, and OGU_CELLS_PER_G_OF_GDNA_KEY,
        in addition to various intermediate calculation columns.  Could be
        None if no cell counts were calculated for any sample.
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    log_messages_list = []

    # reformat biom info into a "long format" table with
    # columns needed for per-sample calculation
    working_df, prep_log_messages = _prepare_cell_counts_calc_df(
        ogu_counts_per_sample_df, ogu_lengths_df, read_length, min_coverage)
    log_messages_list.extend(prep_log_messages)

    # loop through a series of the unique sample ids in the working_df
    cell_counts_df = None
    sample_ids = working_df[SAMPLE_ID_KEY].unique()
    for curr_sample_id in sample_ids:
        # calculate the predicted number of cells of each OGU per gram of
        # gDNA in this sample and also per gram of stool in this sample
        curr_sample_df, curr_log_msgs = _calc_ogu_cell_counts_df_for_sample(
            curr_sample_id, linregress_by_sample_id,
            per_sample_calc_info_df, working_df, min_rsquared)
        log_messages_list.extend(curr_log_msgs)
        if curr_sample_df is None:
            log_messages_list.append(f"No cell counts calculated for "
                                     f"sample {curr_sample_id}")

            # NB: if no cell counts were calculated for this sample,
            # this sample is left out of the final cell_counts_df.
            continue

        # if cell_counts_df does not yet exist, create it from curr_sample_df;
        # otherwise, append curr_sample_df to the existing cell_counts_df
        if cell_counts_df is None:
            cell_counts_df = curr_sample_df
        else:
            # append the current sample's df to the existing cell_counts_df
            cell_counts_df = pd.concat([cell_counts_df, curr_sample_df])
    # next sample_id

    if cell_counts_df is None:
        raise ValueError("No cell counts calculated for any sample")

    return cell_counts_df, log_messages_list


def _prepare_cell_counts_calc_df(
        ogu_counts_per_sample_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        read_length: int,
        min_coverage: float) -> (pd.DataFrame, List[str]):

    """Prepares long-format dataframe containing fields needed for later calcs.

    Parameters
    ----------
    ogu_counts_per_sample_df: pd.DataFrame
        Wide-format dataframe with ogu ids as index and one
        column for each sample id, which holds the read counts
        aligned to that OGU in that sample.
    ogu_lengths_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable % coverage of an OGU in a sample needed to include
        that OGU/sample in the output.  Zaramela paper uses 1%.

    Returns
    -------
    working_df : pd.DataFrame
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        OGU_READ_COUNT_KEY, TOTAL_OGU_READS_KEY, OGU_LEN_IN_BP_KEY, and
        OGU_PERCENT_COVERAGE_KEY; contains rows for OGU/sample combinations
        with % coverage >= min_coverage
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    log_messages_list = []

    # cast scalar inputs in case they come in as strings or something
    min_coverage = float(min_coverage)
    read_length = int(read_length)

    # calculate the total number of reads per OGU by summing up the read counts
    # for each OGU across all samples (i.e., summing each row of biom table)
    total_ogu_counts_series = ogu_counts_per_sample_df.sum(axis=1)
    total_ogu_counts_df = _series_to_df(
        total_ogu_counts_series, OGU_ID_KEY, TOTAL_OGU_READS_KEY)

    # move the ogu ids from the index to a column, bc I hate implicit
    working_df = ogu_counts_per_sample_df.copy()
    working_df = working_df.reset_index(names=[OGU_ID_KEY])

    # reformat biom info into a "long format" table with
    # columns for OGU_ID_KEY, SAMPLE_ID_KEY, OGU_READ_COUNT_KEY
    working_df = working_df.melt(
        id_vars=[OGU_ID_KEY], var_name=SAMPLE_ID_KEY,
        value_name=OGU_READ_COUNT_KEY)

    # add a column for OGU_LEN_IN_BP_KEY (yes, this will be repeated, but it is
    # convenient to have everything in one table)
    working_df = working_df.merge(ogu_lengths_df, on=OGU_ID_KEY, how='left')

    # add total reads per OGU column to table (again, will be redundancies)
    working_df = working_df.merge(
        total_ogu_counts_df, on=OGU_ID_KEY, how='left')

    # calculate the % coverage per OGU per sample by multiplying each
    # read_count cell value by the number of bases in the read (read_length)
    # and then dividing by the ogu_length for that OGU, then multiplying by 100
    working_df[OGU_PERCENT_COVERAGE_KEY] = (
            (working_df[OGU_READ_COUNT_KEY] * read_length) /
            working_df[OGU_LEN_IN_BP_KEY]) * 100

    # drop records for OGU/sample combinations with % coverage < min_coverage
    too_low_cov_mask = working_df[OGU_PERCENT_COVERAGE_KEY] < min_coverage
    temp_ids = working_df[SAMPLE_ID_KEY] + ';' + working_df[OGU_ID_KEY]
    too_low_cov_samples_list = (
        temp_ids.loc[too_low_cov_mask].tolist())
    if len(too_low_cov_samples_list) > 0:
        log_messages_list.append(f'The following items have % coverage lower'
                                 f' than the minimum of {min_coverage}: '
                                 f'{too_low_cov_samples_list}')
    working_df = working_df[
        working_df[OGU_PERCENT_COVERAGE_KEY] >= min_coverage]
    working_df = working_df.reset_index(drop=True)

    return working_df, log_messages_list


def _calc_ogu_cell_counts_df_for_sample(
        sample_id: str,
        linregress_by_sample_id: Dict[str, Dict[str, float]],
        per_sample_info_df: pd.DataFrame,
        working_df: pd.DataFrame,
        min_rsquared: float,
        is_test: Optional[bool] = False) \
        -> (Union[pd.DataFrame, None], List[str]):

    """Calculates # cells of each OGU per gram of sample material for sample.

    Parameters
    ----------
    sample_id: str
        Sample id for which to calculate cell counts.
    linregress_by_sample_id : dict[str, dict[str: float]]
        Dictionary keyed by sample id, containing for each sample either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a dictionary
        representation of the sample's LinregressResult.
    per_sample_info_df : pd.DataFrame
        A Dataframe of SAMPLE_ID_KEY, GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY, and
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY, and SAMPLE_TOTAL_READS_KEY
        for each sample.
    working_df : pd.DataFrame
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        OGU_READ_COUNT_KEY, and OGU_LEN_IN_BP_KEY
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.
    is_test: Optional[bool]
        Default is False.  If True, the function will use the less-precise
        value of Avogadro's number (6.022*(10^23)) used in
        SynDNA_saliva_samples_analysis.ipynb.  Otherwise, the more precise
        value (6.02214076*(10^23)) will be used.  This is True in testing ONLY.

    Returns
    -------
    sample_df : pd.DataFrame | None
        None if the specified sample id has no linear model or has a model with
        R^2 < min_rsquared.  Otherwise, a long-format dataframe with columns
        for at least OGU_ID_KEY, SAMPLE_ID_KEY, OGU_READ_COUNT_KEY,
        OGU_LEN_IN_BP_KEY, OGU_CELLS_PER_G_OF_GDNA_KEY,
        OGU_GENOMES_PER_G_OF_GDNA_KEY, OGU_CELLS_PER_G_OF_GDNA_KEY,
        and OGU_CELLS_PER_G_OF_SAMPLE_KEY
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    log_messages_list = []

    # cast scalar inputs in case they come in as strings or something
    min_rsquared = float(min_rsquared)

    # get the linear regression result for this sample
    linregress_result = linregress_by_sample_id.get(sample_id)
    if linregress_result is None:
        log_messages_list.append(f"No linear regression fitted for sample "
                                 f"{sample_id}")
        return None, log_messages_list

    r_squared = linregress_result["rvalue"]**2
    if r_squared < min_rsquared:
        log_messages_list.append(f"R^2 of linear regression for sample "
                                 f"{sample_id} is {r_squared}, which is less "
                                 f"than the minimum allowed value of "
                                 f"{min_rsquared}.")
        return None, log_messages_list

    # get df of the rows of the working_df specific to this sample
    sample_df = working_df[
        working_df[SAMPLE_ID_KEY] == sample_id].copy()

    # get the total reads sequenced for this sample
    sample_total_reads = per_sample_info_df.loc[
        per_sample_info_df[SAMPLE_ID_KEY] == sample_id,
        SAMPLE_TOTAL_READS_KEY].values[0]

    # predict mass of each OGU's gDNA in this sample from its counts
    # using the linear model
    ogu_gdna_masses = _calc_ogu_gdna_mass_ng_series_for_sample(
            sample_df, linregress_result[SLOPE_KEY],
            linregress_result[INTERCEPT_KEY], sample_total_reads)
    sample_df[OGU_GDNA_MASS_NG_KEY] = \
        sample_df[OGU_ID_KEY].map(ogu_gdna_masses)

    # get the mass of gDNA put into sequencing for this sample
    sequenced_sample_gdna_mass_ng = per_sample_info_df.loc[
        per_sample_info_df[SAMPLE_ID_KEY] == sample_id,
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY].values[0]

    # calc the # of genomes of each OGU per gram of gDNA in this sample
    ogu_genomes_per_gdnas = _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
        sample_df, sequenced_sample_gdna_mass_ng, is_test=is_test)
    sample_df[OGU_GENOMES_PER_G_OF_GDNA_KEY] = \
        sample_df[OGU_ID_KEY].map(ogu_genomes_per_gdnas)

    # assume the # of cells of the microbe represented by each OGU
    # per gram of gDNA in this sample is the same as the number of genomes
    # of that OGU per gram of gDNA. This is known to be not quite right
    # (since some cells are dividing and thus have an extra copy, and also
    # because apparently polyploidy is not uncommon among microbes), but
    # that's why these things are called "simplifying assumptions" ...
    sample_df[OGU_CELLS_PER_G_OF_GDNA_KEY] = \
        sample_df[OGU_GENOMES_PER_G_OF_GDNA_KEY]

    # calc the # of cells of each OGU per gram of actual sample material
    # (e.g., per gram of stool if these are fecal samples) for this sample
    mass_ratio_for_sample = per_sample_info_df.loc[
        per_sample_info_df[SAMPLE_ID_KEY] == sample_id,
        GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY].values[0]
    sample_df[OGU_CELLS_PER_G_OF_SAMPLE_KEY] = \
        sample_df[OGU_CELLS_PER_G_OF_GDNA_KEY] * \
        mass_ratio_for_sample

    return sample_df, log_messages_list


def _calc_ogu_gdna_mass_ng_series_for_sample(
        sample_df: pd.DataFrame,
        sample_linregress_slope: float,
        sample_linregress_intercept: float,
        sample_total_reads: int) -> pd.Series:

    """Calculates mass of OGU gDNA in ng for each OGU in a sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        A Dataframe with rows for a single sample, containing at least columns
        for OGU_ID_KEY and OGU_READ_COUNT_KEY.
    sample_linregress_slope: float
        Slope of the linear regression model for the sample.
    sample_linregress_intercept: float
        Intercept of the linear regression model for the sample.
    sample_total_reads: int
        Total number of reads for the sample (including all reads, not just
        aligned ones).

    Returns
    -------
    ogu_genomes_per_g_of_gdna_series : pd.Series
        A Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU per gram of gDNA in the sample.
    """
    working_df = sample_df.copy()

    # add a column of counts per million (CPM) for each ogu by dividing
    # each read_count by the total number of reads for this sample
    # and then multiplying by a million (1,000,000)
    # NB: dividing int/int in python gives float
    working_df[OGU_CPM_KEY] = (working_df[OGU_READ_COUNT_KEY] /
                               sample_total_reads) * 1000000

    # add column of log10(ogu CPM) by taking log base 10 of the ogu CPM column
    working_df[LOG_10_OGU_CPM_KEY] = np.log10(working_df[OGU_CPM_KEY])

    # calculate log10(ogu gdna mass) of each OGU's gDNA in this sample
    # by multiplying each OGU's log10(ogu CPM) by the slope of this sample's
    # regression model and adding the model's intercept.
    # NB: this requires that the linear regression models were derived
    # using synDNA masses *in ng* and not in some other unit.
    working_df[LOG_10_OGU_GDNA_MASS_NG_KEY] = (
            working_df[LOG_10_OGU_CPM_KEY] *
            sample_linregress_slope +
            sample_linregress_intercept)

    # calculate the actual mass in ng of each OGU's gDNA by raising 10 to the
    # log10(ogu gdna mass) power; set the series index to the OGU_ID_KEY
    ogu_gdna_mass_ng_series = \
        10 ** working_df[LOG_10_OGU_GDNA_MASS_NG_KEY]
    ogu_gdna_mass_ng_series.name = OGU_GDNA_MASS_NG_KEY
    ogu_gdna_mass_ng_series.index = sample_df[OGU_ID_KEY]

    return ogu_gdna_mass_ng_series


def _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
        sample_df: pd.DataFrame,
        total_sample_gdna_mass_ng: float,
        is_test: Optional[bool] = False) -> pd.Series:

    """Calculates # of OGU genomes per gram of gDNA for each OGU in a sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        A Dataframe with rows related to only a single sample, containing
        at least columns for OGU_ID_KEY, OGU_LEN_IN_BP_KEY, and
        OGU_GDNA_MASS_NG_KEY.
    total_sample_gdna_mass_ng: float
        Total mass of gDNA in the sample (across all OGUs) in ng. Note this
        should NOT include the mass of the syndna added to the sample
    is_test: Optional[bool]
        Default is False.  If True, the function will use the less-precise
        value of Avogadro's number (6.022*(10^23)) used in cell [16] of the
        notebook, rather than the more precise value (6.02214076*(10^23))
        calculation used if False.  This is True in testing ONLY.

    Returns
    -------
    ogu_genomes_per_g_of_gdna_series : pd.Series
        A Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU per gram of gDNA of the sample.
    """

    # calculate the number of genomes of each OGU in the sequenced sample
    ogu_genomes_series_for_sample = \
        _calc_ogu_genomes_series_for_sample(sample_df, is_test=is_test)

    # the above is the number of genomes of each OGU that were found in all
    # the sequnced sample gDNA; we want the number of genomes of each OGU
    # per gram of gDNA in the sample, so we divide by the total gDNA mass
    # in the sample (across all OGUs).  Note that both measurements are in ng.
    ogu_genomes_per_ng_of_gdna_series = \
        ogu_genomes_series_for_sample / total_sample_gdna_mass_ng

    # to get the number of genomes per gram of gDNA, we multiply by 1e9
    # (since there are 1e9 ng/gram)
    ogu_genomes_per_g_of_gdna_series = \
        ogu_genomes_per_ng_of_gdna_series * 1e9

    return ogu_genomes_per_g_of_gdna_series


def _calc_ogu_genomes_series_for_sample(
        sample_df: pd.DataFrame,
        is_test: Optional[bool] = False) -> pd.Series:

    """Calculates # of OGU genomes for each OGU in a sequenced sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        A Dataframe with rows related to only a single sample, containing
        at least columns for OGU_ID_KEY, OGU_LEN_IN_BP_KEY, and
        OGU_GDNA_MASS_NG_KEY.
    is_test: Optional[bool]
        Default is False.  If True, the function will use the less-precise
        value of Avogadro's number (6.022*(10^23)) used in cell [16] of the
        https://github.com/lzaramela/SynDNA/blob/main/SynDNA_saliva_samples_analysis.ipynb
        notebook, rather than the more precise value (6.02214076*(10^23))
        calculation used if False.  This is True in testing ONLY.

    Returns
    -------
    ogu_genomes_series : pd.Series
        A Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU in the sequenced sample.

    This calculates the total number of genomes for each OGU in the sequenced
    sample by the equation:

        mass of OGU's gDNA in ng * Avogadro's number in genomes/mole
    =	---------------------------------------------------------------
        length of OGU genome in basepairs *
            650 g/mole per basepair (on average) * 10^9 ng/g

    NB: the constant factor on the bottom right MUST CHANGE if the
    units of the OGU gDNA mass are NOT nanograms!

    Avogadro's number is 6.02214076 × 10^23 , and is the number of
    molecules--in this case, genomes--in a mole of a substance.
    """

    ogu_copies_per_g_series = calc_copies_genomic_element_per_g_series(
        sample_df[OGU_LEN_IN_BP_KEY], DNA_BASEPAIR_G_PER_MOLE, is_test=is_test)
    ogu_copies_per_extracted_sample_series = \
        sample_df[OGU_GDNA_MASS_NG_KEY] * \
        ogu_copies_per_g_series / NANOGRAMS_PER_GRAM

    # Set the index of the series to be the OGU_ID_KEY
    ogu_copies_per_extracted_sample_series.index = sample_df[OGU_ID_KEY]
    return ogu_copies_per_extracted_sample_series


def calc_ogu_cell_counts_biom(
        absolute_quant_params_per_sample_df: pd.DataFrame,
        linregress_by_sample_id: Dict[str, Dict[str, float]],
        ogu_counts_per_sample_biom: biom.Table,
        ogu_percent_coverage_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        read_length: int,
        min_coverage: float,
        min_rsquared: float,
        output_cell_counts_metric: str) -> (biom.Table, List[str]):

    """Calcs input cell count metric for each ogu & sample via linear models.

    Parameters
    ----------
    absolute_quant_params_per_sample_df:  pd.DataFrame
        A Dataframe of at least SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
        SAMPLE_IN_ALIQUOT_MASS_G_KEY, ELUTE_VOL_UL_KEY,
        SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY, and SAMPLE_TOTAL_READS_KEY
        for each sample.
    linregress_by_sample_id : dict[str, dict[str: float]]
        Dictionary keyed by sample id, containing for each sample either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a dictionary
        representation of the sample's LinregressResult.
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_percent_coverage_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_PERCENT_COVERAGE_KEY for each OGU.
    ogu_lengths_df : pd.DataFrame
        A Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable % coverage of an OGU across the whole dataset
        required to include that OGU in the output.
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.
    output_cell_counts_metric : str
        Name of the desired output cell count metric; options are
        OGU_CELLS_PER_G_OF_GDNA_KEY and OGU_CELLS_PER_G_OF_SAMPLE_KEY.

    Returns
    -------
    ogu_cell_counts_biom : biom.Table
        An OGU/sample biom.Table holding the predicted number of cells per gram
        of material, with the material type being defined by the
        output_cell_counts_metric.
    log_messages_list : list[str]
        List of strings containing log messages generated by this function.
    """

    # check if the inputs all have the required columns
    required_cols_list = list(
        {SAMPLE_ID_KEY, SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY,
         SAMPLE_IN_ALIQUOT_MASS_G_KEY} |
        set(REQUIRED_DNA_PREP_INFO_KEYS))
    validate_required_columns_exist(
        absolute_quant_params_per_sample_df, required_cols_list,
        "sample info is missing required column(s)")

    validate_required_columns_exist(
        ogu_lengths_df, [OGU_ID_KEY, OGU_LEN_IN_BP_KEY],
        "OGU lengths are missing required column(s)")

    validate_required_columns_exist(
        ogu_percent_coverage_df, [OGU_ID_KEY, OGU_PERCENT_COVERAGE_KEY],
        "OGU percent coverage is missing required column(s)")

    # Check if any samples in the reads data are missing from the metadata;
    # Not bothering to report samples that are in metadata but not the reads--
    # maybe those failed the sequencing run.
    _ = validate_metadata_vs_reads_id_consistency(
        absolute_quant_params_per_sample_df, ogu_counts_per_sample_biom)

    # Check if there are any OGUs in the read data that are missing from the
    # OGU lengths data; if so, throw an error


    working_params_df = absolute_quant_params_per_sample_df.copy()

    # cast the GDNA_CONCENTRATION_NG_UL_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
    # ELUTE_VOL_UL_KEY, and SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY columns of
    # params df to float if they aren't already
    float_col_names = [
        GDNA_CONCENTRATION_NG_UL_KEY, SAMPLE_IN_ALIQUOT_MASS_G_KEY,
        ELUTE_VOL_UL_KEY, SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY]
    working_params_df = cast_cols(working_params_df, float_col_names, True)

    # cast SAMPLE_TOTAL_READS_KEY column of params df to numeric if it isn't
    working_params_df = \
        cast_cols(working_params_df, [SAMPLE_TOTAL_READS_KEY])

    # Remove from input biom any samples that have bad params
    cols_to_filter_on = required_cols_list.copy()
    cols_to_filter_on.remove(SAMPLE_ID_KEY)  # don't filter on sample id
    # TODO: this is a hack; I'm not sure what effect setting the index to
    #  SAMPLE_ID_KEY on the "real" df would have, and I don't have time to
    #  vet it, so I'm just making a copy and setting the index on that.
    filter_params_df = working_params_df.copy()
    filter_params_df.set_index(SAMPLE_ID_KEY, inplace=True)
    filtered_ogu_counts_per_sample_biom, log_msgs_list = \
        filter_data_by_sample_info(
            filter_params_df, ogu_counts_per_sample_biom, cols_to_filter_on)

    # calculate the ratio of extracted gDNA mass to sample mass put into
    # extraction for each sample
    gdna_mass_to_sample_mass_by_sample_series = \
        _calc_gdna_mass_to_sample_mass_by_sample_df(working_params_df)
    per_sample_calc_info_df = _series_to_df(
        gdna_mass_to_sample_mass_by_sample_series, SAMPLE_ID_KEY,
        GDNA_MASS_TO_SAMPLE_MASS_RATIO_KEY)

    # merge the SAMPLE_TOTAL_READS_KEY and SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY
    # columns of working_params_df into gdna_mass_to_sample_mass_df
    # by SAMPLE_ID_KEY
    per_sample_calc_info_df = per_sample_calc_info_df.merge(
        working_params_df[[SAMPLE_ID_KEY, SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY,
                           SAMPLE_TOTAL_READS_KEY]],
        on=SAMPLE_ID_KEY, how='left')

    # convert input biom table to a dataframe with sparse columns, which
    # should act basically the same as a dense dataframe but use less memory
    ogu_counts_per_sample_df = \
        filtered_ogu_counts_per_sample_biom.to_dataframe(dense=False)

    ogu_cell_counts_long_format_df, calc_log_msgs_list = (
        _calc_long_format_ogu_cell_counts_df(
            linregress_by_sample_id, ogu_counts_per_sample_df,
            ogu_lengths_df, per_sample_calc_info_df, read_length,
            min_coverage, min_rsquared))
    log_msgs_list.extend(calc_log_msgs_list)

    ogu_cell_counts_wide_format_df = ogu_cell_counts_long_format_df.pivot(
        index=OGU_ID_KEY, columns=SAMPLE_ID_KEY)[output_cell_counts_metric]

    # replace NaNs with 0s; per Daniel McDonald, much downstream analysis
    # cannot handle NaNs, and it is preferable to set invalid values
    # to 0 and provide a log message saying they are not usable than to leave
    # them as NaNs
    ogu_cell_counts_wide_format_df.fillna(0, inplace=True)

    # convert dataframe to biom table; input params are
    # data (the "output_cell_count_metric"s), observation_ids (the "ogu_id"s),
    # and sample_ids (er, the "sample_id"s)
    ogu_cell_counts_biom = biom.Table(
        ogu_cell_counts_wide_format_df.values,
        ogu_cell_counts_wide_format_df.index,
        ogu_cell_counts_wide_format_df.columns)

    return ogu_cell_counts_biom, log_msgs_list


def calc_ogu_cell_counts_per_g_of_sample_for_qiita(
        sample_info_df: pd.DataFrame,
        prep_info_df: pd.DataFrame,
        linregress_by_sample_id_fp: str,
        ogu_counts_per_sample_biom: biom.Table,
        ogu_lengths_fp: str,
        read_length: int = DEFAULT_READ_LENGTH,
        min_coverage: float = DEFAULT_MIN_PERCENT_COVERAGE,
        min_rsquared: float = DEFAULT_MIN_RSQUARED,
        syndna_mass_fraction_of_sample: float =
        DEFAULT_SYNDNA_MASS_FRACTION_OF_SAMPLE) \
        -> Dict[str, Union[str, biom.Table]]:

    """Gets # of cells of each OGU/g of sample for samples from Qiita.

    Parameters
    ----------
    sample_info_df: pd.DataFrame
        A Dataframe containing sample info for all samples in the prep,
        including SAMPLE_ID_KEY and SAMPLE_IN_ALIQUOT_MASS_G_KEY
    prep_info_df: pd.DataFrame
        A Dataframe containing prep info for all samples in the prep,
        including SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
        ELUTE_VOL_UL_KEY, SYNDNA_POOL_MASS_NG_KEY, and SAMPLE_TOTAL_READS_KEY.
    linregress_by_sample_id_fp: str
        String containing the filepath to the yaml file holding the
        dictionary keyed by sample id, containing for each sample a dictionary
        representation of the sample's LinregressResult.
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_lengths_fp : str
        String containing the filepath to a tab-separated, two-column,
        no-header file in which the first column is the OGU id and the
         second is the OGU length in basepairs
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable % coverage of an OGU in a sample needed to include
        that OGU/sample in the output.
    min_rsquared: float
        Minimum allowable R^2 value for the linear regression model for a
        sample; any sample with an R^2 value less than this will be excluded
        from the output.
    syndna_mass_fraction_of_sample: float
        Fraction of the mass of the sample that is added as syndna (usually
        0.05, which is to say 5%).

    Returns
    -------
    output_by_out_type : dict of str or biom.Table
        Dictionary of outputs keyed by their type Currently, the following keys
        are defined:
        CELL_COUNT_RESULT_KEY: biom.Table holding the calculated number of
        cells per gram of sample material for each OGU in each sample.
        CELL_COUNT_LOG_KEY: log of messages from the cell count calc process.
    """

    # check if the inputs all have the required columns
    validate_required_columns_exist(
        sample_info_df, REQUIRED_SAMPLE_INFO_KEYS,
        "sample info is missing required column(s)")

    required_prep_cols = list(
        {SYNDNA_POOL_MASS_NG_KEY} | set(REQUIRED_DNA_PREP_INFO_KEYS))
    validate_required_columns_exist(
        prep_info_df, required_prep_cols,
        "prep info is missing required column(s)")

    # Check if any samples in the prep are missing from the sample info;
    # Not bothering to report samples that are in sample info but not the prep
    # --maybe those just weren't included in this prep.
    _ = validate_metadata_vs_prep_id_consistency(
        sample_info_df, prep_info_df)

    # cast in case the input comes in as string or something
    syndna_mass_fraction_of_sample = float(syndna_mass_fraction_of_sample)

    # make sure the SYNDNA_POOL_MASS_NG_KEY column of prep_info_df is a float,
    # then calculate the mass of gDNA sequenced for each sample.  We have the
    # mass of syndna pool that was added to each sample, and we know that the
    # syndna pool mass is calculated to be a certain percentage of the mass of
    # the sample (added into the library prep in addition to the sample mass).
    # Therefore, if the syndna fraction is 0.05 or 5%, the mass of the sample
    # gDNA put into sequencing is 1/0.05 = 20x the mass of syndna pool added.
    prep_info_df = cast_cols(prep_info_df, [SYNDNA_POOL_MASS_NG_KEY], True)
    prep_info_df[SEQUENCED_SAMPLE_GDNA_MASS_NG_KEY] = \
        prep_info_df[SYNDNA_POOL_MASS_NG_KEY] * \
        (1 / syndna_mass_fraction_of_sample)

    # merge the sample info and prep info dataframes
    absolute_quant_params_per_sample_df = \
        sample_info_df.merge(prep_info_df, on=SAMPLE_ID_KEY, how='left')

    # read in the linregress_by_sample_id yaml file
    with open(linregress_by_sample_id_fp) as f:
        linregress_by_sample_id = yaml.load(f, Loader=yaml.FullLoader)

    # read in the ogu_lengths file
    ogu_lengths_df = pd.read_csv(ogu_lengths_fp, sep='\t', header=None,
                                 names=[OGU_ID_KEY, OGU_LEN_IN_BP_KEY])

    # calculate # cells per gram of sample material of each OGU in each sample
    output_biom, log_msgs_list = calc_ogu_cell_counts_biom(
        absolute_quant_params_per_sample_df, linregress_by_sample_id,
        ogu_counts_per_sample_biom, ogu_lengths_df, read_length, min_coverage,
        min_rsquared, OGU_CELLS_PER_G_OF_SAMPLE_KEY)

    out_txt_by_out_type = {
        CELL_COUNT_RESULT_KEY: output_biom,
        CELL_COUNT_LOG_KEY: '\n'.join(log_msgs_list)}

    return out_txt_by_out_type
