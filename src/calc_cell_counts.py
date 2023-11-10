import biom
import numpy as np
import pandas as pd
import scipy
import yaml

from fit_syndna_models import SAMPLE_ID_KEY

DEFAULT_READ_LENGTH = 150
DEFAULT_MIN_COVERAGE = 1

GDNA_CONCENTRATION_NG_UL_KEY = 'minipico_dna_concentration_ng_ul'
SAMPLE_IN_ALIQUOT_MASS_G_KEY = 'calc_mass_sample_aliquot_input_g'
ELUTE_VOL_UL_KEY = 'ELUTE_VOL_UL_KEY'
SAMPLE_CONCENTRATION_NG_UL_KEY = 'sample_concentration_ng_ul'
OGU_ID_KEY = 'ogu_id'
OGU_LEN_IN_BP_KEY = 'ogu_len_in_bp'
OGU_GDNA_MASS_NG_KEY = 'ogu_gdna_mass_ng'
OGU_CELLS_PER_G_OF_GDNA_KEY = 'ogu_cells_per_g_of_gdna'


def get_ogu_cell_counts_per_g_of_sample_for_qiita(
        sample_info_df: pd.DataFrame,
        prep_info_df: pd.DataFrame,
        linregress_by_sample_id_fp: str,
        ogu_counts_per_sample_biom: biom.Table,
        ogu_lengths_fp: str,
        read_length: int = DEFAULT_READ_LENGTH,
        min_coverage: float = DEFAULT_MIN_COVERAGE) -> biom.Table:

    """Gets # of cells of each OGU/g of sample for samples from Qiita.

    Parameters
    ----------
    sample_info_df: pd.DataFrame
        Dataframe containing sample info for all samples in the prep,
        including SAMPLE_ID_KEY and SAMPLE_IN_ALIQUOT_MASS_G_KEY
    prep_info_df: pd.DataFrame
        Dataframe containing prep info for all samples in the prep,
        including SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
        and ELUTE_VOL_UL_KEY
    linregress_by_sample_id_fp: str
        String containing the filepath to the yaml file holding the
        dictionary of scipy.stats.LinregressResult objects keyed by sample id
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_lengths_fp : str
        String containing the filepath to a tab-separated, two-column,
        no-header file in which the first column is the OGU id and the
         second is the OGU length in basepairs
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable coverage of an OGU needed to include that OGU
        in the output.

    Returns
    -------
    ogu_cell_counts_per_g_of_sample_biom : biom.Table
        Biom table with a column for OGU_ID_KEY and then one additional column
        for each sample id, which holds the calculated number of cells per gram
        of sample material of that OGU in that sample.
    """

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
    return _generate_ogu_cell_counts_biom(
        absolute_quant_params_per_sample_df, linregress_by_sample_id,
        ogu_counts_per_sample_biom, ogu_lengths_df, read_length, min_coverage,
        OGU_CELLS_PER_G_OF_GDNA_KEY)


def _generate_ogu_cell_counts_biom(
        absolute_quant_params_per_sample_df: pd.DataFrame,
        linregress_by_sample_id: dict[str, scipy.stats.LinregressResult],
        ogu_counts_per_sample_biom: biom.Table,
        ogu_lengths_df: pd.DataFrame,
        read_length: int,
        min_coverage: float,
        output_cell_counts_metric: str) -> biom.Table:

    """Uses linear models to get cell # per g of sample, for each ogu & sample.

    Parameters
    ----------
    absolute_quant_params_per_sample_df:  pd.DataFrame
        Dataframe of at least SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
        SAMPLE_IN_ALIQUOT_MASS_G_KEY, and ELUTE_VOL_UL_KEY for
        each sample.
    linregress_by_sample_id : dict[str, scipy.stats.LinregressResult]
        Dictionary keyed by sample_id.  Dictionary values are either None
        (if no model could be trained for that SAMPLE_ID_KEY) or a
        scipy.stats.LinregressResult object defining the trained model.
    ogu_counts_per_sample_biom: biom.Table
        Biom table holding the read counts aligned to each OGU in each sample.
    ogu_lengths_df : pd.DataFrame
        Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable coverage of an OGU needed to include that OGU
        in the output.
    output_cell_counts_metric : str
        Name of the desired output cell count metric; options are
        'ogu_cells_per_g_of_gdna' and 'ogu_cells_per_g_of_stool'.

    Returns
    -------
    ogu_cell_counts_biom : biom.Table
        Dataframe with a column for OGU_ID_KEY and then one additional column
        for each sample id, which holds the predicted number of cells per gram
        of sample material of that OGU in that sample.
    """

    # calculate the ratio of gDNA mass to sample mass for each sample
    gdna_mass_to_sample_mass_by_sample_series = \
        _calc_gdna_mass_to_sample_mass_by_sample_df(
            absolute_quant_params_per_sample_df)
    gdna_mass_to_sample_mass_df = _series_to_df(
        gdna_mass_to_sample_mass_by_sample_series, 'gdna_mass_to_sample_mass')

    # convert input biom table to a pd.SparseDataFrame, which is should act
    # basically like a pd.DataFrame but take up less memory
    ogu_counts_per_sample_df = ogu_counts_per_sample_biom.to_dataframe(
        dense=False)

    ogu_cell_counts_long_format_df = _calc_long_format_ogu_cell_counts_df(
        linregress_by_sample_id, ogu_counts_per_sample_df,
        ogu_lengths_df, gdna_mass_to_sample_mass_df, read_length, min_coverage)

    ogu_cell_counts_wide_format_df = ogu_cell_counts_long_format_df.pivot(
        index=OGU_ID_KEY, columns=SAMPLE_ID_KEY)[output_cell_counts_metric]

    # convert dataframe to biom table; input params are
    # data (the "output_cell_count_metric"s), observation_ids (the "ogu_id"s),
    # and sample_ids (er, the "sample_id"s)
    ogu_cell_counts_biom = biom.Table(
        ogu_cell_counts_wide_format_df.values,
        ogu_cell_counts_wide_format_df.index,
        ogu_cell_counts_wide_format_df.columns)

    return ogu_cell_counts_biom


def _calc_gdna_mass_to_sample_mass_by_sample_df(
        absolute_quant_params_per_sample_df: pd.DataFrame) -> pd.Series:

    """Calculates the ratio of gDNA mass to sample mass for each sample.

    Parameters
    ----------
    absolute_quant_params_per_sample_df:  pd.DataFrame
        Dataframe of at least SAMPLE_ID_KEY, GDNA_CONCENTRATION_NG_UL_KEY,
        SAMPLE_IN_ALIQUOT_MASS_G_KEY, and ELUTE_VOL_UL_KEY for
        each sample.

    Returns
    -------
    gdna_mass_to_sample_mass_by_sample_series : pd.Series
        Series with index of sample id and values of the ratio of gDNA mass
        units extracted from each mass unit of input sample (only) mass.
    """

    working_df = absolute_quant_params_per_sample_df.copy()
    # get the ngs of *sample* material that are represented by each uL of
    # elute after extraction; this is sample-specific:
    # (mass of sample material (only) in g that went into the extraction
    # times 10^6 ng/g) divided by (volume of elute from the extraction in uL)
    working_df[SAMPLE_CONCENTRATION_NG_UL_KEY] = \
        (working_df[SAMPLE_IN_ALIQUOT_MASS_G_KEY] * 10 ** 6) / \
        working_df[ELUTE_VOL_UL_KEY]

    # determine how many mass units of gDNA are produced from the extraction of
    # each mass unit of sample material; this is sample-specific:
    # ngs/uL of gDNA after extraction divided by ngs/uL of sample material
    # Note that the units cancel out, leaving a unitless ratio, so it is true
    # for any mass unit of interest (e.g., if the ratio is X, there are
    # X grams of gDNA produced from the extraction of 1 gram of sample material
    # and, likewise, X ng of gDNA produced from the extraction of 1 ng of
    # sample material, etc.)

    gdna_mass_to_sample_mass_ratio = \
        working_df[GDNA_CONCENTRATION_NG_UL_KEY] / \
        working_df[SAMPLE_CONCENTRATION_NG_UL_KEY]

    return gdna_mass_to_sample_mass_ratio


# TODO: probably need to pass *two* col names and change index handling
def _series_to_df(a_series, col_name):
    """Converts a pd.Series to two-column pd.DataFrame (from index and value)

    Parameters
    ----------
    a_series : pd.Series
        Series to be converted to a dataframe.
    col_name : str
        Name of the column in the resulting dataframe.

    Returns
    -------
    a_df : pd.DataFrame
        Dataframe with a single column, named col_name, containing the values
        from the input series.
    """

    a_df = a_series.to_frame().reset_index()
    a_df = a_df.rename(columns={0: col_name})

    return a_df


def _calc_long_format_ogu_cell_counts_df(
        linregress_by_sample_id: dict[str, scipy.stats.LinregressResult],
        ogu_counts_per_sample_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        gdna_mass_to_sample_mass_by_sample_df: pd.DataFrame,
        read_length: int,
        min_coverage: float) -> pd.DataFrame:

    """Applies per-sample linear regression models to predict the number of
    cells of each OGU in each sample from the read counts for that OGU in that
    sample.

    Parameters
    ----------
    linregress_by_sample_id : dict[str, scipy.stats.LinregressResult]
        Dictionary keyed by sample id.  Dictionary values are either None
        (if no model could be trained for that sample id) or a
        scipy.stats.LinregressResult object defining the trained model.
    ogu_counts_per_sample_df: pd.DataFrame
        Dataframe with a column for OGU_ID_KEY and then one additional column
        for each sample id, which holds the read counts aligned to that OGU in
        that sample.
    ogu_lengths_df : pd.DataFrame
        Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    gdna_mass_to_sample_mass_by_sample_df : pd.DataFrame
        Dataframe of SAMPLE_ID_KEY and gdna_mass_to_sample_mass_ratio for each
        sample.
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable coverage of an OGU needed to include that OGU
        in the output.

    Returns
    -------
    ogu_cell_counts_df : pd.DataFrame
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        ogu_cells_per_g_of_gdna, and ogu_cells_per_g_of_stool, in addition
        to various intermediate calculation columns.
    """

    # reformat biom info into a "long format" table with
    # columns needed for per-sample calculation
    working_df = _prepare_cell_counts_calc_df(
        ogu_counts_per_sample_df, ogu_lengths_df, read_length, min_coverage)

    # loop through a series of the unique sample ids in the working_df
    cell_counts_df = None
    sample_ids = working_df[SAMPLE_ID_KEY].unique()
    for curr_sample_id in sample_ids:
        # calculate the predicted number of cells of each OGU per gram of
        # gDNA in this sample and also per gram of stool in this sample
        curr_sample_df = _calc_ogu_cell_counts_df_for_sample(
            curr_sample_id, linregress_by_sample_id,
            gdna_mass_to_sample_mass_by_sample_df, working_df)

        # TODO: what if no cell counts could be calculated for this sample?
        if curr_sample_df is None:
            continue

        # if cell_counts_df does not yet exist, create it from curr_sample_df;
        # otherwise, append curr_sample_df to the existing cell_counts_df
        if cell_counts_df is None:
            cell_counts_df = curr_sample_df
        else:
            cell_counts_df = cell_counts_df.append(curr_sample_df)
    # next sample_id

    return cell_counts_df


def _prepare_cell_counts_calc_df(
        ogu_counts_per_sample_df: pd.DataFrame,
        ogu_lengths_df: pd.DataFrame,
        read_length: int,
        min_coverage: float) -> pd.DataFrame:

    """Prepares long-format dataframe containing fields needed for later calcs.

    Parameters
    ----------
    ogu_counts_per_sample_df: pd.DataFrame
        Dataframe with a column for OGU_ID_KEY and then one additional column
        for each sample id, which holds the read counts aligned to that OGU in
        that sample.
    ogu_lengths_df : pd.DataFrame
        Dataframe of OGU_ID_KEY and OGU_LEN_IN_BP_KEY for each OGU.
    read_length : int
        Length of reads in bp (usually but not always 150).
    min_coverage : float
        Minimum allowable coverage of an OGU needed to include that OGU
        in the output.  Zaramela paper uses 1.

    Returns
    -------
    working_df : pd.DataFrame
        Long-format dataframe with columns for OGU_ID_KEY, SAMPLE_ID_KEY,
        ogu_read_count, total_reads_per_ogu, OGU_LEN_IN_BP_KEY, coverage_of_ogu;
        contains rows for OGU/sample combinations with coverage >= min_coverage
    """

    # calculate the total number of reads per OGU by summing up the read counts
    # for each OGU across all samples (i.e., summing each row of biom table)
    total_ogu_counts_series = ogu_counts_per_sample_df.sum(axis=1)
    total_ogu_counts_df = _series_to_df(
        total_ogu_counts_series, 'total_reads_per_ogu')

    # reformat biom info into a "long format" table with
    # columns for OGU_ID_KEY, SAMPLE_ID_KEY, ogu_read_count
    working_df = ogu_counts_per_sample_df.melt(
        id_vars=[OGU_ID_KEY], var_name=SAMPLE_ID_KEY, value_name='ogu_read_count')

    # add a column for OGU_LEN_IN_BP_KEY (yes, this will be repeated, but it is
    # convenient to have everything in one table)
    working_df = working_df.merge(ogu_lengths_df, on=OGU_ID_KEY, how='left')

    # add total reads per OGU column to table (again, will be redundancies)
    working_df = working_df.merge(total_ogu_counts_df, on=OGU_ID_KEY, how='left')

    # calculate the coverage per OGU per sample by multiplying each
    # read_count cell value by the number of bases in the read (read_length)
    # and then dividing by the ogu_length for that OGU
    working_df['coverage_of_ogu'] = (
            (working_df['ogu_read_count'] * read_length) /
            working_df[OGU_LEN_IN_BP_KEY])

    # drop records for OGU/sample combinations with coverage < min_coverage
    working_df = working_df[working_df['coverage_of_ogu'] >= min_coverage]
    # TODO: do the failed entries get recorded somewhere?

    return working_df


def _calc_ogu_cell_counts_df_for_sample(
        sample_id: str,
        linregress_by_sample_id: dict[str, scipy.stats.LinregressResult],
        gdna_mass_to_sample_mass_by_sample_df: pd.DataFrame,
        working_df: pd.DataFrame) -> pd.DataFrame | None:

    # TODO: add docstring

    # get the linear regression result for this sample
    linregress_result = linregress_by_sample_id[sample_id]
    if linregress_result is None:
        # TODO: what to do if there is no linregress result for that sample
        #  (i.e., if the regression failed)?
        return None

    # TODO: Should there be a lower limit on R^2 of the linregress result?
    #  Should we refuse to use a model with "too low" an R^2?

    # get df of the rows of the working_df specific to this sample
    sample_df = working_df[
        working_df[SAMPLE_ID_KEY] == sample_id]

    # predict mass of each OGU's gDNA in this sample using the linear model
    sample_df[OGU_GDNA_MASS_NG_KEY] = \
        _calc_ogu_gdna_mass_ng_series_for_sample(
            sample_df, linregress_result.slope,
            linregress_result.intercept)

    # calc the # of genomes of each OGU per gram of gDNA in this sample
    sample_df['ogu_genomes_per_g_of_gdna'] = \
        _calc_ogu_genomes_per_g_of_gdna_series_for_sample(sample_df)

    # assume the # of cells of the microbe represented by each OGU
    # per gram of gDNA in this sample is the same as the number of genomes
    # of that OGU per gram of gDNA. This is known to be not quite right
    # (since some cells are dividing and thus have an extra copy, and also
    # because apparently polyploidy is not uncommon among microbes), but
    # that's why these things are called "simplifying assumptions" ...
    sample_df['ogu_cells_per_g_of_gdna'] = \
        sample_df['ogu_genomes_per_g_of_gdna']

    # TODO: don't think this will always be stool; what should I name this?
    # calc the # of cells of each OGU per gram of *stool* in this sample
    sample_df['ogu_cells_per_g_of_stool'] = \
        sample_df['ogu_cells_per_g_of_gdna'] * \
        gdna_mass_to_sample_mass_by_sample_df.loc[
            sample_id, 'gdna_mass_to_sample_mass_ratio']

    return sample_df


def _calc_ogu_gdna_mass_ng_series_for_sample(
        sample_df: pd.DataFrame,
        sample_linregress_slope: float,
        sample_linregress_intercept: float) -> pd.Series:

    """Calculates mass of OGU gDNA in ng for each OGU in a sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        Dataframe with rows for a single sample, containing *at least* columns
        for OGU_ID_KEY, OGU_LEN_IN_BP_KEY, and OGU_GDNA_MASS_NG_KEY.
    sample_linregress_slope: float
        Slope of the linear regression model for the sample.
    sample_linregress_intercept: float
        Intercept of the linear regression model for the sample.

    Returns
    -------
    ogu_genomes_per_g_of_gdna_series : pd.Series
        Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU per gram of gDNA in the sample.
    """

    # calculate the total number of reads for this sample (a scalar)
    # by summing read counts for all the rows in the sample table
    # TODO: double-check axis here
    total_reads_per_sample = sample_df['ogu_read_count'].sum()

    # add a column of counts per million (CPM) for each ogu by dividing
    # each read_count by the total number of reads for this sample
    # and then multiplying by a million (1,000,000)
    # TODO: do I need to worry about type here, dividing int/int and
    #  wanting to get a float?
    sample_df['ogu_CPM'] = (sample_df['ogu_read_count'] /
                            total_reads_per_sample) * 1000000

    # add column of log10(ogu_CPM) by taking log base 10 of the ogu_CPM column
    sample_df['log10_ogu_CPM'] = np.log10(sample_df['ogu_CPM'])

    # calculate log10_ogu_gdna_mass_ng of each OGU's gDNA in this sample
    # by multiplying each OGU's log10_ogu_CPM by the slope of this sample's
    # regression model and adding the model's intercept.
    # NB: this requires that the linear regression models were derived
    # using synDNA masses *in ng* and not in some other unit.
    sample_df['log10_ogu_gdna_mass_ng'] = (
            sample_df['log10_ogu_CPM'] *
            sample_linregress_slope +
            sample_linregress_intercept)

    # calculate the actual mass in ng of each OGU's gDNA by raising 10 to the
    # log10_ogu_gdna_mass_ng power
    ogu_gdna_mass_ng_series = \
        10 ** sample_df['log10_ogu_gdna_mass_ng']

    return ogu_gdna_mass_ng_series


def _calc_ogu_genomes_per_g_of_gdna_series_for_sample(
        sample_df: pd.DataFrame) -> pd.Series:

    """Calculates # of OGU genomes per gram of gDNA for each OGU in a sample.

    Parameters
    ----------
    sample_df: pd.DataFrame
        Dataframe with rows for a single sample, containing *at least* columns
        for OGU_ID_KEY, OGU_LEN_IN_BP_KEY, and OGU_GDNA_MASS_NG_KEY.

    Returns
    -------
    ogu_genomes_per_g_of_gdna_series : pd.Series
        Series with index of OGU_ID_KEY and values of the number of genomes
        of each OGU per gram of gDNA in the sample.

    This calculates the total number of genomes for each OGU in the sample
    by the equation:

        mass of OGU's gDNA in ng * Avogadro's number in genomes/mole
    =	---------------------------------------------------------------
        length of OGU genome in basepairs *
         650 g/mole per basepair (on average) * 10^9 ng/g

    = a result in units of genomes of OGU/g of gdna
    ~= a result in units of cells of microbe/g of gdna

    NB: the constant factor on the bottom right MUST CHANGE if the
    units of the OGU gDNA mass are NOT nanograms!

    Avogadro's number is 6.02214076 × 10^23 , and is the number of
    molecules--in this case, genomes--in a mole of a substance.
    """

    # TODO: do we have to worry about integer overflow here?
    #  Dan H. said, "if you use ints, the length * 650 * 10^9
    #  can overflow integers with very long genomes".  HOWEVER,
    #  the internet says that python *3* , "[o]nly floats have a hard
    #  limit in python. Integers are implemented as “long” integer
    #  objects of arbitrary size"(https://stackoverflow.com/a/52151786)
    #  HOWEVER HOWEVER, *numpy* integer types are fixed width, and
    #  "Some pandas and numpy functions, such as sum on arrays or
    #  Series return an np.int64 so this might be the reason you are
    #  seeing int overflows in Python3."
    #  (https://stackoverflow.com/a/58640340)
    #  What to do?

    ogu_genomes_per_g_of_gdna_series = (
            (sample_df[OGU_GDNA_MASS_NG_KEY] * 6.02214076e23) /
            (sample_df[OGU_LEN_IN_BP_KEY] * 650 * 1e9))

    return ogu_genomes_per_g_of_gdna_series
